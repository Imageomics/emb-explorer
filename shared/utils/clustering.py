from typing import Optional, Tuple
import os
import sys
import subprocess
import tempfile
import time
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Optional FAISS support for faster clustering
try:
    import faiss
    HAS_FAISS = True
    logger.debug("FAISS available")
except ImportError:
    HAS_FAISS = False
    logger.debug("FAISS not available")

# Optional cuML support for GPU acceleration
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    from cuml.manifold import TSNE as cuTSNE
    from cuml.manifold import UMAP as cuUMAP
    import cupy as cp
    HAS_CUML = True
    logger.debug("cuML available")
except ImportError:
    HAS_CUML = False
    logger.debug("cuML not available")

# Check for CUDA availability
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    try:
        import cupy as cp
        HAS_CUDA = cp.cuda.is_available()
    except ImportError:
        HAS_CUDA = False

logger.debug(f"CUDA available: {HAS_CUDA}")


class VRAMExceededError(Exception):
    """Raised when GPU VRAM is exceeded during computation."""
    pass


class GPUArchitectureError(Exception):
    """Raised when GPU architecture is not supported."""
    pass


def get_gpu_memory_info() -> Optional[Tuple[int, int]]:
    """
    Get GPU memory info (used, total) in MB.

    Returns:
        Tuple of (used_mb, total_mb) or None if unavailable.
    """
    try:
        if HAS_CUML and HAS_CUDA:
            meminfo = cp.cuda.Device().mem_info
            free_bytes, total_bytes = meminfo
            used_bytes = total_bytes - free_bytes
            return (used_bytes // (1024 * 1024), total_bytes // (1024 * 1024))
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() // (1024 * 1024)
            total = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            return (used, total)
    except Exception:
        pass

    return None


def estimate_memory_requirement(n_samples: int, n_features: int, method: str) -> int:
    """
    Estimate memory requirement in MB for dimensionality reduction.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        method: Reduction method (PCA, TSNE, UMAP)

    Returns:
        Estimated memory in MB
    """
    # Base memory for input data (float32)
    base_mb = (n_samples * n_features * 4) / (1024 * 1024)

    # Method-specific multipliers (empirical estimates)
    if method.upper() == "PCA":
        return int(base_mb * 2)  # Relatively low overhead
    elif method.upper() == "TSNE":
        return int(base_mb * 4 + (n_samples * n_samples * 4) / (1024 * 1024))  # Distance matrix
    elif method.upper() == "UMAP":
        return int(base_mb * 3 + (n_samples * 15 * 4) / (1024 * 1024))  # kNN graph
    else:
        return int(base_mb * 3)

def _prepare_embeddings(embeddings: np.ndarray, operation: str) -> np.ndarray:
    """Validate, cast to float32, and L2-normalize embeddings.

    L2 normalization projects vectors onto the unit hypersphere (magnitude 1).
    This stabilises cuML's NN-descent (prevents SIGFPE from large magnitudes)
    and is appropriate for contrastive-model embeddings (e.g. CLIP, BioCLIP)
    whose training objective is cosine-similarity based.

    Args:
        embeddings: Raw embedding matrix (n_samples, n_features).
        operation: Label for log messages (e.g. "reduce_dim", "kmeans").

    Returns:
        L2-normalized float32 embedding matrix.
    """
    n_samples, n_features = embeddings.shape

    # Cast to float32
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # Check for non-finite values
    n_nonfinite = (~np.isfinite(embeddings)).sum()
    if n_nonfinite > 0:
        logger.warning(f"[{operation}] {n_nonfinite} non-finite values found, replacing with 0")
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    n_zero = (norms.ravel() < 1e-10).sum()
    if n_zero > 0:
        logger.warning(f"[{operation}] {n_zero} near-zero-norm vectors found (will clamp to avoid division by zero)")
    embeddings = embeddings / np.maximum(norms, 1e-10)

    logger.info(f"[{operation}] Prepared embeddings: {n_samples} samples, {n_features} features, "
                f"dtype=float32, L2-normalized "
                f"(input norms: min={norms.min():.2f}, max={norms.max():.2f}, mean={norms.mean():.2f})")
    return embeddings


def reduce_dim(embeddings: np.ndarray, method: str = "PCA", seed: Optional[int] = None, n_workers: int = 1, backend: str = "auto"):
    """
    Reduce the dimensionality of embeddings to 2D using PCA, t-SNE, or UMAP.

    Args:
        embeddings (np.ndarray): The input feature embeddings of shape (n_samples, n_features).
        method (str, optional): The dimensionality reduction method, "PCA", "TSNE", or "UMAP". Defaults to "PCA".
        seed (int, optional): Random seed for reproducibility. Defaults to None (random).
        n_workers (int, optional): Number of parallel workers for t-SNE/UMAP. Defaults to 1.
        backend (str, optional): Backend to use - "auto", "sklearn", "cuml". Defaults to "auto".

    Returns:
        np.ndarray: The 2D reduced embeddings of shape (n_samples, 2).

    Raises:
        ValueError: If an unsupported method is provided.
    """
    n_samples, n_features = embeddings.shape
    logger.info(f"Dimensionality reduction: method={method}, samples={n_samples}, features={n_features}, backend={backend}")

    # Validate, cast, and L2-normalize
    embeddings = _prepare_embeddings(embeddings, "reduce_dim")

    # Determine which backend to use
    use_cuml = False
    if backend == "cuml" and HAS_CUML and HAS_CUDA:
        use_cuml = True
    elif backend == "auto" and HAS_CUML and HAS_CUDA and n_samples > 5000:
        # Use cuML automatically for large datasets on GPU
        use_cuml = True

    start_time = time.time()
    if use_cuml:
        logger.info(f"Using cuML backend for {method}")
        result = _reduce_dim_cuml(embeddings, method, seed, n_workers)
    else:
        logger.info(f"Using sklearn backend for {method}")
        result = _reduce_dim_sklearn(embeddings, method, seed, n_workers)

    elapsed = time.time() - start_time
    logger.info(f"Dimensionality reduction completed in {elapsed:.2f}s")
    return result


def _reduce_dim_sklearn(embeddings: np.ndarray, method: str, seed: Optional[int], n_workers: int):
    """Dimensionality reduction using sklearn/umap backends."""
    # Use -1 (all available cores) instead of specific values > 1 to avoid
    # thread count restrictions on HPC clusters (OMP_NUM_THREADS, SLURM cgroups)
    effective_workers = -1 if n_workers > 1 else n_workers

    if method.upper() == "PCA":
        reducer = PCA(n_components=2)
    elif method.upper() == "TSNE":
        # Adjust perplexity to be valid for the sample size
        n_samples = embeddings.shape[0]
        perplexity = min(30, max(5, n_samples // 3))  # Ensure perplexity is reasonable

        if seed is not None:
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_jobs=effective_workers)
        else:
            reducer = TSNE(n_components=2, perplexity=perplexity, n_jobs=effective_workers)
    elif method.upper() == "UMAP":
        # Adjust n_neighbors to be valid for the sample size
        n_samples = embeddings.shape[0]
        n_neighbors = min(15, max(2, n_samples - 1))

        if seed is not None:
            reducer = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=seed, n_jobs=effective_workers)
        else:
            reducer = UMAP(n_components=2, n_neighbors=n_neighbors, n_jobs=effective_workers)
    else:
        raise ValueError("Unsupported method. Choose 'PCA', 'TSNE', or 'UMAP'.")
    return reducer.fit_transform(embeddings)


def _reduce_dim_cuml(embeddings: np.ndarray, method: str, seed: Optional[int], n_workers: int):
    """Dimensionality reduction using cuML GPU backends.

    Expects embeddings to already be L2-normalized float32 from _prepare_embeddings().
    """
    try:
        if method.upper() == "UMAP":
            # cuML UMAP can crash with SIGFPE on certain data distributions
            # (NN-descent numerical instability).  SIGFPE is a signal, not a
            # Python exception, so try/except cannot catch it.  Run in an
            # isolated subprocess so the main process (Streamlit) survives.
            return _run_cuml_umap_subprocess(embeddings, seed)

        # PCA and TSNE are stable — run in-process
        embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)

        if method.upper() == "PCA":
            reducer = cuPCA(n_components=2)
        elif method.upper() == "TSNE":
            n_samples = embeddings.shape[0]
            perplexity = min(30, max(5, n_samples // 3))

            if seed is not None:
                reducer = cuTSNE(n_components=2, perplexity=perplexity, random_state=seed)
            else:
                reducer = cuTSNE(n_components=2, perplexity=perplexity)
        else:
            raise ValueError("Unsupported method. Choose 'PCA', 'TSNE', or 'UMAP'.")

        result_gpu = reducer.fit_transform(embeddings_gpu)
        return cp.asnumpy(result_gpu)

    except RuntimeError as e:
        error_msg = str(e).lower()
        if "no kernel image" in error_msg or "cudaerrornokernel" in error_msg:
            logger.warning(f"cuML {method} not supported on this GPU architecture, falling back to sklearn")
        else:
            logger.warning(f"cuML reduction failed ({e}), falling back to sklearn")
        return _reduce_dim_sklearn(embeddings, method, seed, n_workers)
    except Exception as e:
        logger.warning(f"cuML reduction failed ({e}), falling back to sklearn")
        return _reduce_dim_sklearn(embeddings, method, seed, n_workers)


# Standalone script executed in a subprocess for cuML UMAP.
# Kept minimal: only imports cuml/cupy/numpy, no project dependencies.
_CUML_UMAP_SCRIPT = """\
import sys, numpy as np, cupy as cp
from cuml.manifold import UMAP as cuUMAP

input_path, output_path = sys.argv[1], sys.argv[2]
n_neighbors = int(sys.argv[3])
seed = int(sys.argv[4]) if sys.argv[4] else None

embeddings = np.load(input_path)
emb_gpu = cp.asarray(embeddings, dtype=cp.float32)

# Embeddings arrive L2-normalized from _prepare_embeddings().
# Verify as a safety net — re-normalize if needed (prevents SIGFPE from NN-descent).
norms = cp.linalg.norm(emb_gpu, axis=1)
if cp.abs(norms.mean() - 1.0) > 0.01:
    emb_gpu = emb_gpu / cp.maximum(norms.reshape(-1, 1), 1e-10)

kw = dict(n_components=2, n_neighbors=n_neighbors)
if seed is not None:
    kw["random_state"] = seed
reducer = cuUMAP(**kw)
result = reducer.fit_transform(emb_gpu)
np.save(output_path, cp.asnumpy(result))
"""


def _run_cuml_umap_subprocess(embeddings: np.ndarray, seed: Optional[int]) -> np.ndarray:
    """Run cuML UMAP in an isolated subprocess to survive SIGFPE crashes.

    cuML UMAP's NN-descent can trigger a floating-point exception (SIGFPE) on
    certain data distributions, which kills the entire process.  By running in
    a child process, the parent (Streamlit) survives and can fall back to
    sklearn UMAP.
    """
    n_samples = embeddings.shape[0]
    n_neighbors = min(15, max(2, n_samples - 1))

    # Use /dev/shm for fast IPC when available, else /tmp
    shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    input_path = os.path.join(shm_dir, f"cuml_umap_in_{os.getpid()}.npy")
    output_path = os.path.join(shm_dir, f"cuml_umap_out_{os.getpid()}.npy")

    np.save(input_path, embeddings)
    seed_arg = str(seed) if seed is not None else ""

    try:
        logger.info(f"Running cuML UMAP in subprocess ({n_samples} samples, "
                    f"n_neighbors={n_neighbors})")
        result = subprocess.run(
            [sys.executable, "-c", _CUML_UMAP_SCRIPT,
             input_path, output_path, str(n_neighbors), seed_arg],
            capture_output=True, text=True, timeout=300,
        )

        if result.returncode == 0 and os.path.exists(output_path):
            reduced = np.load(output_path)
            logger.info("cuML UMAP subprocess completed successfully")
            return reduced

        stderr = result.stderr.strip()
        raise RuntimeError(
            f"cuML UMAP subprocess failed (rc={result.returncode}): "
            f"{stderr[-500:] if stderr else 'no stderr'}"
        )
    finally:
        for path in (input_path, output_path):
            try:
                os.unlink(path)
            except OSError:
                pass

def run_kmeans(embeddings: np.ndarray, n_clusters: int, seed: Optional[int] = None, n_workers: int = 1, backend: str = "auto"):
    """
    Perform KMeans clustering on the given embeddings.

    Args:
        embeddings (np.ndarray): The input feature embeddings of shape (n_samples, n_features).
        n_clusters (int): The number of clusters to form.
        seed (int, optional): Random seed for reproducibility. Defaults to None (random).
        n_workers (int, optional): Number of parallel workers (used by FAISS and cuML if available).
        backend (str, optional): Clustering backend - "auto", "sklearn", "faiss", or "cuml". Defaults to "auto".

    Returns:
        kmeans (KMeans or custom object): The fitted clustering object.
        labels (np.ndarray): Cluster labels for each sample.
    """
    n_samples = embeddings.shape[0]
    logger.info(f"KMeans clustering: n_clusters={n_clusters}, samples={n_samples}, backend={backend}")

    # Validate, cast, and L2-normalize
    embeddings = _prepare_embeddings(embeddings, "kmeans")

    start_time = time.time()

    # Determine which backend to use
    if backend == "cuml" and HAS_CUML and HAS_CUDA:
        logger.info("Using cuML backend for KMeans")
        result = _run_kmeans_cuml(embeddings, n_clusters, seed, n_workers)
    elif backend == "faiss" and HAS_FAISS:
        logger.info("Using FAISS backend for KMeans")
        result = _run_kmeans_faiss(embeddings, n_clusters, seed, n_workers)
    elif backend == "auto":
        # Auto selection priority: cuML > FAISS > sklearn
        if HAS_CUML and HAS_CUDA and n_samples > 500:
            logger.info("Auto-selected cuML backend for KMeans (GPU available, large dataset)")
            result = _run_kmeans_cuml(embeddings, n_clusters, seed, n_workers)
        elif HAS_FAISS and n_samples > 500:
            logger.info("Auto-selected FAISS backend for KMeans (large dataset)")
            result = _run_kmeans_faiss(embeddings, n_clusters, seed, n_workers)
        else:
            logger.info("Using sklearn backend for KMeans")
            result = _run_kmeans_sklearn(embeddings, n_clusters, seed)
    else:
        logger.info("Using sklearn backend for KMeans")
        result = _run_kmeans_sklearn(embeddings, n_clusters, seed)

    elapsed = time.time() - start_time
    logger.info(f"KMeans clustering completed in {elapsed:.2f}s")
    return result


def _run_kmeans_cuml(embeddings: np.ndarray, n_clusters: int, seed: Optional[int] = None, n_workers: int = 1):
    """KMeans using cuML GPU backend."""
    try:
        # Convert to cupy array for GPU processing
        embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)
        
        # Create cuML KMeans object
        if seed is not None:
            kmeans = cuKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                max_iter=300,
                init='k-means++',
                tol=1e-4
            )
        else:
            kmeans = cuKMeans(
                n_clusters=n_clusters,
                max_iter=300,
                init='k-means++',
                tol=1e-4
            )
        
        # Fit and predict on GPU
        labels_gpu = kmeans.fit_predict(embeddings_gpu)
        
        # Convert results back to numpy
        labels = cp.asnumpy(labels_gpu)
        centroids = cp.asnumpy(kmeans.cluster_centers_)
        
        # Create a simple object to mimic sklearn KMeans interface
        class cuMLKMeans:
            def __init__(self, centroids, labels):
                self.cluster_centers_ = centroids
                self.labels_ = labels
                self.n_clusters = len(centroids)
        
        return cuMLKMeans(centroids, labels), labels
        
    except Exception as e:
        logger.warning(f"cuML clustering failed ({e}), falling back to sklearn")
        return _run_kmeans_sklearn(embeddings, n_clusters, seed)


def _run_kmeans_sklearn(embeddings: np.ndarray, n_clusters: int, seed: Optional[int] = None):
    """KMeans using scikit-learn backend."""
    if seed is not None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    else:
        kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels


def _run_kmeans_faiss(embeddings: np.ndarray, n_clusters: int, seed: Optional[int] = None, n_workers: int = 1):
    """KMeans using FAISS backend for faster clustering."""
    try:
        import faiss
        
        # Ensure embeddings are float32 and C-contiguous (FAISS requirement)
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        n_samples, d = embeddings.shape
        
        # Set number of threads for FAISS
        if n_workers > 1:
            faiss.omp_set_num_threads(n_workers)
        
        # Create FAISS KMeans object
        kmeans = faiss.Clustering(d, n_clusters)
        
        # Set clustering parameters
        kmeans.verbose = False
        kmeans.niter = 20  # Number of iterations
        kmeans.nredo = 1   # Number of redos
        if seed is not None:
            kmeans.seed = seed
        
        # Use L2 distance (equivalent to sklearn's default)
        index = faiss.IndexFlatL2(d)
        
        # Run clustering
        kmeans.train(embeddings, index)
        
        # Get centroids
        centroids = faiss.vector_to_array(kmeans.centroids).reshape(n_clusters, d)
        
        # Assign labels by finding nearest centroid for each point
        _, labels = index.search(embeddings, 1)
        labels = labels.flatten()
        
        # Create a simple object to mimic sklearn KMeans interface
        class FAISSKMeans:
            def __init__(self, centroids, labels):
                self.cluster_centers_ = centroids
                self.labels_ = labels
                self.n_clusters = len(centroids)
        
        return FAISSKMeans(centroids, labels), labels
        
    except Exception as e:
        # Fallback to sklearn if FAISS fails
        logger.warning(f"FAISS clustering failed ({e}), falling back to sklearn")
        return _run_kmeans_sklearn(embeddings, n_clusters, seed)


