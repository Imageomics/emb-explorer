from typing import Optional, Tuple
import time
import numpy as np

from shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Lazy-loaded module references (None until first use)
_faiss = None
_cuml_modules = None
_sklearn_modules = None
_umap_module = None

# Availability flags (None = not checked yet)
_HAS_FAISS: Optional[bool] = None
_HAS_CUML: Optional[bool] = None
_HAS_CUDA: Optional[bool] = None


def _get_sklearn_modules():
    """Lazy load sklearn modules."""
    global _sklearn_modules
    if _sklearn_modules is None:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        _sklearn_modules = {
            'KMeans': KMeans,
            'PCA': PCA,
            'TSNE': TSNE,
        }
        logger.debug("sklearn modules loaded")
    return _sklearn_modules


def _get_umap_module():
    """Lazy load UMAP."""
    global _umap_module
    if _umap_module is None:
        from umap import UMAP
        _umap_module = UMAP
        logger.debug("UMAP module loaded")
    return _umap_module


def _check_faiss_available() -> bool:
    """Check if FAISS is available (lazy check)."""
    global _HAS_FAISS, _faiss
    if _HAS_FAISS is None:
        try:
            import faiss
            _faiss = faiss
            _HAS_FAISS = True
            logger.info("FAISS loaded and available")
        except ImportError:
            _HAS_FAISS = False
            logger.debug("FAISS not available")
    return _HAS_FAISS


def _check_cuml_available() -> bool:
    """Check if cuML is available (lazy check)."""
    global _HAS_CUML, _cuml_modules
    if _HAS_CUML is None:
        try:
            import cuml
            from cuml.cluster import KMeans as cuKMeans
            from cuml.decomposition import PCA as cuPCA
            from cuml.manifold import TSNE as cuTSNE
            from cuml.manifold import UMAP as cuUMAP
            import cupy as cp
            _cuml_modules = {
                'cuml': cuml,
                'KMeans': cuKMeans,
                'PCA': cuPCA,
                'TSNE': cuTSNE,
                'UMAP': cuUMAP,
                'cp': cp,
            }
            _HAS_CUML = True
            logger.info("cuML loaded and available")
        except ImportError:
            _HAS_CUML = False
            logger.debug("cuML not available")
    return _HAS_CUML


def _check_cuda_available() -> bool:
    """Check if CUDA is available (lazy check)."""
    global _HAS_CUDA
    if _HAS_CUDA is None:
        try:
            import torch
            _HAS_CUDA = torch.cuda.is_available()
        except ImportError:
            try:
                import cupy as cp
                _HAS_CUDA = cp.cuda.is_available()
            except ImportError:
                _HAS_CUDA = False
        logger.debug(f"CUDA available: {_HAS_CUDA}")
    return _HAS_CUDA




class VRAMExceededError(Exception):
    """Raised when GPU VRAM is exceeded during computation."""
    pass


class GPUArchitectureError(Exception):
    """Raised when GPU architecture is not supported."""
    pass


def is_cuda_oom_error(error: Exception) -> bool:
    """Check if an exception is a CUDA out-of-memory error."""
    error_msg = str(error).lower()
    oom_indicators = [
        "out of memory",
        "cuda error: out of memory",
        "cudaerroroutofmemory",
        "oom",
        "memory allocation failed",
        "cudamalloc failed",
        "failed to allocate",
    ]
    return any(indicator in error_msg for indicator in oom_indicators)


def is_cuda_arch_error(error: Exception) -> bool:
    """Check if an exception is a CUDA architecture incompatibility error."""
    error_msg = str(error).lower()
    arch_indicators = [
        "no kernel image",
        "cudaerrornokernel",
        "unsupported gpu",
        "compute capability",
    ]
    return any(indicator in error_msg for indicator in arch_indicators)


def get_gpu_memory_info() -> Optional[Tuple[int, int]]:
    """
    Get GPU memory info (used, total) in MB.

    Returns:
        Tuple of (used_mb, total_mb) or None if unavailable.
    """
    try:
        if _check_cuml_available() and _check_cuda_available():
            cp = _cuml_modules['cp']
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

    # Determine which backend to use
    use_cuml = False
    if backend == "cuml" and _check_cuml_available() and _check_cuda_available():
        use_cuml = True
    elif backend == "auto" and _check_cuml_available() and _check_cuda_available() and n_samples > 5000:
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
    sklearn = _get_sklearn_modules()

    # Use -1 (all available cores) instead of specific values > 1 to avoid
    # thread count restrictions on HPC clusters (OMP_NUM_THREADS, SLURM cgroups)
    effective_workers = -1 if n_workers > 1 else n_workers

    if method.upper() == "PCA":
        reducer = sklearn['PCA'](n_components=2)
    elif method.upper() == "TSNE":
        # Adjust perplexity to be valid for the sample size
        n_samples = embeddings.shape[0]
        perplexity = min(30, max(5, n_samples // 3))  # Ensure perplexity is reasonable

        if seed is not None:
            reducer = sklearn['TSNE'](n_components=2, perplexity=perplexity, random_state=seed, n_jobs=effective_workers)
        else:
            reducer = sklearn['TSNE'](n_components=2, perplexity=perplexity, n_jobs=effective_workers)
    elif method.upper() == "UMAP":
        UMAP = _get_umap_module()
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
    """Dimensionality reduction using cuML GPU backends."""
    cuml = _cuml_modules  # Already loaded by caller check
    cp = cuml['cp']

    try:
        # Convert to cupy array for GPU processing
        embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)

        if method.upper() == "PCA":
            reducer = cuml['PCA'](n_components=2)
        elif method.upper() == "TSNE":
            # Adjust perplexity to be valid for the sample size
            n_samples = embeddings.shape[0]
            perplexity = min(30, max(5, n_samples // 3))  # Ensure perplexity is reasonable

            if seed is not None:
                reducer = cuml['TSNE'](n_components=2, perplexity=perplexity, random_state=seed)
            else:
                reducer = cuml['TSNE'](n_components=2, perplexity=perplexity)
        elif method.upper() == "UMAP":
            # Adjust n_neighbors to be valid for the sample size
            n_samples = embeddings.shape[0]
            n_neighbors = min(15, max(2, n_samples - 1))

            if seed is not None:
                reducer = cuml['UMAP'](n_components=2, n_neighbors=n_neighbors, random_state=seed)
            else:
                reducer = cuml['UMAP'](n_components=2, n_neighbors=n_neighbors)
        else:
            raise ValueError("Unsupported method. Choose 'PCA', 'TSNE', or 'UMAP'.")

        # Fit and transform on GPU
        result_gpu = reducer.fit_transform(embeddings_gpu)

        # Convert back to numpy array
        return cp.asnumpy(result_gpu)

    except RuntimeError as e:
        # Handle CUDA architecture mismatch (e.g., V100 not supported by pip wheels)
        error_msg = str(e).lower()
        if "no kernel image" in error_msg or "cudaerrornokernel" in error_msg:
            logger.warning(f"cuML {method} not supported on this GPU architecture, falling back to sklearn")
        else:
            logger.warning(f"cuML reduction failed ({e}), falling back to sklearn")
        return _reduce_dim_sklearn(embeddings, method, seed, n_workers)
    except Exception as e:
        logger.warning(f"cuML reduction failed ({e}), falling back to sklearn")
        return _reduce_dim_sklearn(embeddings, method, seed, n_workers)

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

    start_time = time.time()

    # Determine which backend to use
    if backend == "cuml" and _check_cuml_available() and _check_cuda_available():
        logger.info("Using cuML backend for KMeans")
        result = _run_kmeans_cuml(embeddings, n_clusters, seed, n_workers)
    elif backend == "faiss" and _check_faiss_available():
        logger.info("Using FAISS backend for KMeans")
        result = _run_kmeans_faiss(embeddings, n_clusters, seed, n_workers)
    elif backend == "auto":
        # Auto selection priority: cuML > FAISS > sklearn
        if _check_cuml_available() and _check_cuda_available() and n_samples > 500:
            logger.info("Auto-selected cuML backend for KMeans (GPU available, large dataset)")
            result = _run_kmeans_cuml(embeddings, n_clusters, seed, n_workers)
        elif _check_faiss_available() and n_samples > 500:
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
    cuml = _cuml_modules  # Already loaded by caller check
    cp = cuml['cp']
    cuKMeans = cuml['KMeans']

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
    sklearn = _get_sklearn_modules()
    KMeans = sklearn['KMeans']

    if seed is not None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    else:
        kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels


def _run_kmeans_faiss(embeddings: np.ndarray, n_clusters: int, seed: Optional[int] = None, n_workers: int = 1):
    """KMeans using FAISS backend for faster clustering."""
    faiss = _faiss  # Already loaded by caller check

    try:
        
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


