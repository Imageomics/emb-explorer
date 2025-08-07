from typing import Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

# Optional FAISS support for faster clustering
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# Optional cuML support for GPU acceleration
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    from cuml.manifold import TSNE as cuTSNE
    from cuml.manifold import UMAP as cuUMAP
    import cupy as cp
    HAS_CUML = True
except ImportError:
    HAS_CUML = False

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
    # Determine which backend to use
    use_cuml = False
    if backend == "cuml" and HAS_CUML and HAS_CUDA:
        use_cuml = True
    elif backend == "auto" and HAS_CUML and HAS_CUDA and embeddings.shape[0] > 5000:
        # Use cuML automatically for large datasets on GPU
        use_cuml = True
    
    if use_cuml:
        return _reduce_dim_cuml(embeddings, method, seed, n_workers)
    else:
        return _reduce_dim_sklearn(embeddings, method, seed, n_workers)


def _reduce_dim_sklearn(embeddings: np.ndarray, method: str, seed: Optional[int], n_workers: int):
    """Dimensionality reduction using sklearn/umap backends."""
    if method.upper() == "PCA":
        reducer = PCA(n_components=2)
    elif method.upper() == "TSNE":
        # Adjust perplexity to be valid for the sample size
        n_samples = embeddings.shape[0]
        perplexity = min(30, max(5, n_samples // 3))  # Ensure perplexity is reasonable
        
        if seed is not None:
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=seed, n_jobs=n_workers)
        else:
            reducer = TSNE(n_components=2, perplexity=perplexity, n_jobs=n_workers)
    elif method.upper() == "UMAP":
        if seed is not None:
            reducer = UMAP(n_components=2, random_state=seed, n_jobs=n_workers)
        else:
            reducer = UMAP(n_components=2, n_jobs=n_workers)
    else:
        raise ValueError("Unsupported method. Choose 'PCA', 'TSNE', or 'UMAP'.")
    return reducer.fit_transform(embeddings)


def _reduce_dim_cuml(embeddings: np.ndarray, method: str, seed: Optional[int], n_workers: int):
    """Dimensionality reduction using cuML GPU backends."""
    try:
        # Convert to cupy array for GPU processing
        embeddings_gpu = cp.asarray(embeddings, dtype=cp.float32)
        
        if method.upper() == "PCA":
            reducer = cuPCA(n_components=2)
        elif method.upper() == "TSNE":
            # Adjust perplexity to be valid for the sample size
            n_samples = embeddings.shape[0]
            perplexity = min(30, max(5, n_samples // 3))  # Ensure perplexity is reasonable
            
            if seed is not None:
                reducer = cuTSNE(n_components=2, perplexity=perplexity, random_state=seed)
            else:
                reducer = cuTSNE(n_components=2, perplexity=perplexity)
        elif method.upper() == "UMAP":
            if seed is not None:
                reducer = cuUMAP(n_components=2, random_state=seed)
            else:
                reducer = cuUMAP(n_components=2)
        else:
            raise ValueError("Unsupported method. Choose 'PCA', 'TSNE', or 'UMAP'.")
        
        # Fit and transform on GPU
        result_gpu = reducer.fit_transform(embeddings_gpu)
        
        # Convert back to numpy array
        return cp.asnumpy(result_gpu)
        
    except Exception as e:
        print(f"cuML reduction failed ({e}), falling back to sklearn")
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
    # Determine which backend to use
    if backend == "cuml" and HAS_CUML and HAS_CUDA:
        return _run_kmeans_cuml(embeddings, n_clusters, seed, n_workers)
    elif backend == "faiss" and HAS_FAISS:
        return _run_kmeans_faiss(embeddings, n_clusters, seed, n_workers)
    elif backend == "auto":
        # Auto selection priority: cuML > FAISS > sklearn
        if HAS_CUML and HAS_CUDA and embeddings.shape[0] > 500:
            return _run_kmeans_cuml(embeddings, n_clusters, seed, n_workers)
        elif HAS_FAISS and embeddings.shape[0] > 500:
            return _run_kmeans_faiss(embeddings, n_clusters, seed, n_workers)
        else:
            return _run_kmeans_sklearn(embeddings, n_clusters, seed)
    else:
        return _run_kmeans_sklearn(embeddings, n_clusters, seed)


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
        print(f"cuML clustering failed ({e}), falling back to sklearn")
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
        print(f"FAISS clustering failed ({e}), falling back to sklearn")
        return _run_kmeans_sklearn(embeddings, n_clusters, seed)


