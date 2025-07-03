import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

def reduce_dim(embeddings: np.ndarray, method: str = "PCA", seed: int = 614, n_workers: int = 1):
    """
    Reduce the dimensionality of embeddings to 2D using PCA, t-SNE, or UMAP.

    Args:
        embeddings (np.ndarray): The input feature embeddings of shape (n_samples, n_features).
        method (str, optional): The dimensionality reduction method, "PCA", "TSNE", or "UMAP". Defaults to "PCA".
        seed (int, optional): Random seed for reproducibility. Defaults to 614 :)
        n_workers (int, optional): Number of parallel workers for t-SNE/UMAP. Defaults to 1.

    Returns:
        np.ndarray: The 2D reduced embeddings of shape (n_samples, 2).

    Raises:
        ValueError: If an unsupported method is provided.
    """
    if method.upper() == "PCA":
        reducer = PCA(n_components=2)
    elif method.upper() == "TSNE":
        reducer = TSNE(n_components=2, perplexity=30, random_state=seed, n_jobs=n_workers)
    elif method.upper() == "UMAP":
        import umap
        #reducer = umap.UMAP(n_components=2, random_state=seed, n_jobs=n_workers)
        reducer = umap.UMAP(n_components=2, n_jobs=n_workers)
    else:
        raise ValueError("Unsupported method. Choose 'PCA', 'TSNE', or 'UMAP'.")
    return reducer.fit_transform(embeddings)

def run_kmeans(embeddings: np.ndarray, n_clusters: int, seed: int = 614):
    """
    Perform KMeans clustering on the given embeddings.

    Args:
        embeddings (np.ndarray): The input feature embeddings of shape (n_samples, n_features).
        n_clusters (int): The number of clusters to form.
        seed (int, optional): Random seed for reproducibility. Defaults to 614 :)

    Returns:
        kmeans (KMeans): The fitted KMeans object.
        labels (np.ndarray): Cluster labels for each sample.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    labels = kmeans.fit_predict(embeddings)
    return kmeans, labels


