"""
Clustering service.
"""

import numpy as np
import pandas as pd
import os
import time
from typing import Tuple, Dict, List, Any

from shared.utils.clustering import run_kmeans, reduce_dim
from shared.utils.logging_config import get_logger

logger = get_logger(__name__)


class ClusteringService:
    """Service for handling clustering workflows"""

    @staticmethod
    def run_clustering(
        embeddings: np.ndarray,
        valid_paths: List[str],
        n_clusters: int,
        reduction_method: str,
        n_workers: int = 1,
        dim_reduction_backend: str = "auto",
        clustering_backend: str = "auto",
        seed: int = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Run clustering on embeddings.

        Args:
            embeddings: Input embeddings
            valid_paths: List of image paths
            n_clusters: Number of clusters
            reduction_method: Dimensionality reduction method
            n_workers: Number of workers for reduction
            dim_reduction_backend: Backend for dimensionality reduction ("auto", "sklearn", "faiss", "cuml")
            clustering_backend: Backend for clustering ("auto", "sklearn", "faiss", "cuml")
            seed: Random seed for reproducibility (None for random)

        Returns:
            Tuple of (cluster dataframe, cluster labels)
        """
        logger.info(f"Starting clustering workflow: n_samples={len(embeddings)}, n_clusters={n_clusters}, "
                    f"reduction={reduction_method}, dim_backend={dim_reduction_backend}, "
                    f"clustering_backend={clustering_backend}")

        total_start = time.time()

        # Step 1: Perform K-means clustering on full high-dimensional embeddings
        logger.info("Step 1/2: Running KMeans clustering on high-dimensional embeddings")
        kmeans, labels = run_kmeans(
            embeddings,  # Use original high-dimensional embeddings for clustering
            int(n_clusters),
            seed=seed,
            n_workers=n_workers,
            backend=clustering_backend
        )

        # Step 2: Reduce dimensionality to 2D for visualization only
        logger.info("Step 2/2: Reducing dimensionality to 2D for visualization")
        reduced = reduce_dim(
            embeddings,
            reduction_method,
            seed=seed,
            n_workers=n_workers,
            backend=dim_reduction_backend
        )

        df_plot = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "cluster": labels.astype(str),
            "image_path": valid_paths,
            "file_name": [os.path.basename(p) for p in valid_paths],
            "idx": range(len(valid_paths))
        })

        total_elapsed = time.time() - total_start
        logger.info(f"Clustering workflow completed in {total_elapsed:.2f}s")

        return df_plot, labels

    @staticmethod
    def generate_clustering_summary(
        embeddings: np.ndarray,
        labels: np.ndarray,
        df_plot: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
        """
        Generate clustering summary statistics and representative images.

        Args:
            embeddings: Original embeddings
            labels: Cluster labels
            df_plot: Clustering dataframe

        Returns:
            Tuple of (summary dataframe, representatives dict)
        """
        logger.info("Generating clustering summary statistics")
        cluster_ids = np.unique(labels)
        logger.debug(f"Found {len(cluster_ids)} unique clusters")
        summary_data = []
        representatives = {}

        for k in cluster_ids:
            idxs = np.where(labels == k)[0]
            cluster_embeds = embeddings[idxs]
            centroid = cluster_embeds.mean(axis=0)

            # Internal variance
            variance = np.mean(np.sum((cluster_embeds - centroid) ** 2, axis=1))

            # Find 3 closest images
            dists = np.sum((cluster_embeds - centroid) ** 2, axis=1)
            closest_indices = idxs[np.argsort(dists)[:3]]
            representatives[k] = closest_indices

            summary_data.append({
                "Cluster": int(k),
                "Count": len(idxs),
                "Variance": round(variance, 3),
            })

        summary_df = pd.DataFrame(summary_data)
        return summary_df, representatives
