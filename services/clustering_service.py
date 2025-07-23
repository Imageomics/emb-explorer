"""
Clustering service.
"""

import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, List, Any

from utils.clustering import run_kmeans, reduce_dim


class ClusteringService:
    """Service for handling clustering workflows"""
    
    @staticmethod
    def run_clustering(
        embeddings: np.ndarray,
        valid_paths: List[str],
        n_clusters: int,
        reduction_method: str,
        n_workers: int = 1
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Run clustering on embeddings.
        
        Args:
            embeddings: Input embeddings
            valid_paths: List of image paths
            n_clusters: Number of clusters
            reduction_method: Dimensionality reduction method
            n_workers: Number of workers for reduction
            
        Returns:
            Tuple of (cluster dataframe, cluster labels)
        """
        reduced = reduce_dim(embeddings, reduction_method, n_workers=n_workers)
        kmeans, labels = run_kmeans(reduced, int(n_clusters))
        
        df_plot = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "cluster": labels.astype(str),
            "image_path": valid_paths,
            "file_name": [os.path.basename(p) for p in valid_paths],
            "idx": range(len(valid_paths))
        })
        
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
        cluster_ids = np.unique(labels)
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
