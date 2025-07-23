"""
File operations service.
"""

import os
import pandas as pd
import concurrent.futures
from typing import List, Dict, Any, Optional, Callable, Tuple

from utils.io import copy_image


class FileService:
    """Service for handling file operations like saving and repartitioning"""
    
    @staticmethod
    def save_cluster_images(
        cluster_rows: pd.DataFrame,
        save_dir: str,
        max_workers: int,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Save images from selected clusters.
        
        Args:
            cluster_rows: DataFrame containing cluster data to save
            save_dir: Directory to save images
            max_workers: Number of worker threads
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (summary dataframe, csv path)
        """
        os.makedirs(save_dir, exist_ok=True)
        save_rows = []
        
        if progress_callback:
            progress_callback(0.0, "Copying images...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(copy_image, row, save_dir)
                for idx, row in cluster_rows.iterrows()
            ]
            total_files = len(futures)
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    save_rows.append(result)
                    
                # Progress callback with same logic as before
                if i % 50 == 0 or i == total_files:
                    if progress_callback:
                        progress = i / total_files
                        progress_callback(progress, f"Copied {i} / {total_files} images")
        
        save_summary_df = pd.DataFrame(save_rows)
        csv_path = os.path.join(save_dir, "saved_cluster_summary.csv")
        save_summary_df.to_csv(csv_path, index=False)
        
        return save_summary_df, csv_path
    
    @staticmethod
    def repartition_images_by_cluster(
        df_plot: pd.DataFrame,
        repartition_dir: str,
        max_workers: int,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[pd.DataFrame, str]:
        """
        Repartition all images by cluster.
        
        Args:
            df_plot: DataFrame containing all cluster data
            repartition_dir: Directory to repartition images
            max_workers: Number of worker threads
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (summary dataframe, csv path)
        """
        os.makedirs(repartition_dir, exist_ok=True)
        repartition_rows = []
        
        if progress_callback:
            progress_callback(0.0, "Starting repartitioning...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(copy_image, row, repartition_dir)
                for idx, row in df_plot.iterrows()
            ]
            total_files = len(futures)
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    repartition_rows.append(result)
                    
                if i % 100 == 0 or i == total_files:
                    if progress_callback:
                        progress = i / total_files
                        progress_callback(progress, f"Repartitioned {i} / {total_files} images")
        
        repartition_summary_df = pd.DataFrame(repartition_rows)
        csv_path = os.path.join(repartition_dir, "cluster_summary.csv")
        repartition_summary_df.to_csv(csv_path, index=False)
        
        return repartition_summary_df, csv_path
