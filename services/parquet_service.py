"""
Service for handling parquet file operations with embeddings and metadata.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Optional DuckDB support for lazy loading
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class ParquetService:
    """Service for handling parquet file operations with embeddings and metadata"""
    
    # Define the expected taxonomic columns based on your schema
    TAXONOMIC_COLUMNS = [
        'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'
    ]
    
    METADATA_COLUMNS = [
        'source_dataset', 'scientific_name', 'common_name',
        'publisher', 'basisOfRecord', 'img_type'
    ] + TAXONOMIC_COLUMNS
    
    @staticmethod
    @st.cache_data
    def load_parquet_file(file_path: str) -> pd.DataFrame:
        """
        Load a parquet file and return as DataFrame.
        
        Args:
            file_path: Path to the parquet file
            
        Returns:
            DataFrame with the parquet data
        """
        try:
            df = pd.read_parquet(file_path)
            return df
        except Exception as e:
            raise ValueError(f"Error loading parquet file: {e}")
    
    @staticmethod
    def validate_parquet_structure(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that the parquet file has the expected structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for required columns
        if 'uuid' not in df.columns:
            issues.append("Missing required 'uuid' column")
        if 'emb' not in df.columns:
            issues.append("Missing required 'emb' column")
            
        # Check for null values in critical columns
        if 'uuid' in df.columns and df['uuid'].isnull().any():
            issues.append("Found null values in 'uuid' column")
        if 'emb' in df.columns and df['emb'].isnull().any():
            issues.append("Found null values in 'emb' column")
            
        # Check embedding format
        if 'emb' in df.columns:
            try:
                # Try to convert first embedding to check format
                first_emb = df['emb'].iloc[0]
                if not isinstance(first_emb, (list, np.ndarray)):
                    issues.append("Embedding column 'emb' does not contain arrays")
                elif len(first_emb) == 0:
                    issues.append("Empty embeddings found")
            except Exception as e:
                issues.append(f"Error parsing embeddings: {e}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def extract_embeddings(df: pd.DataFrame) -> np.ndarray:
        """
        Extract embeddings from the DataFrame.
        
        Args:
            df: DataFrame containing 'emb' column
            
        Returns:
            numpy array of embeddings with shape (n_samples, embedding_dim)
        """
        if 'emb' not in df.columns:
            raise ValueError("DataFrame does not contain 'emb' column")
            
        # Convert embeddings to numpy array
        embeddings = np.array(df['emb'].tolist())
        
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings should be 2D, got shape {embeddings.shape}")
            
        return embeddings
    
    @staticmethod
    def get_column_info(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Get information about each column for filtering purposes.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary mapping column names to their info (type, unique_values, etc.)
        """
        column_info = {}
        
        for col in df.columns:
            if col in ['uuid', 'emb']:  # Skip technical columns
                continue
                
            col_data = df[col].dropna()  # Remove nulls for analysis
            
            if len(col_data) == 0:
                col_type = 'empty'
                unique_values = []
                value_counts = {}
            elif pd.api.types.is_numeric_dtype(col_data):
                col_type = 'numeric'
                unique_values = None
                value_counts = None
            elif len(col_data.unique()) <= 50:  # Categorical if <= 50 unique values
                col_type = 'categorical'
                unique_values = sorted(col_data.unique().tolist())
                value_counts = col_data.value_counts().to_dict()
            else:
                col_type = 'text'
                unique_values = None
                value_counts = None
            
            column_info[col] = {
                'type': col_type,
                'unique_values': unique_values,
                'value_counts': value_counts,
                'null_count': df[col].isnull().sum(),
                'total_count': len(df),
                'null_percentage': (df[col].isnull().sum() / len(df)) * 100
            }
        
        return column_info
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to the DataFrame.
        
        Args:
            df: DataFrame to filter
            filters: Dictionary of column_name -> filter_value pairs
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        for col, filter_value in filters.items():
            if col not in df.columns or filter_value is None:
                continue
                
            col_data = df[col]
            
            if isinstance(filter_value, dict):
                # Numeric range filter
                if 'min' in filter_value and filter_value['min'] is not None:
                    filtered_df = filtered_df[col_data >= filter_value['min']]
                if 'max' in filter_value and filter_value['max'] is not None:
                    filtered_df = filtered_df[col_data <= filter_value['max']]
            elif isinstance(filter_value, list):
                # Categorical filter (multiple values)
                if len(filter_value) > 0:
                    filtered_df = filtered_df[col_data.isin(filter_value)]
            elif isinstance(filter_value, str):
                # Text filter (contains)
                if filter_value.strip():
                    filtered_df = filtered_df[
                        col_data.str.contains(filter_value, case=False, na=False)
                    ]
        
        return filtered_df
    
    @staticmethod
    def create_cluster_dataframe(
        df: pd.DataFrame,
        embeddings_2d: np.ndarray,
        labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Create a dataframe for clustering visualization.
        
        Args:
            df: Original dataframe with metadata
            embeddings_2d: 2D reduced embeddings
            labels: Cluster labels
            
        Returns:
            DataFrame suitable for plotting
        """
        df_plot = pd.DataFrame({
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "cluster": labels.astype(str),
            "uuid": df['uuid'].values,
            "idx": range(len(df))
        })
        
        # Add key metadata columns for tooltips
        metadata_cols = ['scientific_name', 'common_name', 'family', 'genus', 'species']
        for col in metadata_cols:
            if col in df.columns:
                df_plot[col] = df[col].values
        
        return df_plot
