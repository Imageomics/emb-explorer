"""
Service for handling parquet file operations with embeddings and metadata.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pandas as pd  # Keep for DataFrame output compatibility
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path


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
    def load_parquet_table(file_path: str) -> pa.Table:
        """
        Load a parquet file as PyArrow Table (zero-copy, memory efficient).
        
        Args:
            file_path: Path to the parquet file
            
        Returns:
            PyArrow Table with the parquet data
        """
        try:
            return pq.read_table(file_path)
        except Exception as e:
            raise ValueError(f"Error loading parquet file: {e}")
    
    @staticmethod
    def validate_parquet_structure(df: Union[pd.DataFrame, pa.Table]) -> Tuple[bool, List[str]]:
        """
        Validate that the parquet file has the expected structure.
        
        Args:
            df: DataFrame or PyArrow Table to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        if isinstance(df, pa.Table):
            # PyArrow Table validation
            column_names = df.column_names
            
            # Check for required columns
            if 'uuid' not in column_names:
                issues.append("Missing required 'uuid' column")
            if 'emb' not in column_names:
                issues.append("Missing required 'emb' column")
                
            # Check for null values in critical columns
            if 'uuid' in column_names:
                uuid_col = df.column('uuid')
                null_count = pc.sum(pc.is_null(uuid_col)).as_py()
                if null_count > 0:
                    issues.append("Found null values in 'uuid' column")
                    
            if 'emb' in column_names:
                emb_col = df.column('emb')
                null_count = pc.sum(pc.is_null(emb_col)).as_py()
                if null_count > 0:
                    issues.append("Found null values in 'emb' column")
                
                # Check embedding format
                try:
                    # Try to get first non-null embedding to check format
                    first_emb = None
                    for i in range(min(len(emb_col), 100)):  # Check first 100 rows
                        if emb_col[i].is_valid:
                            first_emb = emb_col[i].as_py()
                            break
                    
                    if first_emb is not None:
                        if not isinstance(first_emb, (list, tuple)):
                            issues.append("Embedding column 'emb' does not contain arrays")
                        elif len(first_emb) == 0:
                            issues.append("Empty embeddings found")
                    else:
                        issues.append("No valid embeddings found")
                except Exception as e:
                    issues.append(f"Error parsing embeddings: {e}")
        else:
            # pandas DataFrame validation (fallback for compatibility)
            df = df.to_pandas() if isinstance(df, pa.Table) else df
            
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
    def extract_embeddings(df: Union[pd.DataFrame, pa.Table]) -> np.ndarray:
        """
        Extract embeddings from the DataFrame or PyArrow Table.
        
        Args:
            df: DataFrame or PyArrow Table containing 'emb' column
            
        Returns:
            numpy array of embeddings with shape (n_samples, embedding_dim)
        """
        if isinstance(df, pa.Table):
            if 'emb' not in df.column_names:
                raise ValueError("Table does not contain 'emb' column")
            
            # Extract embeddings column as PyArrow array
            emb_column = df.column('emb')
            # Convert to numpy - PyArrow list arrays need special handling
            embeddings = emb_column.to_pylist()
            embeddings = np.array(embeddings)
        else:
            # pandas DataFrame fallback
            if 'emb' not in df.columns:
                raise ValueError("DataFrame does not contain 'emb' column")
            embeddings = np.array(df['emb'].tolist())
        
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings should be 2D, got shape {embeddings.shape}")
            
        return embeddings
    
    @staticmethod
    def get_column_info(df: Union[pd.DataFrame, pa.Table]) -> Dict[str, Dict[str, Any]]:
        """
        Get information about each column for filtering purposes.
        
        Args:
            df: DataFrame or PyArrow Table to analyze
            
        Returns:
            Dictionary mapping column names to their info (type, unique_values, etc.)
        """
        column_info = {}
        
        # Convert to PyArrow table if pandas DataFrame
        if isinstance(df, pd.DataFrame):
            df = pa.Table.from_pandas(df)
        
        # PyArrow Table processing
        for col_name in df.column_names:
            if col_name in ['uuid', 'emb']:  # Skip technical columns
                continue
            
            col_array = df.column(col_name)
            
            # Handle null values
            non_null_mask = pc.is_valid(col_array)
            non_null_count = pc.sum(non_null_mask).as_py()
            total_count = len(col_array)
            null_count = total_count - non_null_count
            
            if non_null_count == 0:
                col_type = 'empty'
                unique_values = []
                value_counts = {}
            else:
                # Check data type
                arrow_type = col_array.type
                
                if (pa.types.is_integer(arrow_type) or 
                    pa.types.is_floating(arrow_type) or 
                    pa.types.is_decimal(arrow_type)):
                    col_type = 'numeric'
                    unique_values = None
                    value_counts = None
                else:
                    # Get unique values for categorical determination
                    try:
                        unique_array = pc.unique(col_array)
                        unique_count = len(unique_array)
                        
                        if unique_count <= 50:  # Categorical if <= 50 unique values
                            col_type = 'categorical'
                            unique_values = sorted([v.as_py() for v in unique_array if v.is_valid])
                            
                            # Get value counts
                            value_counts_result = pc.value_counts(col_array)
                            value_counts = {}
                            for i in range(len(value_counts_result)):
                                struct = value_counts_result[i].as_py()
                                if struct['values'] is not None:
                                    value_counts[struct['values']] = struct['counts']
                        else:
                            col_type = 'text'
                            unique_values = None
                            value_counts = None
                    except:
                        col_type = 'text'
                        unique_values = None
                        value_counts = None
            
            column_info[col_name] = {
                'type': col_type,
                'unique_values': unique_values,
                'value_counts': value_counts,
                'null_count': null_count,
                'total_count': total_count,
                'null_percentage': (null_count / total_count) * 100 if total_count > 0 else 0
            }
        
        return column_info
    
    @staticmethod
    def apply_filters_arrow(table: pa.Table, filters: Dict[str, Any]) -> pa.Table:
        """
        Apply filters to PyArrow Table (more memory efficient).
        
        Args:
            table: PyArrow Table to filter
            filters: Dictionary of column_name -> filter_value pairs
            
        Returns:
            Filtered PyArrow Table
        """
        filter_expressions = []
        
        for col, filter_value in filters.items():
            if col not in table.column_names or filter_value is None:
                continue
            
            col_ref = pc.field(col)
            
            if isinstance(filter_value, dict):
                # Numeric range filter
                if 'min' in filter_value and filter_value['min'] is not None:
                    filter_expressions.append(pc.greater_equal(col_ref, filter_value['min']))
                if 'max' in filter_value and filter_value['max'] is not None:
                    filter_expressions.append(pc.less_equal(col_ref, filter_value['max']))
            elif isinstance(filter_value, list):
                # Categorical filter (multiple values) 
                if len(filter_value) > 0:
                    filter_expressions.append(pc.is_in(col_ref, pa.array(filter_value)))
            elif isinstance(filter_value, str):
                # Text filter (contains)
                if filter_value.strip():
                    # PyArrow string matching (case insensitive)
                    pattern = f"*{filter_value.lower()}*"
                    filter_expressions.append(
                        pc.match_substring_regex(
                            pc.utf8_lower(col_ref), 
                            pattern.replace("*", ".*")
                        )
                    )
        
        # Combine all filter expressions with AND
        if filter_expressions:
            if len(filter_expressions) == 1:
                combined_filter = filter_expressions[0]
            else:
                # Combine filters using reduce pattern
                from functools import reduce
                try:
                    # Try pc.and_kleene first (newer PyArrow versions)
                    combined_filter = reduce(lambda a, b: pc.and_kleene(a, b), filter_expressions)
                except AttributeError:
                    # Fallback for older PyArrow versions - apply filters sequentially
                    filtered_table = table
                    for expr in filter_expressions:
                        filtered_table = filtered_table.filter(expr)
                    return filtered_table
            
            return table.filter(combined_filter)
        
        return table
    
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

    @staticmethod
    def load_and_filter_efficient(
        file_path: str, 
        filters: Optional[Dict[str, Any]] = None,
        columns: Optional[List[str]] = None
    ) -> Tuple[pa.Table, pd.DataFrame]:
        """
        Load parquet file efficiently with PyArrow and apply filters.
        Returns both PyArrow table (for efficient operations) and pandas DataFrame (for compatibility).
        
        Args:
            file_path: Path to parquet file
            filters: Optional filters to apply
            columns: Optional list of columns to select
            
        Returns:
            Tuple of (PyArrow Table, pandas DataFrame)
        """
        # Load as PyArrow table
        table = ParquetService.load_parquet_table(file_path)
        
        # Apply column selection if specified
        if columns:
            # Ensure required columns are included
            required_cols = ['uuid', 'emb']
            all_columns = list(set(columns + required_cols))
            available_columns = [col for col in all_columns if col in table.column_names]
            table = table.select(available_columns)
        
        # Apply filters efficiently with PyArrow
        if filters:
            table = ParquetService.apply_filters_arrow(table, filters)
        
        # Convert to pandas for compatibility (only the filtered data)
        df = table.to_pandas()
        
        return table, df
