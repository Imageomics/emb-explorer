"""
Sidebar components for the precalculated embeddings page.
"""

import streamlit as st
import os
from typing import Dict, Any, Optional, Tuple

from services.parquet_service import ParquetService
from services.clustering_service import ClusteringService
from components.shared.clustering_controls import render_clustering_backend_controls, render_basic_clustering_controls


def render_file_section() -> Tuple[bool, Optional[str]]:
    """
    Render the file loading section.
    
    Returns:
        Tuple of (file_loaded, file_path)
    """
    with st.expander("📁 Load Parquet File", expanded=True):
        file_path = st.text_input(
            "Parquet file path",
            help="Path to your parquet file containing embeddings and metadata. Large files are loaded efficiently."
        )
        
        # Option for lazy loading with DuckDB (if available)
        try:
            import duckdb
            use_lazy_loading = st.checkbox(
                "Use lazy loading (DuckDB)", 
                value=False,
                help="Recommended for very large parquet files (>1GB). Requires DuckDB."
            )
        except ImportError:
            use_lazy_loading = False
            st.info("💡 Install DuckDB for lazy loading of very large files: `pip install duckdb`")
        
        load_button = st.button("Load File")
        
        if load_button and file_path and os.path.exists(file_path):
            try:
                with st.spinner("Loading parquet file..."):
                    df = ParquetService.load_parquet_file(file_path)
                    
                # Validate structure
                is_valid, issues = ParquetService.validate_parquet_structure(df)
                
                if not is_valid:
                    st.error("File validation failed:")
                    for issue in issues:
                        st.error(f"• {issue}")
                    return False, file_path
                
                # Store in session state
                st.session_state.parquet_df = df
                st.session_state.parquet_file_path = file_path
                st.session_state.column_info = ParquetService.get_column_info(df)
                
                # Reset downstream state
                st.session_state.filtered_df = None
                st.session_state.embeddings = None
                st.session_state.data = None
                st.session_state.labels = None
                st.session_state.selected_image_idx = None
                
                st.success(f"✅ Loaded {len(df):,} records from parquet file")
                st.info(f"Embedding dimension: {len(df['emb'].iloc[0])}")
                
                return True, file_path
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
                return False, file_path
                
        elif load_button and file_path:
            st.error(f"File not found: {file_path}")
            return False, file_path
        elif load_button:
            st.error("Please provide a file path")
            return False, None
    
    return False, file_path


def render_filter_section() -> Dict[str, Any]:
    """
    Render the metadata filtering section.
    
    Returns:
        Dictionary of applied filters
    """
    with st.expander("🔍 Filter Data", expanded=True):
        df = st.session_state.get("parquet_df", None)
        column_info = st.session_state.get("column_info", {})
        
        if df is None:
            st.info("Load a parquet file first to enable filtering.")
            return {}
        
        st.markdown(f"**Total records:** {len(df):,}")
        
        filters = {}
        
        # Create filter widgets based on column types
        for col, info in column_info.items():
            # Skip the removed fields
            if col in ['source_id', 'identifier', 'resolution_status']:
                continue
                
            if info['type'] == 'empty':
                continue
                
            with st.container():
                st.markdown(f"**{col}** ({info['type']})")
                
                if info['null_count'] > 0:
                    st.caption(f"⚠️ {info['null_count']:,} null values ({info['null_percentage']:.1f}%)")
                
                if info['type'] == 'categorical':
                    # Multi-select for categorical
                    selected_values = st.multiselect(
                        f"Select {col} values",
                        options=info['unique_values'],
                        key=f"filter_{col}",
                        help=f"{len(info['unique_values'])} unique values"
                    )
                    if selected_values:
                        filters[col] = selected_values
                        
                elif info['type'] == 'numeric':
                    # Range slider for numeric
                    col_data = df[col].dropna()
                    if len(col_data) > 0:
                        min_val, max_val = float(col_data.min()), float(col_data.max())
                        if min_val != max_val:
                            range_values = st.slider(
                                f"{col} range",
                                min_value=min_val,
                                max_value=max_val,
                                value=(min_val, max_val),
                                key=f"filter_{col}"
                            )
                            if range_values != (min_val, max_val):
                                filters[col] = {'min': range_values[0], 'max': range_values[1]}
                
                elif info['type'] == 'text':
                    # Text input for text search
                    search_text = st.text_input(
                        f"Search in {col}",
                        key=f"filter_{col}",
                        help="Case-insensitive substring search"
                    )
                    if search_text.strip():
                        filters[col] = search_text.strip()
        
        # Apply filters button and results
        if st.button("Apply Filters", type="primary"):
            if filters:
                with st.spinner("Applying filters..."):
                    filtered_df = ParquetService.apply_filters(df, filters)
                    st.session_state.filtered_df = filtered_df
                    st.session_state.current_filters = filters
                    
                    # Reset downstream state
                    st.session_state.embeddings = None
                    st.session_state.data = None
                    st.session_state.labels = None
                    st.session_state.selected_image_idx = None
                    
                    st.success(f"✅ Filtered to {len(filtered_df):,} records")
            else:
                # No filters applied, use full dataset
                st.session_state.filtered_df = df
                st.session_state.current_filters = {}
                st.info("No filters applied, using full dataset")
        
        # Show current filter summary
        current_filters = st.session_state.get("current_filters", {})
        if current_filters:
            st.markdown("**Active filters:**")
            for col, filter_val in current_filters.items():
                if isinstance(filter_val, list):
                    st.caption(f"• {col}: {len(filter_val)} values selected")
                elif isinstance(filter_val, dict):
                    st.caption(f"• {col}: {filter_val['min']} - {filter_val['max']}")
                else:
                    st.caption(f"• {col}: contains '{filter_val}'")
        
        return filters


def render_clustering_section() -> Tuple[bool, int, str, str, str, int, Optional[int]]:
    """
    Render the clustering section.
    
    Returns:
        Tuple of (cluster_button_clicked, n_clusters, reduction_method, dim_reduction_backend, clustering_backend, n_workers, seed)
    """
    with st.expander("🎯 Cluster Embeddings", expanded=False):
        filtered_df = st.session_state.get("filtered_df", None)
        
        if filtered_df is None or len(filtered_df) == 0:
            st.info("Apply filters first to enable clustering.")
            return False, 5, "TSNE", "auto", "auto", 8, None
        
        st.markdown(f"**Ready to cluster:** {len(filtered_df):,} records")
        
        n_clusters = st.slider("Number of clusters", 2, min(100, len(filtered_df)//2), 5)
        reduction_method = st.selectbox("Dimensionality Reduction", ["TSNE", "PCA", "UMAP"])
        
        # Backend and advanced controls
        dim_reduction_backend, clustering_backend, n_workers, seed = render_clustering_backend_controls()
        
        cluster_button = st.button("Run Clustering", type="primary")
        
        if cluster_button:
            try:
                with st.spinner("Extracting embeddings..."):
                    embeddings = ParquetService.extract_embeddings(filtered_df)
                    st.session_state.embeddings = embeddings
                
                with st.spinner("Running clustering..."):
                    df_plot, labels = ClusteringService.run_clustering(
                        embeddings, 
                        filtered_df['uuid'].tolist(),  # Use UUIDs as "paths"
                        n_clusters, 
                        reduction_method,
                        n_workers,  # Pass the workers parameter
                        dim_reduction_backend,  # Explicit dimensionality reduction backend
                        clustering_backend,  # Explicit clustering backend
                        seed  # Random seed
                    )
                    
                    # Create enhanced plot dataframe with metadata
                    df_plot = ParquetService.create_cluster_dataframe(
                        filtered_df.reset_index(drop=True), 
                        df_plot[['x', 'y']].values, 
                        labels
                    )
                
                # Store results
                st.session_state.data = df_plot
                st.session_state.labels = labels
                st.session_state.selected_image_idx = 0
                st.session_state.filtered_df_for_clustering = filtered_df.reset_index(drop=True)
                
                st.success(f"✅ Clustering complete! Found {n_clusters} clusters.")
                
            except Exception as e:
                st.error(f"Error during clustering: {e}")
        
        return cluster_button, n_clusters, reduction_method, dim_reduction_backend, clustering_backend, n_workers, seed


def render_precalculated_sidebar():
    """Render the complete precalculated embeddings sidebar."""
    tab_load, tab_cluster = st.tabs(["Load & Filter", "Cluster"])
    
    with tab_load:
        file_loaded, file_path = render_file_section()
        filters = render_filter_section()
    
    with tab_cluster:
        cluster_button, n_clusters, reduction_method, dim_reduction_backend, clustering_backend, n_workers, seed = render_clustering_section()
    
    return {
        'file_loaded': file_loaded,
        'file_path': file_path,
        'filters': filters,
        'cluster_button': cluster_button,
        'n_clusters': n_clusters,
        'reduction_method': reduction_method,
        'dim_reduction_backend': dim_reduction_backend,
        'clustering_backend': clustering_backend,
        'n_workers': n_workers,
        'seed': seed,
    }
