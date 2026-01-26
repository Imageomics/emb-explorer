"""
Shared clustering controls component.
"""

import streamlit as st
from typing import Tuple, Optional


def render_clustering_backend_controls():
    """
    Render clustering backend selection controls.
    
    Returns:
        Tuple of (dim_reduction_backend, clustering_backend, n_workers, seed)
    """
    # Backend availability detection
    dim_reduction_options = ["auto", "sklearn"]
    clustering_options = ["auto", "sklearn"]
    
    has_faiss = False
    has_cuml = False
    has_cuda = False
    
    # Check for FAISS (clustering only)
    try:
        import faiss
        has_faiss = True
        clustering_options.append("faiss")
    except ImportError:
        pass
    
    # Check for cuML + CUDA (both dim reduction and clustering)
    try:
        import cuml
        import cupy as cp
        has_cuml = True
        if cp.cuda.is_available():
            has_cuda = True
            dim_reduction_options.append("cuml")
            clustering_options.append("cuml")
    except ImportError:
        pass
    
    # Show backend status
    use_seed = st.checkbox(
        "Use fixed seed",
        value=False,
        help="Enable for reproducible results"
    )
    
    if use_seed:
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=999999,
            value=614,
            step=1,
            help="Random seed for reproducible clustering results"
        )
    else:
        seed = None
        
    with st.expander("🔧 Available Backends:", expanded=False):
        
        # Explicit backend selection with two columns
        col1, col2 = st.columns(2)
        
        with col1:
            dim_reduction_backend = st.selectbox(
                "Dimensionality Reduction Backend",
                options=dim_reduction_options,
                index=0,
                help="Backend for PCA/t-SNE/UMAP computation"
            )
        
        with col2:
            clustering_backend = st.selectbox(
                "Clustering Backend",
                options=clustering_options,
                index=0,
                help="Backend for K-means clustering computation"
            )
        
        # Performance and reproducibility settings
        n_workers = st.number_input(
                "N workers", 
                min_value=1, 
                max_value=64, 
                value=8, 
                step=1,
                help="Number of parallel workers for CPU backends (sklearn, FAISS). Not used by cuML (GPU manages parallelization automatically)."
            )
        
    
    return dim_reduction_backend, clustering_backend, n_workers, seed


def render_basic_clustering_controls():
    """
    Render basic clustering parameter controls.
    
    Returns:
        Tuple of (n_clusters, reduction_method)
    """
    n_clusters = st.slider("Number of clusters", 2, 100, 5)
    reduction_method = st.selectbox("Dimensionality Reduction", ["TSNE", "PCA", "UMAP"])
    
    return n_clusters, reduction_method
