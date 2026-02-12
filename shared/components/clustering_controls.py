"""
Shared clustering controls component.
"""

import streamlit as st
from typing import Tuple, Optional

from shared.utils.backend import HAS_FAISS_PACKAGE, HAS_CUML_PACKAGE, HAS_CUPY_PACKAGE


def render_clustering_backend_controls():
    """
    Render clustering backend selection controls.

    Returns:
        Tuple of (dim_reduction_backend, clustering_backend, n_workers, seed)
    """
    # Backend availability detection — uses find_spec() flags (instant, no heavy imports)
    dim_reduction_options = ["auto", "sklearn"]
    clustering_options = ["auto", "sklearn"]

    if HAS_FAISS_PACKAGE:
        clustering_options.append("faiss")

    if HAS_CUML_PACKAGE and HAS_CUPY_PACKAGE:
        dim_reduction_options.append("cuml")
        clustering_options.append("cuml")
    
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
