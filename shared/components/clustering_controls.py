"""
Shared clustering controls component.

Uses lazy loading to avoid importing heavy libraries at startup.
Backend availability is checked only when the user expands the backend controls.
"""

import streamlit as st
from typing import Tuple, Optional

# Lazy-loaded availability flags (None = not checked yet)
_BACKEND_AVAILABILITY: Optional[dict] = None


def _get_backend_availability() -> dict:
    """
    Lazy check for backend availability.
    Only imports libraries when this function is first called.
    """
    global _BACKEND_AVAILABILITY
    if _BACKEND_AVAILABILITY is not None:
        return _BACKEND_AVAILABILITY

    has_faiss = False
    has_cuml = False
    has_cuda = False

    # Check for FAISS (clustering only)
    try:
        import faiss
        has_faiss = True
    except ImportError:
        pass

    # Check for cuML + CUDA (both dim reduction and clustering)
    try:
        import cuml
        import cupy as cp
        has_cuml = True
        if cp.cuda.is_available():
            has_cuda = True
    except ImportError:
        pass

    _BACKEND_AVAILABILITY = {
        'has_faiss': has_faiss,
        'has_cuml': has_cuml,
        'has_cuda': has_cuda,
    }
    return _BACKEND_AVAILABILITY


def render_clustering_backend_controls():
    """
    Render clustering backend selection controls.

    Returns:
        Tuple of (dim_reduction_backend, clustering_backend, n_workers, seed)
    """
    # Backend availability is checked lazily only when user expands the backend controls
    dim_reduction_options = ["auto", "sklearn"]
    clustering_options = ["auto", "sklearn"]
    
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
        # Lazy check backend availability only when user expands this section
        avail = _get_backend_availability()

        # Update options based on availability
        if avail['has_faiss']:
            clustering_options.append("faiss")
        if avail['has_cuml'] and avail['has_cuda']:
            dim_reduction_options.append("cuml")
            clustering_options.append("cuml")

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
