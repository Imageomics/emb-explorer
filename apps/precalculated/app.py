"""
Precalculated Embeddings Explorer - Standalone Application

A Streamlit application for exploring precomputed embeddings stored in parquet files.
Features dynamic filter generation based on available columns.
"""

import streamlit as st

from apps.precalculated.components.sidebar import (
    render_file_section,
    render_dynamic_filters,
    render_clustering_section,
)
from apps.precalculated.components.data_preview import render_data_preview
from shared.components.visualization import render_scatter_plot
from shared.components.summary import render_clustering_summary


def main():
    """Main application entry point."""
    st.set_page_config(
        layout="wide",
        page_title="Precalculated Embeddings Explorer",
        page_icon="📊"
    )

    # Initialize session state
    if "page_type" not in st.session_state or st.session_state.page_type != "precalculated_app":
        # Clear any stale state from other apps
        keys_to_clear = ["embeddings", "valid_paths", "last_image_dir", "embedding_complete"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page_type = "precalculated_app"

    # Header
    st.title("📊 Precalculated Embeddings Explorer")
    st.markdown(
        "Load parquet files with embeddings, apply dynamic filters, and cluster for visualization. "
        "Filters are automatically generated based on your data columns."
    )

    # Row 1: File loading
    render_file_section()

    # Row 2: Dynamic filters
    render_dynamic_filters()

    # Row 3: Main content
    col_settings, col_plot, col_preview = st.columns([2, 7, 3])

    with col_settings:
        render_clustering_section()

    with col_plot:
        render_scatter_plot()

    with col_preview:
        render_data_preview()

    # Bottom: Clustering summary
    st.markdown("---")
    render_clustering_summary(show_taxonomy=True)


if __name__ == "__main__":
    main()
