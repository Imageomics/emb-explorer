"""
Precalculated Embeddings page for the embedding explorer.
Works with parquet files containing precomputed embeddings and metadata.
"""

import streamlit as st

from components.precalculated.sidebar import render_precalculated_sidebar
from components.clustering.visualization import render_scatter_plot
from components.precalculated.data_preview import render_data_preview
from components.clustering.summary import render_clustering_summary


def main():
    """Main precalculated embeddings page function."""
    st.set_page_config(
        layout="wide",
        page_title="Precalculated Embeddings",
        page_icon="📊"
    )
    
    # Clear clustering page data to prevent carry-over
    if "page_type" not in st.session_state or st.session_state.page_type != "precalculated":
        # Clear regular clustering data
        clustering_keys = ["embeddings", "valid_paths", "last_image_dir", "embedding_complete"]
        for key in clustering_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.page_type = "precalculated"
    
    st.title("📊 Precalculated Embeddings")
    st.markdown("Load and cluster precomputed embeddings from parquet files with metadata filtering.")
    
    # Create the main layout
    col_settings, col_plot, col_preview = st.columns([2, 6, 3])
    
    with col_settings:
        # Render the sidebar with all controls
        sidebar_state = render_precalculated_sidebar()
    
    with col_plot:
        # Render the main scatter plot (reused from clustering page)
        render_scatter_plot()
    
    with col_preview:
        # Render the data preview (metadata instead of images)
        render_data_preview()
    
    # Bottom section: Clustering summary with taxonomy tree
    st.markdown("---")
    render_clustering_summary(show_taxonomy=True)


if __name__ == "__main__":
    main()
