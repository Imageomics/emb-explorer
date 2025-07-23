"""
Clustering page for the embedding explorer.
"""

import streamlit as st
import os

from components.clustering.sidebar import render_clustering_sidebar
from components.clustering.visualization import render_scatter_plot, render_image_preview
from components.clustering.summary import render_clustering_summary


def main():
    """Main clustering page function."""
    st.set_page_config(
        layout="wide",
        page_title="Image Clustering",
        page_icon="🔍"
    )
    
    st.title("🔍 Image Clustering")
    
    # Create the main layout
    col_settings, col_plot, col_preview = st.columns([2, 6, 3])
    
    with col_settings:
        # Render the sidebar with all controls
        sidebar_state = render_clustering_sidebar()
    
    with col_plot:
        # Render the main scatter plot
        render_scatter_plot()
    
    with col_preview:
        # Render the image preview
        render_image_preview()
    
    # Bottom section: Clustering summary
    st.markdown("---")
    render_clustering_summary()


if __name__ == "__main__":
    main()
