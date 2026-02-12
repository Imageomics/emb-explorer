"""
BYO Images Embed & Explore application.

This application allows users to bring their own images, generate embeddings,
cluster them, and explore the results visually.
"""

import streamlit as st

from apps.embed_explore.components.sidebar import render_clustering_sidebar
from apps.embed_explore.components.image_preview import render_image_preview
from shared.components.summary import render_clustering_summary
from shared.components.visualization import render_scatter_plot


def main():
    """Main application entry point."""
    st.set_page_config(
        layout="wide",
        page_title="Embed & Explore",
        page_icon="🔍"
    )

    st.title("🔍 Embed & Explore")
    st.markdown("Generate embeddings from your images, cluster them, and explore the results.")

    # Create the main layout
    col_settings, col_plot, col_preview = st.columns([2, 6, 3])

    with col_settings:
        # Render the sidebar with all controls
        render_clustering_sidebar()

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
