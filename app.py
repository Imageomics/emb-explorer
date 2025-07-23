import streamlit as st

def main():
    """Main application entry point."""
    st.set_page_config(
        layout="wide",
        page_title="Embedding Explorer",
        page_icon="🔍"
    )
    
    # Welcome page content
    st.title("🔍 Embedding Explorer")
    st.markdown("""
    Welcome to the Embedding Explorer! This tool helps you visualize and cluster image datasets using various embedding models.
    
    ## 🚀 Getting Started
    
    Choose from two main workflows:
    
    ### 🔍 Clustering Page
    - Generate embeddings for your image datasets
    - Run clustering analysis with different algorithms  
    - Visualize clusters in interactive plots
    - Save and repartition images by cluster
    
    ### 📊 Precalculated Embeddings Page
    - Load parquet files with precomputed embeddings and metadata
    - Filter data based on taxonomic and source metadata
    - Cluster and visualize filtered embeddings
    - Explore biological/taxonomic patterns in your data
    
    ## 📊 Features
    
    - **Multiple Models**: Support for CLIP, BioCLIP, and OpenCLIP models
    - **Flexible Clustering**: PCA, t-SNE, and UMAP dimensionality reduction
    - **Interactive Visualization**: Click on points to preview images or view metadata
    - **Metadata Filtering**: Filter by taxonomic hierarchy, source datasets, and more
    - **Batch Operations**: Save specific clusters or repartition all images
    - **Progress Tracking**: Real-time progress bars for long operations
    
    Select a page from the sidebar to begin!
    """)

if __name__ == "__main__":
    main()
