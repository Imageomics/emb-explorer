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
    
    Navigate to the **🔍 Clustering** page in the sidebar to:
    - Generate embeddings for your image datasets
    - Run clustering analysis with different algorithms  
    - Visualize clusters in interactive plots
    - Save and repartition images by cluster
    
    ## 📊 Features
    
    - **Multiple Models**: Support for CLIP, BioCLIP, and OpenCLIP models
    - **Flexible Clustering**: PCA, t-SNE, and UMAP dimensionality reduction
    - **Interactive Visualization**: Click on points to preview images
    - **Batch Operations**: Save specific clusters or repartition all images
    - **Progress Tracking**: Real-time progress bars for long operations
    
    Select a page from the sidebar to begin!
    """)

if __name__ == "__main__":
    main()
