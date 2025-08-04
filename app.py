import streamlit as st

def main():
    """Main application entry point."""
    st.set_page_config(
        layout="wide",
        page_title="emb-explorer",
        page_icon="🔍"
    )
    
    # Welcome page content
    st.title("🔍 emb-explorer")
    st.markdown("**Visual exploration and clustering tool for image datasets and pre-calculated image embeddings**")
    
    st.markdown("---")
    
    # Two-column layout to match README structure
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Embed & Explore Images")
        st.markdown("**Upload and process your own image datasets**")
        
        st.markdown("""
        **🔋 Key Features:**
        - **Batch Image Embedding**: Process large image collections using pre-trained models (CLIP, BioCLIP, OpenCLIP)
        - **Multi-Model Support**: Choose from various vision-language models optimized for different domains
        - **K-Means Analysis**: Clustering with customizable KMeans parameters
        - **Interactive Clustering**: Explore data with PCA, t-SNE, and UMAP dimensionality reduction
        - **Cluster Repartitioning**: Organize images into cluster-specific folders with one click
        - **Summary Statistics**: Analyze cluster quality with size, variance, and representative samples
        """)
        
        
    
    with col2:
        st.markdown("### 📊 Explore Pre-calculated Embeddings")
        st.markdown("**Work with existing embeddings and rich metadata**")
        
        st.markdown("""
        **🔍 Key Features:**
        - **Parquet File Support**: Load precomputed embeddings with associated metadata
        - **Advanced Filtering**: Filter by custom metadata
        - **K-Means Analysis**: Clustering with customizable KMeans parameters
        - **Interactive Clustering**: Explore data with PCA and UMAP dimensionality reduction
        - **Taxonomy Tree Navigation**: Browse hierarchical taxonomy classifications with interactive tree view
        """)
        
        
    st.markdown("---")
    
    # Getting started section
    st.markdown("## 🚀 Getting Started")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Choose Your Workflow:**
        
        **For New Images** → Use **Clustering** page
        - Upload your image folder
        - Select embedding model  
        - Generate embeddings and explore clusters
        
        **For Existing Data** → Use **Precalculated Embeddings** page
        - Load your parquet file
        - Apply filters and explore patterns
        - Perform targeted clustering analysis
        """)
    
    with col2:
        st.markdown("""
        **⚡ Technical Capabilities:**
        
        - **Models**: CLIP, BioCLIP-2, OpenCLIP variants
        - **Acceleration**: CPU and GPU (CUDA) support
        - **Formats**: Images (PNG, JPG, etc.), Parquet files
        - **Clustering**: K-Means with multiple initialization methods
        - **Visualization**: Interactive scatter plots with image preview
        - **Export**: CSV summaries, folder organization, filtered datasets
        """)
    
    st.markdown("---")
    
    # Navigation help
    st.markdown("### 📋 Navigation")
    st.markdown("""
    Use the **sidebar navigation** to select your workflow:
    - **🔍 Clustering**: Process and explore new image datasets
    - **📊 Precalculated Embeddings**: Analyze existing embeddings with metadata filtering
    
    Each page provides step-by-step guidance and real-time feedback for your analysis workflow.
    """)
    
    # Quick tips
    with st.expander("💡 Pro Tips"):
        st.markdown("""
        - **GPU Acceleration**: Install with `uv pip install -e ".[gpu]"` for faster processing
        - **Large Datasets**: Use batch processing and monitor memory usage in the sidebar
        - **Custom Filtering**: Combine multiple filter criteria for precise data selection
        - **Export Results**: Save cluster summaries and repartitioned images for downstream analysis
        """)

if __name__ == "__main__":
    main()
