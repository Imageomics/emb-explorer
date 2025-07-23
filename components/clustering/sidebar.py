"""
Sidebar components for the clustering page.
"""

import streamlit as st
import os
from typing import Tuple, List, Optional

from services.embedding_service import EmbeddingService
from services.clustering_service import ClusteringService
from services.file_service import FileService
from lib.progress import StreamlitProgressContext


def render_embedding_section() -> Tuple[bool, Optional[str], Optional[str], int, int]:
    """
    Render the embedding section of the sidebar.
    
    Returns:
        Tuple of (embed_button_clicked, image_dir, model_name, n_workers, batch_size)
    """
    with st.expander("Embed", expanded=True):
        image_dir = st.text_input("Image folder path")
        
        # Get available models dynamically
        available_models = EmbeddingService.get_model_options()
        model_name = st.selectbox("Model", available_models)
        
        col1, col2 = st.columns(2)
        with col1:
            n_workers = st.number_input(
                "N workers", 
                min_value=1, 
                max_value=64, 
                value=16, 
                step=1
            )
        with col2:
            batch_size = st.number_input(
                "Batch size", 
                min_value=1, 
                max_value=2048, 
                value=32, 
                step=1
            )
        embed_button = st.button("Run Embedding")
        
        # Handle embedding execution
        if embed_button and image_dir and os.path.isdir(image_dir):
            with StreamlitProgressContext(st.empty(), "Embedding complete!") as progress:
                try:
                    embeddings, valid_paths = EmbeddingService.generate_embeddings(
                        image_dir, model_name, batch_size, n_workers,
                        progress_callback=progress
                    )
                    
                    if embeddings.shape[0] == 0:
                        st.error("No valid image embeddings found.")
                        st.session_state.embeddings = None
                        st.session_state.valid_paths = None
                        st.session_state.labels = None
                        st.session_state.data = None
                        st.session_state.selected_image_idx = None
                    else:
                        st.success(f"Generated {embeddings.shape[0]} image embeddings.")
                        st.session_state.embeddings = embeddings
                        st.session_state.valid_paths = valid_paths
                        st.session_state.last_image_dir = image_dir
                        st.session_state.embedding_complete = True
                        # Reset clustering/selection state
                        st.session_state.labels = None
                        st.session_state.data = None
                        st.session_state.selected_image_idx = 0
                        
                except Exception as e:
                    st.error(f"Error during embedding: {e}")
        
        elif embed_button:
            st.error("Please provide a valid image directory path.")
    
    return embed_button, image_dir, model_name, n_workers, batch_size


def render_clustering_section() -> Tuple[bool, int, str]:
    """
    Render the clustering section of the sidebar.
    
    Returns:
        Tuple of (cluster_button_clicked, n_clusters, reduction_method)
    """
    with st.expander("Cluster", expanded=False):
        n_clusters = st.slider("Number of clusters", 2, 100, 5)
        reduction_method = st.selectbox("Dimensionality Reduction", ["TSNE", "PCA", "UMAP"])
        cluster_button = st.button("Run Clustering")
        
        # Handle clustering execution
        if cluster_button:
            embeddings = st.session_state.get("embeddings", None)
            valid_paths = st.session_state.get("valid_paths", None)
            
            if embeddings is not None and valid_paths is not None and len(valid_paths) > 1:
                try:
                    with st.spinner("Running clustering..."):
                        df_plot, labels = ClusteringService.run_clustering(
                            embeddings, valid_paths, n_clusters, reduction_method
                        )
                    
                    # Store everything in session state for reruns
                    st.session_state.data = df_plot
                    st.session_state.labels = labels
                    st.session_state.selected_image_idx = 0  # Reset selection
                    st.success(f"Clustering complete! Found {n_clusters} clusters.")
                    
                except Exception as e:
                    st.error(f"Error during clustering: {e}")
            else:
                st.error("Please run embedding first.")
    
    return cluster_button, n_clusters, reduction_method


def render_save_section():
    """Render the save operations section of the sidebar."""
    # --- Save images from a specific cluster utility ---
    save_status_placeholder = st.empty()
    with st.expander("Save Images from Specific Cluster", expanded=True):
        df_plot = st.session_state.get("data", None)
        labels = st.session_state.get("labels", None)
        
        if df_plot is not None and labels is not None:
            available_clusters = sorted(df_plot['cluster'].unique(), key=lambda x: int(x))
            selected_clusters = st.multiselect(
                "Select cluster(s) to save",
                available_clusters,
                default=available_clusters[:1] if available_clusters else [],
                key="save_cluster_select"
            )
            save_dir = st.text_input(
                "Directory to save selected cluster images",
                value="cluster_selected_output",
                key="save_cluster_dir"
            )
            save_cluster_button = st.button("Save images", key="save_cluster_btn")
            
            # Handle save execution
            if save_cluster_button and selected_clusters:
                cluster_rows = df_plot[df_plot['cluster'].isin(selected_clusters)]
                max_workers = st.session_state.get("num_threads", 8)
                
                with StreamlitProgressContext(
                    save_status_placeholder, 
                    f"Images from cluster(s) {', '.join(map(str, selected_clusters))} saved successfully!"
                ) as progress:
                    try:
                        save_summary_df, csv_path = FileService.save_cluster_images(
                            cluster_rows, save_dir, max_workers, progress_callback=progress
                        )
                        st.info(f"Summary CSV saved at {csv_path}")
                        
                    except Exception as e:
                        save_status_placeholder.error(f"Error saving images: {e}")
            
            elif save_cluster_button:
                save_status_placeholder.warning("Please select at least one cluster.")
                
        else:
            st.info("Run clustering first to enable this utility.")
                
    # --- Repartition expander and status ---
    repartition_status_placeholder = st.empty()
    with st.expander("Repartition Images by Cluster", expanded=False):
        st.markdown("**Target directory for repartitioned images (will be created):**")
        repartition_dir = st.text_input(
            "Directory", 
            value="repartitioned_output",
            key="repartition_dir"
        )
        max_workers = st.number_input(
            "Number of threads (higher = faster, try 8–32)", 
            min_value=1, 
            max_value=64, 
            value=8,
            step=1,
            key="num_threads"
        )
        repartition_button = st.button("Repartition images by cluster", key="repartition_btn")
        
        # Handle repartition execution
        if repartition_button:
            df_plot = st.session_state.get("data", None)
            
            if df_plot is None or len(df_plot) < 1:
                repartition_status_placeholder.warning("Please run clustering first before repartitioning images.")
            else:
                with StreamlitProgressContext(
                    repartition_status_placeholder,
                    f"Repartition complete! Images organized in {repartition_dir}"
                ) as progress:
                    try:
                        repartition_summary_df, csv_path = FileService.repartition_images_by_cluster(
                            df_plot, repartition_dir, max_workers, progress_callback=progress
                        )
                        st.info(f"Summary CSV saved at {csv_path}")
                        
                    except Exception as e:
                        repartition_status_placeholder.error(f"Error repartitioning images: {e}")


def render_clustering_sidebar():
    """Render the complete clustering sidebar with all sections."""
    tab_compute, tab_save = st.tabs(["Compute", "Save"])
    
    with tab_compute:
        embed_button, image_dir, model_name, n_workers, batch_size = render_embedding_section()
        cluster_button, n_clusters, reduction_method = render_clustering_section()
    
    with tab_save:
        render_save_section()
    
    return {
        'embed_button': embed_button,
        'image_dir': image_dir,
        'model_name': model_name,
        'n_workers': n_workers,
        'batch_size': batch_size,
        'cluster_button': cluster_button,
        'n_clusters': n_clusters,
        'reduction_method': reduction_method,
    }
