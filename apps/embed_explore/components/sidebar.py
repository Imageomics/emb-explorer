"""
Sidebar components for the embed_explore application.
"""

import streamlit as st
import os
import time
import hashlib
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from shared.services.embedding_service import EmbeddingService
from shared.services.clustering_service import ClusteringService
from shared.services.file_service import FileService
from shared.lib.progress import StreamlitProgressContext
from shared.components.clustering_controls import (
    render_projection_controls,
    render_kmeans_controls,
)
from shared.utils.backend import check_cuda_available, resolve_backend, is_oom_error
from shared.utils.logging_config import get_logger

logger = get_logger(__name__)


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
                        logger.warning("Embedding generation returned 0 embeddings")
                        st.session_state.embeddings = None
                        st.session_state.valid_paths = None
                        st.session_state.labels = None
                        st.session_state.data = None
                        st.session_state.selected_image_idx = None
                    else:
                        logger.info(f"Embeddings stored: shape={embeddings.shape}, dtype={embeddings.dtype}")
                        st.success(f"Generated {embeddings.shape[0]} image embeddings.")
                        st.session_state.embeddings = embeddings
                        st.session_state.valid_paths = valid_paths
                        st.session_state.last_image_dir = image_dir
                        st.session_state.embedding_complete = True
                        # Reset projection/clustering/selection state for the new embeddings
                        st.session_state.labels = None
                        st.session_state.kmeans_column = None
                        st.session_state.data = None
                        st.session_state.selected_image_idx = None

                except Exception as e:
                    st.error(f"Error during embedding: {e}")
                    logger.exception("Embedding generation failed")

        elif embed_button:
            st.error("Please provide a valid image directory path.")

    return embed_button, image_dir, model_name, n_workers, batch_size


def render_projection_section():
    """Render the 2D projection section."""
    with st.expander("Project to 2D", expanded=False):
        embeddings = st.session_state.get("embeddings", None)
        valid_paths = st.session_state.get("valid_paths", None)

        if embeddings is None or valid_paths is None or len(valid_paths) < 2:
            st.info("Run embedding first to enable projection.")
            return

        n_samples, emb_dim = embeddings.shape
        st.markdown(f"**Ready to project:** {n_samples:,} images ({emb_dim}-dim embeddings)")

        reduction_method = st.selectbox(
            "Dimensionality Reduction",
            ["TSNE", "PCA", "UMAP"],
            help="Method to project high-dimensional embeddings to 2D for visualization.",
        )

        dim_reduction_backend, seed = render_projection_controls()

        if st.button("Project to 2D", type="primary"):
            _run_projection(embeddings, valid_paths, reduction_method, dim_reduction_backend, seed)


def render_kmeans_section():
    """Render the optional KMeans clustering section."""
    with st.expander("KMeans Clustering", expanded=False):
        df_plot = st.session_state.get("data", None)
        embeddings = st.session_state.get("embeddings", None)

        if df_plot is None or embeddings is None:
            st.info("Run projection first to enable KMeans.")
            return

        emb_dim = embeddings.shape[1]
        st.markdown(f"**{len(df_plot):,} points** ({emb_dim}-dim embeddings)")

        n_clusters = st.slider("Number of clusters", 2, min(100, max(2, len(df_plot) // 2)), 5)

        clustering_backend, n_workers, seed = render_kmeans_controls()

        if st.button("Run KMeans", type="primary"):
            _run_kmeans(embeddings, n_clusters, clustering_backend, n_workers, seed)


def _run_projection(embeddings, valid_paths, reduction_method, dim_reduction_backend, seed):
    """Run dim reduction and create the 2D scatter plot dataframe."""
    try:
        cuda_available, device_info = check_cuda_available()
        actual_backend = resolve_backend(dim_reduction_backend, "reduction")

        logger.info("=" * 60)
        logger.info("PROJECTION START")
        logger.info(f"Device: {device_info} (CUDA: {'Yes' if cuda_available else 'No'})")
        logger.info(f"Backend: {actual_backend} (requested: {dim_reduction_backend})")

        t_start = time.time()
        n_samples, emb_dim = embeddings.shape
        logger.info(f"Records: {n_samples:,} | Dim: {emb_dim}")

        with st.spinner(f"Running {reduction_method}..."):
            reduced = ClusteringService.run_dim_reduction_safe(
                embeddings, reduction_method,
                n_workers=8, dim_reduction_backend=actual_backend, seed=seed
            )

        t_total = time.time() - t_start
        logger.info(f"Projection complete in {t_total:.2f}s")

        # Build plot dataframe (no cluster column)
        df_plot = pd.DataFrame({
            "x": reduced[:, 0],
            "y": reduced[:, 1],
            "image_path": valid_paths,
            "file_name": [os.path.basename(p) for p in valid_paths],
            "idx": range(len(valid_paths)),
        })

        # Carry over any prior KMeans columns from the previous df_plot (if length matches)
        prev_df = st.session_state.get("data")
        if prev_df is not None and len(prev_df) == len(df_plot):
            for col in prev_df.columns:
                if col.startswith("KMeans (k="):
                    df_plot[col] = prev_df[col].values

        data_hash = hashlib.md5(f"{len(df_plot)}_{reduction_method}_{t_total}".encode()).hexdigest()[:8]
        st.session_state.data = df_plot
        st.session_state.data_version = data_hash
        st.session_state.selected_image_idx = None

        logger.info("=" * 60)
        st.success(f"Projected {n_samples:,} points to 2D using {reduction_method}.")

    except (RuntimeError, OSError) as e:
        if is_oom_error(e):
            st.error("**GPU Out of Memory**")
            st.info("Try: Reduce dataset size, use 'sklearn' backend, or try PCA.")
            logger.exception("GPU OOM during projection")
        else:
            st.error(f"Error during projection: {e}")
            logger.exception("Projection error")
    except MemoryError:
        st.error("**System Out of Memory** - Reduce dataset size")
        logger.exception("System memory exhausted during projection")
    except Exception as e:
        st.error(f"Error: {e}")
        logger.exception("Unexpected projection error")


def _run_kmeans(embeddings, n_clusters, clustering_backend, n_workers, seed):
    """Run KMeans on already-extracted embeddings and add labels to df_plot."""
    try:
        actual_backend = resolve_backend(clustering_backend, "clustering")
        logger.info(f"KMeans: k={n_clusters}, backend={actual_backend}")

        with st.spinner(f"Running KMeans (k={n_clusters})..."):
            labels = ClusteringService.run_kmeans_only_safe(
                embeddings, n_clusters,
                n_workers=n_workers, clustering_backend=actual_backend, seed=seed
            )

        df_plot = st.session_state.data
        kmeans_col = f"KMeans (k={n_clusters})"

        df_plot[kmeans_col] = labels.astype(str)
        st.session_state.data = df_plot
        st.session_state.labels = labels
        st.session_state.kmeans_column = kmeans_col

        # Compute clustering summary on the full embedding space.
        # Cache by kmeans_col so multiple KMeans runs can each have their own
        # summary + representatives that the user can switch between.
        logger.info("Computing clustering summary statistics...")
        summary_df, representatives = ClusteringService.generate_clustering_summary(
            embeddings, labels, df_plot
        )
        summaries = st.session_state.get("clustering_summaries", {})
        reps_by_col = st.session_state.get("clustering_representatives_by_col", {})
        summaries[kmeans_col] = summary_df
        reps_by_col[kmeans_col] = representatives
        st.session_state.clustering_summaries = summaries
        st.session_state.clustering_representatives_by_col = reps_by_col
        logger.info(f"Clustering summary computed for {kmeans_col}: {len(summary_df)} clusters")

        logger.info(f"KMeans complete: {len(np.unique(labels))} clusters")
        st.success(f"KMeans complete! {len(np.unique(labels))} clusters assigned.")

    except (RuntimeError, OSError) as e:
        if is_oom_error(e):
            st.error("**GPU Out of Memory**")
            logger.exception("GPU OOM during KMeans")
        else:
            st.error(f"Error during KMeans: {e}")
            logger.exception("KMeans error")
    except MemoryError:
        st.error("**System Out of Memory** - Reduce dataset size")
        logger.exception("System memory exhausted during KMeans")
    except Exception as e:
        st.error(f"Error: {e}")
        logger.exception("Unexpected KMeans error")


def _get_available_kmeans_cols(df_plot) -> list:
    """Return KMeans columns in df_plot sorted by k value."""
    if df_plot is None:
        return []
    return sorted(
        [c for c in df_plot.columns if c.startswith("KMeans (k=")],
        key=lambda c: int(c.split("=")[1].rstrip(")")),
    )


def render_save_section():
    """Render the save operations section of the sidebar.

    Both 'Save Images from Specific Cluster' and 'Repartition Images by Cluster'
    require at least one KMeans run. When multiple KMeans runs exist, the user
    picks which one to operate on via a shared selector at the top.
    """
    df_plot = st.session_state.get("data", None)
    kmeans_cols = _get_available_kmeans_cols(df_plot)

    if not kmeans_cols:
        st.info("Run KMeans first to enable saving by cluster.")
        return

    # Shared selector: which KMeans run drives both save operations
    default_idx = len(kmeans_cols) - 1  # most recent run
    selected_kmeans_col = st.selectbox(
        "KMeans result",
        options=kmeans_cols,
        index=default_idx,
        key="save_kmeans_selector",
        help="Pick which KMeans run to use for save / repartition.",
    )

    # --- Save images from a specific cluster utility ---
    save_status_placeholder = st.empty()
    with st.expander("Save Images from Specific Cluster", expanded=True):
        available_clusters = sorted(df_plot[selected_kmeans_col].unique(), key=lambda x: int(x))
        selected_clusters = st.multiselect(
            "Select cluster(s) to save",
            available_clusters,
            default=available_clusters[:1] if available_clusters else [],
            key="save_cluster_select",
        )
        save_dir = st.text_input(
            "Directory to save selected cluster images",
            value="cluster_selected_output",
            key="save_cluster_dir",
        )
        save_cluster_button = st.button("Save images", key="save_cluster_btn")

        if save_cluster_button and selected_clusters:
            cluster_rows = df_plot[df_plot[selected_kmeans_col].isin(selected_clusters)].copy()
            # FileService expects a 'cluster' column
            cluster_rows["cluster"] = cluster_rows[selected_kmeans_col]
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

    # --- Repartition expander and status ---
    repartition_status_placeholder = st.empty()
    with st.expander("Repartition Images by Cluster", expanded=False):
        st.markdown("**Target directory for repartitioned images (will be created):**")
        repartition_dir = st.text_input(
            "Directory",
            value="repartitioned_output",
            key="repartition_dir",
        )
        max_workers = st.number_input(
            "Number of threads (higher = faster, try 8-32)",
            min_value=1,
            max_value=64,
            value=8,
            step=1,
            key="num_threads",
        )
        repartition_button = st.button("Repartition images by cluster", key="repartition_btn")

        if repartition_button:
            df_for_repartition = df_plot.copy()
            df_for_repartition["cluster"] = df_for_repartition[selected_kmeans_col]
            with StreamlitProgressContext(
                repartition_status_placeholder,
                f"Repartition complete! Images organized in {repartition_dir}",
            ) as progress:
                try:
                    repartition_summary_df, csv_path = FileService.repartition_images_by_cluster(
                        df_for_repartition, repartition_dir, max_workers, progress_callback=progress
                    )
                    st.info(f"Summary CSV saved at {csv_path}")
                except Exception as e:
                    repartition_status_placeholder.error(f"Error repartitioning images: {e}")


def render_clustering_sidebar():
    """Render the complete sidebar with embed / project / KMeans / save sections."""
    tab_compute, tab_save = st.tabs(["Compute", "Save"])

    with tab_compute:
        render_embedding_section()
        render_projection_section()
        render_kmeans_section()

    with tab_save:
        render_save_section()
