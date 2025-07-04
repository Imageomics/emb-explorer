import streamlit as st
import torch
import numpy as np
import clip
import open_clip
import os
import pandas as pd
import altair as alt
import concurrent.futures

from utils.io import list_image_files, copy_image
from utils.clustering import run_kmeans, reduce_dim
from utils.dataset import ImageFolderDataset

@st.cache_resource(show_spinner=True)
def load_clip_model(name, device):
    if name == "CLIP ViT-B/32":
        model, preprocess = clip.load("ViT-B/32", device=device)
    elif name == "BioCLIP-2":
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2", device=device
        )
    model = torch.compile(model.to(device))
    return model, preprocess


@torch.no_grad()
def main():
    st.set_page_config(
        layout="wide",
        page_title="Image Clustering Tool",
    )

    col_settings, col_plot, col_preview = st.columns([2, 6, 3])
    
    with st.container():
        with col_settings:
            tab_compute, tab_save = st.tabs(["Compute", "Save"])
            with tab_compute:
                with st.expander("Embed", expanded=True):
                    image_dir = st.text_input("Image folder path")
                    model_name = st.selectbox("Model", ["CLIP ViT-B/32", "BioCLIP-2"])
                    #device = st.selectbox("Device", ["cuda", "cpu"])
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
                    
                with st.expander("Cluster", expanded=False):
                    n_clusters = st.slider("Number of clusters", 2, 100, 5)
                    reduction_method = st.selectbox("Dimensionality Reduction", ["TSNE", "PCA", "UMAP"])
                    cluster_button = st.button("Run Clustering")
            
            with tab_save:
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
                            default=available_clusters[:1],
                            key="save_cluster_select"
                        )
                        save_dir = st.text_input(
                            "Directory to save selected cluster images",
                            value="cluster_selected_output",
                            key="save_cluster_dir"
                        )
                        save_cluster_button = st.button("Save images", key="save_cluster_btn")
                    else:
                        st.info("Run clustering first to enable this utility.")
                        
                # --- Repartition expander and status ---
                repartition_status_placeholder = st.empty()  # For progress and status
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
                
                
             
            # --- Save images from specific cluster logic ---
            if 'save_cluster_button' not in st.session_state:
                st.session_state['save_cluster_button'] = False
            if 'save_cluster_btn' in st.session_state and st.session_state['save_cluster_btn']:
                st.session_state['save_cluster_button'] = True
            if 'save_cluster_button' in st.session_state and st.session_state['save_cluster_button']:
                df_plot = st.session_state.get("data", None)
                if df_plot is not None:
                    selected_clusters = st.session_state.get("save_cluster_select", [])
                    save_dir = st.session_state.get("save_cluster_dir", "cluster_selected_output")
                    if selected_clusters:
                        cluster_rows = df_plot[df_plot['cluster'].isin(selected_clusters)]
                        os.makedirs(save_dir, exist_ok=True)
                        save_rows = []
                        progress_bar = save_status_placeholder.progress(0, text="Copying images...")
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = [
                                executor.submit(copy_image, row, save_dir)
                                for idx, row in cluster_rows.iterrows()
                            ]
                            total_files = len(futures)
                            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                                result = future.result()
                                if result is not None:
                                    save_rows.append(result)
                                if i % 50 == 0 or i == total_files:
                                    progress_bar.progress(i / total_files, text=f"Copied {i} / {total_files} images")
                        save_summary_df = pd.DataFrame(save_rows)
                        csv_path = os.path.join(save_dir, "saved_cluster_summary.csv")
                        save_summary_df.to_csv(csv_path, index=False)
                        save_status_placeholder.success(
                            f"Images from cluster(s) {', '.join(map(str, selected_clusters))} saved in {save_dir}. Summary CSV at {csv_path}"
                        )
                        #st.dataframe(save_summary_df.head(10), use_container_width=True)
                    else:
                        save_status_placeholder.warning("Please select at least one cluster.")
                st.session_state['save_cluster_button'] = False
            
            # --- Repartitioning logic OUTSIDE the expander ---
            if repartition_button:
                df_plot = st.session_state.get("data", None)
                if df_plot is None or len(df_plot) < 1:
                    repartition_status_placeholder.warning("Please run clustering first before repartitioning images.")
                else:
                    os.makedirs(repartition_dir, exist_ok=True)
                    repartition_rows = []
                    progress_bar = repartition_status_placeholder.progress(0, text="Starting repartitioning...")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(copy_image, row, repartition_dir)
                            for idx, row in df_plot.iterrows()
                        ]
                        total_files = len(futures)
                        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                            result = future.result()
                            if result is not None:
                                repartition_rows.append(result)
                            if i % 100 == 0 or i == total_files:
                                progress_bar.progress(i / total_files, text=f"Repartitioned {i} / {total_files} images")
                    repartition_summary_df = pd.DataFrame(repartition_rows)
                    csv_path = os.path.join(repartition_dir, "cluster_summary.csv")
                    repartition_summary_df.to_csv(csv_path, index=False)
                    repartition_status_placeholder.success(
                        f"Repartition complete! Images organized in {repartition_dir}. Summary CSV at {csv_path}"
                    )
                    #st.dataframe(repartition_summary_df.head(10), use_container_width=True)

        # -------- Embedding Step --------
        if embed_button and image_dir and os.path.isdir(image_dir):
            st.write("Listing images...")
            image_paths = list_image_files(image_dir)
            st.write(f"Found {len(image_paths)} images.")

            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = load_clip_model(model_name, torch_device)
            
            # Create dataset & DataLoader
            dataset = ImageFolderDataset(image_dir, transform=preprocess)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=n_workers,
                pin_memory=True
            )

            progress_bar = st.progress(0, text="Embedding images...")
            status_text = st.empty()
            total = len(image_paths)

            valid_paths = []
            embeddings = []
            
            processed = 0
            for batch_paths, batch_imgs in dataloader:
                batch_imgs = batch_imgs.to(torch_device, non_blocking=True)
                batch_embeds = model.encode_image(batch_imgs).cpu().numpy()
                embeddings.append(batch_embeds)
                valid_paths.extend(batch_paths)
                processed += len(batch_paths)
                progress = processed / total
                progress_bar.progress(progress, text=f"Embedding {processed}/{total}")
            status_text.write(f"Embedding {processed}/{total}")

            progress_bar.empty()
            status_text.empty()

            # Stack embeddings if available
            if embeddings:
                embeddings = np.vstack(embeddings)
            else:
                embeddings = np.empty((0, model.visual.output_dim))

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

        # -------- Clustering & Altair Plot Step --------
        with col_plot:
            embeddings = st.session_state.get("embeddings", None)
            valid_paths = st.session_state.get("valid_paths", None)
            labels = st.session_state.get("labels", None)
            df_plot = st.session_state.get("data", None)
            selected_idx = st.session_state.get("selected_image_idx", 0)

            # Run clustering only if requested
            if cluster_button and embeddings is not None and valid_paths is not None and len(valid_paths) > 1:
                reduced = reduce_dim(embeddings, reduction_method, n_workers=n_workers)
                kmeans, labels = run_kmeans(reduced, int(n_clusters))
                df_plot = pd.DataFrame({
                    "x": reduced[:, 0],
                    "y": reduced[:, 1],
                    "cluster": labels.astype(str),
                    "image_path": valid_paths,
                    "file_name": [os.path.basename(p) for p in valid_paths],
                    "idx": range(len(valid_paths))
                })
                # Store everything in session state for reruns
                st.session_state.data = df_plot
                st.session_state.labels = labels
                st.session_state.selected_image_idx = 0  # Reset selection

            # Always use the session state DataFrame
            df_plot = st.session_state.get("data", None)
            labels = st.session_state.get("labels", None)
            selected_idx = st.session_state.get("selected_image_idx", 0)

            if df_plot is not None and len(df_plot) > 1:
                point_selector = alt.selection_point(fields=["idx"], name="point_selection")
                scatter = (
                    alt.Chart(df_plot)
                    .mark_circle(size=60)
                    .encode(
                        x=alt.X('x', scale=alt.Scale(zero=False)),
                        y=alt.Y('y', scale=alt.Scale(zero=False)),
                        color=alt.Color('cluster:N', legend=alt.Legend(title="Cluster")),
                        tooltip=['file_name', 'cluster'],
                        fillOpacity=alt.condition(point_selector, alt.value(1), alt.value(0.3))
                    )
                    .add_params(point_selector)
                    .properties(
                        width=800,
                        height=700,
                        title="Image Clusters (click a point to preview image)"
                    )
                )
                event = st.altair_chart(scatter, key="alt_chart", on_select="rerun", use_container_width=True)

                # Handle updated event format
                if (
                    event
                    and "selection" in event
                    and "point_selection" in event["selection"]
                    and event["selection"]["point_selection"]
                ):
                    new_idx = int(event["selection"]["point_selection"][0]["idx"])
                    st.session_state["selected_image_idx"] = new_idx
                    selected_idx = new_idx

            else:
                st.info("Run clustering to see the cluster scatter plot.")
                st.session_state['selected_image_idx'] = None

        # -------- Image Preview Panel (Right) --------
        with col_preview:
            valid_paths = st.session_state.get("valid_paths", None)
            labels = st.session_state.get("labels", None)
            selected_idx = st.session_state.get("selected_image_idx", 0)
            if (
                valid_paths is not None and
                labels is not None and
                selected_idx is not None and
                0 <= selected_idx < len(valid_paths)
            ):
                img_path = valid_paths[selected_idx]
                cluster = labels[selected_idx] if labels is not None else "?"
                st.image(img_path, caption=f"Cluster {cluster}: {os.path.basename(img_path)}", use_container_width=True)
                st.markdown(f"**File:** `{os.path.basename(img_path)}`")
                st.markdown(f"**Cluster:** `{cluster}`")
            else:
                st.info("Image preview will appear here after you select a cluster point.")
        
    # ---- Bottom Row: Clustering Summary Panel ----
    st.markdown("---")
    with st.container():
        df_plot = st.session_state.get("data", None)
        labels = st.session_state.get("labels", None)
        embeddings = st.session_state.get("embeddings", None)

        if df_plot is not None and labels is not None and embeddings is not None:
            st.subheader("Clustering Summary")

            cluster_ids = np.unique(labels)
            summary_data = []

            centroids = {}
            representatives = {}

            for k in cluster_ids:
                idxs = np.where(labels == k)[0]
                cluster_embeds = embeddings[idxs]
                centroid = cluster_embeds.mean(axis=0)
                centroids[k] = centroid

                # Internal variance
                variance = np.mean(np.sum((cluster_embeds - centroid) ** 2, axis=1))

                # Find 3 closest images
                dists = np.sum((cluster_embeds - centroid) ** 2, axis=1)
                closest_indices = idxs[np.argsort(dists)[:3]]
                representatives[k] = closest_indices

                summary_data.append({
                    "Cluster": int(k),
                    "Count": len(idxs),
                    "Variance": round(variance, 3),
                })

            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

            st.markdown("#### Representative Images")
            for row in summary_df.itertuples():
                k = row.Cluster
                st.markdown(f"**Cluster {k}**")
                img_cols = st.columns(3)
                for i, img_idx in enumerate(representatives[k]):
                    img_path = df_plot.iloc[img_idx]["image_path"]
                    img_cols[i].image(img_path, use_container_width = True, caption=os.path.basename(img_path))
        else:
            st.info("Clustering summary table will appear here after clustering.")

if __name__ == "__main__":
    main()
