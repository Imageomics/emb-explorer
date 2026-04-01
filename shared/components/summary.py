"""
Shared clustering summary components.
"""

import streamlit as st
import os
import pandas as pd
from shared.utils.taxonomy_tree import build_taxonomic_tree, format_tree_string, get_tree_statistics
from shared.utils.logging_config import get_logger

logger = get_logger(__name__)


def render_taxonomic_tree_summary():
    """Render taxonomic tree summary for precalculated embeddings."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    filtered_df = st.session_state.get("filtered_df_for_clustering", None)

    if df_plot is not None and filtered_df is not None:
        st.markdown("### Taxonomic Distribution")

        # Detect available KMeans columns
        kmeans_cols = sorted(
            [c for c in df_plot.columns if c.startswith("KMeans (k=")],
            key=lambda c: int(c.split("=")[1].rstrip(")"))
        )
        # Fallback for embed_explore app (has 'cluster' column directly)
        has_embed_explore_cluster = 'cluster' in df_plot.columns and not kmeans_cols

        # Add controls at the top of the taxonomy section
        col1, col2, col3, col4 = st.columns([1.5, 1.5, 1, 1])

        with col1:
            if kmeans_cols:
                # Precalculated app: let user pick which KMeans run
                group_by = st.selectbox(
                    "Group by",
                    options=["(none)"] + kmeans_cols,
                    index=0,
                    key="taxonomy_group_by",
                    help="Select a KMeans result to filter taxonomy by cluster"
                )
                if group_by == "(none)":
                    group_by = None
            elif has_embed_explore_cluster:
                group_by = "cluster"
            else:
                group_by = None

        with col2:
            if group_by and group_by in df_plot.columns:
                unique_clusters = sorted(df_plot[group_by].unique(), key=lambda x: int(x))
                cluster_options = ["All"] + [str(c) for c in unique_clusters]
                selected_cluster = st.selectbox(
                    "Cluster",
                    options=cluster_options,
                    index=0,
                    key="taxonomy_cluster_selector",
                    help="Select a specific cluster or 'All'"
                )
            else:
                selected_cluster = "All"

        with col2:
            min_count = st.number_input(
                "Minimum count",
                min_value=1,
                max_value=1000,
                value=5,
                step=1,
                key="taxonomy_min_count",
                help="Minimum number of records for a taxon to appear in the tree"
            )

        with col3:
            tree_depth = st.slider(
                "Tree depth",
                min_value=1,
                max_value=7,
                value=7,
                key="taxonomy_tree_depth",
                help="Maximum depth of the taxonomy tree to display"
            )

        # Create a stable cache key
        data_length = len(filtered_df)
        sample_uuids = filtered_df['uuid'].iloc[:min(10, len(filtered_df))].tolist()
        data_id = f"{data_length}_{len(sample_uuids)}_{sample_uuids[0] if sample_uuids else 'empty'}"
        cache_key = f"taxonomy_{data_id}_{group_by}_{selected_cluster}_{min_count}_{tree_depth}"

        current_cache_key = st.session_state.get("taxonomy_cache_key")
        cache_exists = cache_key in st.session_state

        if (not cache_exists or current_cache_key != cache_key):

            with st.spinner("Building taxonomy tree..."):
                # Filter data based on group_by + selected_cluster
                if group_by and selected_cluster != "All" and group_by in df_plot.columns:
                    cluster_mask = df_plot[group_by] == selected_cluster
                    cluster_uuids = df_plot[cluster_mask]['uuid'].tolist()
                    tree_df = filtered_df[filtered_df['uuid'].isin(cluster_uuids)]
                    display_title = f"Taxonomic Tree for {group_by} = {selected_cluster}"
                else:
                    tree_df = filtered_df
                    display_title = "Taxonomic Tree for All Data"

                # Build taxonomic tree for the selected data (only when needed)
                tree = build_taxonomic_tree(tree_df)
                stats = get_tree_statistics(tree)
                tree_string = format_tree_string(tree, max_depth=tree_depth, min_count=min_count)

                # Cache the results
                st.session_state[cache_key] = {
                    'tree': tree,
                    'stats': stats,
                    'tree_string': tree_string,
                    'display_title': display_title
                }
                st.session_state["taxonomy_cache_key"] = cache_key

        # Use cached results (no regeneration)
        cached_data = st.session_state[cache_key]

        # Show statistics
        st.markdown(f"**{cached_data['display_title']}**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{cached_data['stats']['total_records']:,}")
        with col2:
            st.metric("Kingdoms", cached_data['stats']['kingdoms'])
        with col3:
            st.metric("Families", cached_data['stats']['families'])
        with col4:
            st.metric("Species", cached_data['stats']['species'])

        # Display the tree
        if cached_data['tree_string']:
            st.code(cached_data['tree_string'], language="text")
        else:
            st.info("No taxonomic data meets the display criteria. Try lowering the minimum count.")


def render_clustering_summary(show_taxonomy=False):
    """Render the clustering summary panel using cached results from clustering action."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)

    # Get pre-computed summary from session state (computed when clustering was run)
    summary_df = st.session_state.get("clustering_summary", None)
    representatives = st.session_state.get("clustering_representatives", None)

    if df_plot is not None:
        has_images = 'image_path' in df_plot.columns

        if has_images:
            # embed_explore app: show full clustering summary with representative images
            if labels is not None:
                st.subheader("Clustering Summary")

                if summary_df is not None and representatives is not None:
                    logger.debug("Displaying cached clustering summary")
                    st.dataframe(summary_df, hide_index=True, width='stretch')

                    st.markdown("#### Representative Images")
                    for row in summary_df.itertuples():
                        k = row.Cluster
                        st.markdown(f"**Cluster {k}**")
                        img_cols = st.columns(3)
                        for i, img_idx in enumerate(representatives[k]):
                            img_path = df_plot.iloc[img_idx]["image_path"]
                            logger.debug(f"Displaying representative image: {img_path}")
                            img_cols[i].image(
                                img_path,
                                width='stretch',
                                caption=os.path.basename(img_path)
                            )
                else:
                    st.info("Clustering summary will be computed when you run clustering.")
        else:
            # Precalculated app: show taxonomy tree (works with or without KMeans)
            if show_taxonomy:
                filtered_df = st.session_state.get("filtered_df_for_clustering", None)
                if filtered_df is not None:
                    render_taxonomic_tree_summary()

    else:
        st.info("Summary will appear here after projection.")
