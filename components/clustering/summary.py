"""
Clustering summary components.
"""

import streamlit as st
import os
import pandas as pd
from services.clustering_service import ClusteringService
from utils.taxonomy_tree import build_taxonomic_tree, format_tree_string, get_tree_statistics


def render_taxonomic_tree_summary():
    """Render taxonomic tree summary for precalculated embeddings."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    filtered_df = st.session_state.get("filtered_df_for_clustering", None)
    
    if df_plot is not None and labels is not None and filtered_df is not None:
        st.markdown("### 🌳 Taxonomic Distribution")
        
        # Add controls at the top of the taxonomy section
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Get available clusters
            cluster_options = ["All"]
            if "cluster" in df_plot.columns:
                unique_clusters = sorted(df_plot["cluster"].unique(), key=lambda x: int(x))
                cluster_options.extend([f"Cluster {c}" for c in unique_clusters])
            
            selected_cluster = st.selectbox(
                "Display taxonomy for:",
                options=cluster_options,
                index=0,
                help="Select a specific cluster to show its taxonomy tree, or 'All' to show the entire dataset"
            )
        
        with col2:
            min_count = st.number_input(
                "Minimum count",
                min_value=1,
                max_value=1000,
                value=5,
                step=1,
                help="Minimum number of records for a taxon to appear in the tree"
            )
        
        with col3:
            tree_depth = st.slider(
                "Tree depth",
                min_value=1,
                max_value=7,
                value=7,
                help="Maximum depth of the taxonomy tree to display"
            )
        
        # Filter data based on selected cluster
        if selected_cluster != "All" and selected_cluster.startswith("Cluster "):
            cluster_id = selected_cluster.replace("Cluster ", "")
            cluster_mask = df_plot['cluster'] == cluster_id
            cluster_uuids = df_plot[cluster_mask]['uuid'].tolist()
            tree_df = filtered_df[filtered_df['uuid'].isin(cluster_uuids)]
            display_title = f"Taxonomic Tree for {selected_cluster}"
        else:
            tree_df = filtered_df
            display_title = "Taxonomic Tree for All Clusters"
        
        # Build taxonomic tree for the selected data
        tree = build_taxonomic_tree(tree_df)
        stats = get_tree_statistics(tree)
        
        # Show statistics
        st.markdown(f"**{display_title}**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{stats['total_records']:,}")
        with col2:
            st.metric("Kingdoms", stats['kingdoms'])
        with col3:
            st.metric("Families", stats['families'])
        with col4:
            st.metric("Species", stats['species'])
        
        # Display the tree
        tree_string = format_tree_string(tree, max_depth=tree_depth, min_count=min_count)
        
        if tree_string:
            st.code(tree_string, language="text")
        else:
            st.info("No taxonomic data meets the display criteria. Try lowering the minimum count.")


def render_clustering_summary(show_taxonomy=False):
    """Render the clustering summary panel."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    embeddings = st.session_state.get("embeddings", None)

    if df_plot is not None and labels is not None and embeddings is not None:
        # Check if this is image data or metadata-only data
        has_images = 'image_path' in df_plot.columns
        
        if has_images:
            # For image data, show the full clustering summary
            st.subheader("Clustering Summary")
            
            try:
                summary_df, representatives = ClusteringService.generate_clustering_summary(
                    embeddings, labels, df_plot
                )
                
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

                st.markdown("#### Representative Images")
                for row in summary_df.itertuples():
                    k = row.Cluster
                    st.markdown(f"**Cluster {k}**")
                    img_cols = st.columns(3)
                    for i, img_idx in enumerate(representatives[k]):
                        img_path = df_plot.iloc[img_idx]["image_path"]
                        img_cols[i].image(
                            img_path, 
                            use_container_width=True, 
                            caption=os.path.basename(img_path)
                        )
                        
            except Exception as e:
                st.error(f"Error generating clustering summary: {e}")
        else:
            # For metadata-only data (precalculated embeddings), show taxonomic tree if requested
            if show_taxonomy:
                filtered_df = st.session_state.get("filtered_df_for_clustering", None)
                
                if filtered_df is not None:
                    render_taxonomic_tree_summary()
                    
    else:
        st.info("Clustering summary will appear here after clustering.")
