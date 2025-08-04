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
                # Check if we have taxonomic clustering with cluster names
                taxonomic_info = st.session_state.get("taxonomic_clustering", {})
                is_taxonomic = taxonomic_info.get('is_taxonomic', False)
                
                if is_taxonomic and 'cluster_name' in df_plot.columns:
                    # Use taxonomic names for display
                    unique_cluster_names = sorted(df_plot["cluster_name"].unique())
                    cluster_options.extend(unique_cluster_names)
                else:
                    # Standard numeric clustering
                    unique_clusters = sorted(df_plot["cluster"].unique(), key=lambda x: int(x))
                    cluster_options.extend([f"Cluster {c}" for c in unique_clusters])
            
            selected_cluster = st.selectbox(
                "Display taxonomy for:",
                options=cluster_options,
                index=0,
                key="taxonomy_cluster_selector",
                help="Select a specific cluster to show its taxonomy tree, or 'All' to show the entire dataset"
            )
        
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
        
        # Create a stable cache key based on the data characteristics and filter parameters
        # Use data length and a sample of UUIDs for a stable data identifier
        data_length = len(filtered_df)
        # Use a stable string representation instead of hash for consistency
        sample_uuids = filtered_df['uuid'].iloc[:min(10, len(filtered_df))].tolist()
        data_id = f"{data_length}_{len(sample_uuids)}_{sample_uuids[0] if sample_uuids else 'empty'}"
        cache_key = f"taxonomy_{data_id}_{selected_cluster}_{min_count}_{tree_depth}"
        
        # Check if we have cached results and they're still valid
        # Also ensure critical session state data hasn't changed unexpectedly
        current_cache_key = st.session_state.get("taxonomy_cache_key")
        cache_exists = cache_key in st.session_state
        
        if (not cache_exists or current_cache_key != cache_key):
            
            # Data or parameters changed, regenerate taxonomy tree
            with st.spinner("Building taxonomy tree..."):
                # Filter data based on selected cluster
                if selected_cluster != "All":
                    taxonomic_info = st.session_state.get("taxonomic_clustering", {})
                    is_taxonomic = taxonomic_info.get('is_taxonomic', False)
                    
                    if is_taxonomic and 'cluster_name' in df_plot.columns:
                        # For taxonomic clustering, filter by cluster_name
                        cluster_mask = df_plot['cluster_name'] == selected_cluster
                        cluster_uuids = df_plot[cluster_mask]['uuid'].tolist()
                        tree_df = filtered_df[filtered_df['uuid'].isin(cluster_uuids)]
                        display_title = f"Taxonomic Tree for {selected_cluster}"
                    elif selected_cluster.startswith("Cluster "):
                        # For numeric clustering, extract cluster ID
                        cluster_id = selected_cluster.replace("Cluster ", "")
                        cluster_mask = df_plot['cluster'] == cluster_id
                        cluster_uuids = df_plot[cluster_mask]['uuid'].tolist()
                        tree_df = filtered_df[filtered_df['uuid'].isin(cluster_uuids)]
                        display_title = f"Taxonomic Tree for {selected_cluster}"
                    else:
                        # Fallback: treat as direct cluster name
                        cluster_mask = df_plot['cluster_name'] == selected_cluster if 'cluster_name' in df_plot.columns else df_plot['cluster'] == selected_cluster
                        cluster_uuids = df_plot[cluster_mask]['uuid'].tolist()
                        tree_df = filtered_df[filtered_df['uuid'].isin(cluster_uuids)]
                        display_title = f"Taxonomic Tree for {selected_cluster}"
                else:
                    tree_df = filtered_df
                    display_title = "Taxonomic Tree for All Clusters"
                
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
