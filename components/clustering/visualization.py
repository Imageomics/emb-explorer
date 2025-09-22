"""
Visualization components for the clustering page.
"""

import streamlit as st
import altair as alt
import os
from typing import Optional


def render_scatter_plot():
    """Render the main clustering scatter plot."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    selected_idx = st.session_state.get("selected_image_idx", 0)

    if df_plot is not None and len(df_plot) > 1:
        point_selector = alt.selection_point(fields=["idx"], name="point_selection")
        
        # Determine tooltip fields based on available columns
        tooltip_fields = []
        
        # Use cluster_name for display if available (taxonomic clustering), otherwise use cluster
        if 'cluster_name' in df_plot.columns:
            tooltip_fields.append('cluster_name:N')
            cluster_legend_field = 'cluster_name:N'
            cluster_legend_title = "Cluster"
        else:
            tooltip_fields.append('cluster:N')
            cluster_legend_field = 'cluster:N'
            cluster_legend_title = "Cluster"
        
        # Add metadata fields if available (for precalculated embeddings)
        metadata_fields = ['scientific_name', 'common_name', 'family', 'genus', 'species', 'uuid']
        for field in metadata_fields:
            if field in df_plot.columns:
                tooltip_fields.append(field)
        
        # Add file_name if available (for image clustering)
        if 'file_name' in df_plot.columns:
            tooltip_fields.append('file_name')
        
        # Determine title based on data type
        if 'uuid' in df_plot.columns:
            title = "Embedding Clusters (click a point to view details)"
        else:
            title = "Image Clusters (click a point to preview image)"
        
        scatter = (
            alt.Chart(df_plot)
            .mark_circle(size=60)
            .encode(
                x=alt.X('x', scale=alt.Scale(zero=False)),
                y=alt.Y('y', scale=alt.Scale(zero=False)),
                color=alt.Color('cluster:N', legend=alt.Legend(title=cluster_legend_title)),
                tooltip=tooltip_fields,
                fillOpacity=alt.condition(point_selector, alt.value(1), alt.value(0.3))
            )
            .add_params(point_selector)
            .properties(
                width=800,
                height=700,
                title=title
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

    else:
        st.info("Run clustering to see the cluster scatter plot.")
        st.session_state['selected_image_idx'] = None


def render_image_preview():
    """Render the image preview panel."""
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
        st.image(img_path, caption=f"Cluster {cluster}: {os.path.basename(img_path)}", width='stretch')
        st.markdown(f"**File:** `{os.path.basename(img_path)}`")
        st.markdown(f"**Cluster:** `{cluster}`")
    else:
        st.info("Image preview will appear here after you select a cluster point.")
