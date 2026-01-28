"""
Shared visualization components for scatter plots.
"""

import streamlit as st
import altair as alt
import os
from typing import Optional


def render_scatter_plot():
    """Render the main clustering scatter plot with dynamic tooltips."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    selected_idx = st.session_state.get("selected_image_idx", 0)

    if df_plot is not None and len(df_plot) > 1:
        # Plot options
        show_density = st.checkbox(
            "Show density heatmap",
            value=st.session_state.get("show_density", False),
            key="density_toggle",
            help="Overlay density heatmap to visualize point concentration"
        )
        st.session_state["show_density"] = show_density

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

        # Create scatter plot
        scatter = (
            alt.Chart(df_plot)
            .mark_circle(size=60, opacity=0.5 if show_density else 0.7)
            .encode(
                x=alt.X('x:Q', scale=alt.Scale(zero=False)),
                y=alt.Y('y:Q', scale=alt.Scale(zero=False)),
                color=alt.Color('cluster:N', legend=alt.Legend(title=cluster_legend_title)),
                tooltip=tooltip_fields,
                fillOpacity=alt.condition(point_selector, alt.value(1), alt.value(0.3))
            )
            .add_params(point_selector)
        )

        if show_density:
            # Create 2D density heatmap layer
            density = (
                alt.Chart(df_plot)
                .mark_rect(opacity=0.4)
                .encode(
                    x=alt.X('x:Q', bin=alt.Bin(maxbins=40), scale=alt.Scale(zero=False)),
                    y=alt.Y('y:Q', bin=alt.Bin(maxbins=40), scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        'count():Q',
                        scale=alt.Scale(scheme='blues'),
                        legend=None
                    )
                )
            )
            # Layer density behind scatter
            chart = alt.layer(density, scatter)
        else:
            chart = scatter

        # Apply common properties and interactivity
        title_suffix = " (scroll to zoom, drag to pan)"
        if not show_density:
            title_suffix += ", click to select"

        chart = (
            chart
            .properties(
                width=800,
                height=700,
                title=title + title_suffix
            )
            .interactive()  # Enable zoom/pan
        )

        # Streamlit doesn't support selections on layered charts, so only enable
        # selection when density is off
        if show_density:
            st.altair_chart(chart, key="alt_chart", width="stretch")
            st.caption("Note: Point selection is disabled when density heatmap is shown.")
        else:
            event = st.altair_chart(chart, key="alt_chart", on_select="rerun", width="stretch")

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
