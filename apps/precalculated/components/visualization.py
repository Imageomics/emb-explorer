"""
Visualization components for the precalculated embeddings application.
"""

import streamlit as st
import altair as alt


def render_scatter_plot():
    """Render the main clustering scatter plot with dynamic tooltips."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)

    if df_plot is not None and len(df_plot) > 1:
        # Plot options
        density_mode = st.radio(
            "Density visualization",
            options=["Off", "Opacity", "Heatmap"],
            index=0,
            horizontal=True,
            key="density_mode",
            help="Off: normal view | Opacity: lower opacity to show overlap | Heatmap: 2D binned density (disables selection)"
        )

        point_selector = alt.selection_point(fields=["idx"], name="point_selection")

        # Build tooltip fields dynamically from available columns
        tooltip_fields = []

        # Always include cluster info
        if 'cluster_name' in df_plot.columns:
            tooltip_fields.append('cluster_name:N')
            cluster_field = 'cluster_name:N'
        else:
            tooltip_fields.append('cluster:N')
            cluster_field = 'cluster:N'

        # Add other metadata columns (limit to prevent tooltip overflow)
        skip_cols = {'x', 'y', 'cluster', 'cluster_name', 'idx', 'uuid', 'emb'}
        metadata_cols = [c for c in df_plot.columns if c not in skip_cols][:8]
        tooltip_fields.extend(metadata_cols)

        # Set opacity based on density mode
        if density_mode == "Opacity":
            point_opacity = 0.15  # Low opacity so overlaps show density
        elif density_mode == "Heatmap":
            point_opacity = 0.5  # Medium opacity when heatmap is behind
        else:
            point_opacity = 0.7  # Normal opacity

        # Create scatter plot
        scatter = (
            alt.Chart(df_plot)
            .mark_circle(size=60, opacity=point_opacity)
            .encode(
                x=alt.X('x:Q', scale=alt.Scale(zero=False)),
                y=alt.Y('y:Q', scale=alt.Scale(zero=False)),
                color=alt.Color('cluster:N', legend=alt.Legend(title="Cluster")),
                tooltip=tooltip_fields,
                fillOpacity=alt.condition(point_selector, alt.value(1), alt.value(0.3))
            )
            .add_params(point_selector)
        )

        if density_mode == "Heatmap":
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
        if density_mode != "Heatmap":
            title_suffix += ", click to select"

        chart = (
            chart
            .properties(
                width=800,
                height=700,
                title="Embedding Clusters" + title_suffix
            )
            .interactive()
        )

        # Streamlit doesn't support selections on layered charts, so only enable
        # selection when not using heatmap mode
        if density_mode == "Heatmap":
            st.altair_chart(chart, key="alt_chart", width="stretch")
            st.caption("Note: Point selection is disabled when heatmap is shown.")
        else:
            event = st.altair_chart(chart, key="alt_chart", on_select="rerun", width="stretch")

            # Handle selection - track data version to ensure selection is tied to current data
            if (
                event
                and "selection" in event
                and "point_selection" in event["selection"]
                and event["selection"]["point_selection"]
            ):
                new_idx = int(event["selection"]["point_selection"][0]["idx"])
                st.session_state["selected_image_idx"] = new_idx
                # Store the data version when this selection was made
                st.session_state["selection_data_version"] = st.session_state.get("data_version", None)

    else:
        st.info("Run clustering to see the visualization.")
        st.session_state['selected_image_idx'] = None
