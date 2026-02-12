"""
Shared visualization components for scatter plots.
"""

import streamlit as st
import altair as alt

from shared.utils.logging_config import get_logger

logger = get_logger(__name__)


def render_scatter_plot():
    """Render the main clustering scatter plot with dynamic tooltips.

    The chart is rendered inside a @st.fragment so that zoom/pan interactions
    only rerun the chart itself — the rest of the page (data preview, summary)
    stays untouched.  A full page rerun is triggered explicitly only when the
    user clicks a *different* point.
    """
    df_plot = st.session_state.get("data", None)

    if df_plot is not None and len(df_plot) > 1:
        _render_chart_fragment(df_plot)
    else:
        st.info("Run clustering to see the cluster scatter plot.")
        st.session_state['selected_image_idx'] = None


@st.fragment
def _render_chart_fragment(df_plot):
    """Fragment-isolated chart rendering — zoom/pan do NOT rerun the page."""
    # Track previous density mode to detect changes
    prev_density_mode = st.session_state.get("_prev_density_mode", None)

    # Plot options in columns for compact layout
    opt_col1, opt_col2 = st.columns([2, 1])

    with opt_col1:
        density_mode = st.radio(
            "Density visualization",
            options=["Off", "Opacity", "Heatmap"],
            index=0,
            horizontal=True,
            key="density_mode",
            help="Off: normal view | Opacity: lower opacity to show overlap | Heatmap: 2D binned density (disables selection)"
        )

    # Log density mode change
    if prev_density_mode != density_mode:
        logger.info(f"[Visualization] Density mode changed: {prev_density_mode} -> {density_mode}")
        st.session_state["_prev_density_mode"] = density_mode

    with opt_col2:
        if density_mode == "Heatmap":
            prev_bins = st.session_state.get("_prev_heatmap_bins", 40)
            heatmap_bins = st.slider(
                "Grid resolution",
                min_value=10,
                max_value=80,
                value=40,
                step=5,
                key="heatmap_bins",
                help="Number of bins for density grid (higher = finer detail)"
            )
            if prev_bins != heatmap_bins:
                logger.info(f"[Visualization] Heatmap bins changed: {prev_bins} -> {heatmap_bins}")
                st.session_state["_prev_heatmap_bins"] = heatmap_bins
        else:
            heatmap_bins = 40  # Default, not used

    point_selector = alt.selection_point(fields=["idx"], name="point_selection")

    # Determine tooltip fields based on available columns
    tooltip_fields = []

    # Use cluster_name for display if available (taxonomic clustering), otherwise use cluster
    if 'cluster_name' in df_plot.columns:
        tooltip_fields.append('cluster_name:N')
        cluster_legend_title = "Cluster"
    else:
        tooltip_fields.append('cluster:N')
        cluster_legend_title = "Cluster"

    # Add other metadata columns dynamically (limit to prevent tooltip overflow)
    skip_cols = {'x', 'y', 'cluster', 'cluster_name', 'idx', 'emb', 'embedding', 'embeddings', 'vector'}
    metadata_cols = [c for c in df_plot.columns if c not in skip_cols][:8]
    tooltip_fields.extend(metadata_cols)

    # Determine title based on data type
    if 'uuid' in df_plot.columns:
        title = "Embedding Clusters (click a point to view details)"
    else:
        title = "Image Clusters (click a point to preview image)"

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
            color=alt.Color('cluster:N', legend=alt.Legend(title=cluster_legend_title)),
            tooltip=tooltip_fields,
            fillOpacity=alt.condition(point_selector, alt.value(1), alt.value(0.3))
        )
        .add_params(point_selector)
    )

    if density_mode == "Heatmap":
        # Create 2D density heatmap layer with configurable bins
        density = (
            alt.Chart(df_plot)
            .mark_rect(opacity=0.4)
            .encode(
                x=alt.X('x:Q', bin=alt.Bin(maxbins=heatmap_bins), scale=alt.Scale(zero=False)),
                y=alt.Y('y:Q', bin=alt.Bin(maxbins=heatmap_bins), scale=alt.Scale(zero=False)),
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
            title=title + title_suffix
        )
        .interactive()  # Enable zoom/pan
    )

    # Log chart render only at DEBUG to avoid noise from zoom/pan reruns
    logger.debug(f"[Visualization] Rendering chart: {len(df_plot)} points, density={density_mode}, "
                 f"bins={heatmap_bins if density_mode == 'Heatmap' else 'N/A'}")

    # Streamlit doesn't support selections on layered charts, so only enable
    # selection when not using heatmap mode
    if density_mode == "Heatmap":
        st.altair_chart(chart, key="alt_chart", width="stretch")
        st.caption("Note: Point selection is disabled when heatmap is shown.")
    else:
        event = st.altair_chart(chart, key="alt_chart", on_select="rerun", width="stretch")

        # Handle point selection — only trigger full page rerun when
        # the selected point actually changes (zoom/pan stay fragment-local)
        if (
            event
            and "selection" in event
            and "point_selection" in event["selection"]
            and event["selection"]["point_selection"]
        ):
            new_idx = int(event["selection"]["point_selection"][0]["idx"])
            prev_idx = st.session_state.get("selected_image_idx")
            if prev_idx != new_idx:
                cluster = df_plot.iloc[new_idx]['cluster'] if 'cluster' in df_plot.columns else '?'
                logger.info(f"[Visualization] Point selected: idx={new_idx}, cluster={cluster}")
                st.session_state["selected_image_idx"] = new_idx
                st.session_state["selection_data_version"] = st.session_state.get("data_version", None)
                # Trigger full page rerun so the preview panel updates
                st.rerun(scope="app")
