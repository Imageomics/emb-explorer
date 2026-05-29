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
    user clicks a *different* point or changes the "Color by" column.
    """
    df_plot = st.session_state.get("data", None)

    if df_plot is not None and len(df_plot) > 1:
        _render_chart_fragment(df_plot)
    else:
        # Detect app type for appropriate message
        is_precalculated = st.session_state.get("page_type") == "precalculated_app"
        if is_precalculated:
            st.info("Run projection to see the scatter plot.")
        else:
            st.info("Run clustering to see the cluster scatter plot.")
        st.session_state['selected_image_idx'] = None


@st.fragment
def _render_chart_fragment(df_plot):
    """Fragment-isolated chart rendering — zoom/pan do NOT rerun the page."""
    # Track previous density mode to detect changes
    prev_density_mode = st.session_state.get("_prev_density_mode", None)

    # Detect app type: precalculated has uuid but no image_path
    is_precalculated = 'uuid' in df_plot.columns and 'image_path' not in df_plot.columns

    # Plot options
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

    # Determine color column — same dropdown pattern for both apps.
    # Build list of colorable columns (skip technical/identifier columns).
    skip_color_cols = {'x', 'y', 'idx', 'uuid', 'emb', 'embedding', 'embeddings', 'vector',
                       'identifier', 'image_url', 'url', 'img_url', 'image',
                       'image_path', 'file_name'}
    colorable_cols = [c for c in df_plot.columns
                      if c not in skip_color_cols and df_plot[c].nunique() <= 100]

    # Sort KMeans columns to front (all runs, sorted by k)
    kmeans_cols = sorted(
        [c for c in colorable_cols if c.startswith("KMeans (k=")],
        key=lambda c: int(c.split("=")[1].rstrip(")"))
    )
    other_cols = [c for c in colorable_cols if not c.startswith("KMeans (k=")]
    colorable_cols = kmeans_cols + other_cols

    # Build unique count lookup for display
    col_nunique = {c: df_plot[c].nunique() for c in colorable_cols}

    if colorable_cols:
        color_col = st.selectbox(
            "Color by",
            options=["(none)"] + colorable_cols,
            index=0,
            key="color_by_column",
            format_func=lambda c: c if c == "(none)" else f"{c} ({col_nunique[c]})",
            help="Select a column to color the points by"
        )
        if color_col == "(none)":
            color_col = None
    else:
        color_col = None

    # Warning for high cardinality
    if color_col and df_plot[color_col].nunique() > 20:
        st.warning(f"'{color_col}' has {df_plot[color_col].nunique()} unique values. Colors may repeat.")

    # Trigger full page rerun when color changes (so bottom section updates).
    # Use a sentinel to distinguish "never set" from "set to None".
    _sentinel = object()
    prev_color = st.session_state.get("_prev_color_by", _sentinel)
    if color_col != prev_color:
        st.session_state["_prev_color_by"] = color_col
        if prev_color is not _sentinel:
            st.rerun(scope="app")

    point_selector = alt.selection_point(fields=["idx"], name="point_selection")

    # Build tooltip fields
    tooltip_fields = []
    skip_cols = {'x', 'y', 'idx', 'emb', 'embedding', 'embeddings', 'vector',
                 'uuid', 'identifier', 'image_url', 'url', 'img_url', 'image'}

    # For embed_explore, include the file_name in the tooltip for quick reference
    if not is_precalculated and 'file_name' in df_plot.columns:
        tooltip_fields.append('file_name:N')
        skip_cols.add('file_name')
    skip_cols.add('image_path')

    # Add the color column first if set (and not already in tooltip)
    if color_col and color_col not in skip_cols:
        tooltip_fields.append(f'{color_col}:N')
        skip_cols.add(color_col)

    # Add remaining metadata columns
    metadata_cols = [c for c in df_plot.columns if c not in skip_cols][:15]
    tooltip_fields.extend(metadata_cols)

    # Title
    if is_precalculated:
        title = "Embedding Space (click a point to view details)"
    else:
        title = "Image Clusters (click a point to preview image)"

    # Set opacity based on density mode
    if density_mode == "Opacity":
        point_opacity = 0.15
    elif density_mode == "Heatmap":
        point_opacity = 0.5
    else:
        point_opacity = 0.7

    # Build chart
    if color_col:
        # Sort legend: numeric for KMeans labels, alphabetical for strings
        unique_vals = df_plot[color_col].unique()
        try:
            sorted_vals = sorted(unique_vals, key=int)
        except (ValueError, TypeError):
            sorted_vals = sorted(unique_vals, key=str)

        scatter = (
            alt.Chart(df_plot)
            .mark_circle(size=60, opacity=point_opacity)
            .encode(
                x=alt.X('x:Q', scale=alt.Scale(zero=False)),
                y=alt.Y('y:Q', scale=alt.Scale(zero=False)),
                color=alt.Color(
                    f'{color_col}:N',
                    legend=alt.Legend(title=color_col),
                    sort=sorted_vals,
                    scale=alt.Scale(scheme='tableau20')
                ),
                tooltip=tooltip_fields,
                fillOpacity=alt.condition(point_selector, alt.value(1), alt.value(0.3))
            )
            .add_params(point_selector)
        )
    else:
        # No color column: all points same color
        scatter = (
            alt.Chart(df_plot)
            .mark_circle(size=60, opacity=point_opacity)
            .encode(
                x=alt.X('x:Q', scale=alt.Scale(zero=False)),
                y=alt.Y('y:Q', scale=alt.Scale(zero=False)),
                tooltip=tooltip_fields,
                fillOpacity=alt.condition(point_selector, alt.value(1), alt.value(0.3))
            )
            .add_params(point_selector)
        )

    if density_mode == "Heatmap":
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
        .interactive()
    )

    logger.debug(f"[Visualization] Rendering chart: {len(df_plot)} points, density={density_mode}, "
                 f"color={color_col or 'none'}")

    # Include data_version in key so zoom/pan resets when projection changes
    data_version = st.session_state.get("data_version", "")
    chart_key = f"alt_chart_{data_version}"

    if density_mode == "Heatmap":
        st.altair_chart(chart, key=chart_key, width="stretch")
        st.caption("Note: Point selection is disabled when heatmap is shown.")
    else:
        event = st.altair_chart(chart, key=chart_key, on_select="rerun", width="stretch")

        if (
            event
            and "selection" in event
            and "point_selection" in event["selection"]
            and event["selection"]["point_selection"]
        ):
            new_idx = int(event["selection"]["point_selection"][0]["idx"])
            prev_idx = st.session_state.get("selected_image_idx")
            if prev_idx != new_idx:
                label = ''
                if color_col and color_col in df_plot.columns:
                    label = f", {color_col}={df_plot.iloc[new_idx][color_col]}"
                logger.info(f"[Visualization] Point selected: idx={new_idx}{label}")
                st.session_state["selected_image_idx"] = new_idx
                st.session_state["selection_data_version"] = st.session_state.get("data_version", None)
                st.rerun(scope="app")
