"""
Data preview components for the precalculated embeddings application.
Dynamically displays all available metadata fields.
"""

import streamlit as st
import pandas as pd
import numpy as np

from shared.utils.logging_config import get_logger
from shared.utils.representatives import find_cluster_representatives
from shared.utils.images import (
    IMAGE_URL_COLUMNS,
    fetch_images_concurrent,
    get_image_from_url,
    resolve_record_image_url,
    _IMAGE_CACHE,
)
from shared.components.representatives import render_representative_images

logger = get_logger(__name__)


def render_data_preview():
    """Render the data preview panel (record details on point click)."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    selected_idx = st.session_state.get("selected_image_idx", None)
    filtered_df = st.session_state.get("filtered_df_for_clustering", None)

    # Validate that selection matches current data version
    current_data_version = st.session_state.get("data_version", None)
    selection_data_version = st.session_state.get("selection_data_version", None)
    selection_valid = (
        selected_idx is not None and
        current_data_version is not None and
        selection_data_version == current_data_version
    )

    if (
        df_plot is not None and
        selection_valid and
        0 <= selected_idx < len(df_plot) and
        filtered_df is not None
    ):
        # Get the selected record
        selected_uuid = df_plot.iloc[selected_idx]['uuid']

        # Find the full record
        record = filtered_df[filtered_df['uuid'] == selected_uuid].iloc[0]

        st.markdown("### Record Details")

        # Try to display image if an image URL column exists (process-cached).
        url = resolve_record_image_url(record)
        if url:
            image = get_image_from_url(url)
            if image is not None:
                st.image(image, width=280)

        st.markdown(f"**UUID:** `{selected_uuid}`")

        # Build metadata table for remaining fields
        skip_fields = {'emb', 'embedding', 'embeddings', 'vector', 'idx', 'uuid'}

        metadata_rows = []
        for field, value in record.items():
            if field.lower() in skip_fields or field in skip_fields:
                continue
            if pd.isna(value):
                continue

            if isinstance(value, float):
                display_val = f"{value:.4f}"
            elif isinstance(value, (list, tuple)):
                display_val = f"[{len(value)} items]"
            else:
                display_val = str(value)

            metadata_rows.append({"Field": field, "Value": display_val})

        if metadata_rows:
            st.markdown("---")
            st.markdown("**Metadata**")
            metadata_df = pd.DataFrame(metadata_rows)
            st.dataframe(
                metadata_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Field": st.column_config.TextColumn("Field", width="small"),
                    "Value": st.column_config.TextColumn("Value", width="large"),
                }
            )

    else:
        # Show appropriate message based on state
        if df_plot is not None:
            st.info("Click a point in the scatter plot to view its details.")
        else:
            st.info("Run projection first, then click a point to view details.")

        # Show dataset summary
        filtered_df_summary = st.session_state.get("filtered_df", None)
        if filtered_df_summary is not None and len(filtered_df_summary) > 0:
            st.markdown("### Dataset Summary")
            st.markdown(f"**Records:** {len(filtered_df_summary):,}")

            column_info = st.session_state.get("column_info", {})
            if column_info:
                with st.expander("Column overview"):
                    for col, info in list(column_info.items())[:10]:
                        unique = len(info['unique_values']) if info['unique_values'] else "many"
                        st.caption(f"**{col}** ({info['type']}): {unique} unique")


def _compute_entropy(counts):
    """Shannon entropy in bits."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def _build_cluster_tree(df_plot, kmeans_col, compare_col):
    """Build a tree-style string summarizing cluster composition against a comparison column."""
    unique_clusters = sorted(df_plot[kmeans_col].unique(), key=lambda x: int(x))
    n_total = len(df_plot)
    n_clusters = len(unique_clusters)

    lines = []
    lines.append(f'KMeans Clustering Summary ({n_total} points, {n_clusters} clusters)')
    lines.append(f'Compared against: {compare_col}')
    lines.append('')

    for ci, cluster_id in enumerate(unique_clusters):
        is_last_cluster = (ci == n_clusters - 1)
        mask = df_plot[kmeans_col] == cluster_id
        cluster_df = df_plot[mask]
        n = len(cluster_df)

        gt_counts = cluster_df[compare_col].value_counts()
        purity = gt_counts.iloc[0] / n if n > 0 else 0
        entropy = _compute_entropy(gt_counts.values)

        prefix = '\u2514\u2500\u2500 ' if is_last_cluster else '\u251c\u2500\u2500 '
        lines.append(f'{prefix}Cluster {cluster_id}  [{n} pts]  purity: {purity:.0%}  entropy: {entropy:.2f}')

        child_prefix = '    ' if is_last_cluster else '\u2502   '
        for ji, (cat, count) in enumerate(gt_counts.items()):
            is_last_cat = (ji == len(gt_counts) - 1)
            pct = count / n * 100
            cat_connector = '\u2514\u2500 ' if is_last_cat else '\u251c\u2500 '
            lines.append(f'{child_prefix}{cat_connector}{str(cat):<20} {count:>4d}  {pct:>5.1f}%')

    return '\n'.join(lines)


def render_cluster_analysis():
    """Render cluster analysis section (full-width bottom area).

    Shows ARI/NMI and tree breakdown when KMeans labels exist and a
    metadata column is selected in the Color by dropdown.
    """
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    kmeans_col = st.session_state.get("kmeans_column", None)
    color_by = st.session_state.get("color_by_column", None)

    if df_plot is None or labels is None or kmeans_col is None:
        return
    if kmeans_col not in df_plot.columns:
        return

    # Only show analysis when comparing KMeans against a different metadata column
    if not color_by or color_by == "(none)" or color_by == kmeans_col:
        return
    if color_by not in df_plot.columns:
        return

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    st.markdown(f"### Cluster Analysis: {kmeans_col} vs {color_by}")

    # Compute ARI/NMI (exclude "N/A" rows from metric computation)
    kmeans_labels = df_plot[kmeans_col].values
    metadata_labels = df_plot[color_by].values
    valid_mask = metadata_labels != "N/A"
    n_valid = valid_mask.sum()
    n_excluded = len(metadata_labels) - n_valid

    if n_valid > 0:
        ari = adjusted_rand_score(metadata_labels[valid_mask], kmeans_labels[valid_mask])
        nmi = normalized_mutual_info_score(
            metadata_labels[valid_mask], kmeans_labels[valid_mask], average_method='arithmetic'
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ARI", f"{ari:.3f}",
                      help="Adjusted Rand Index: 1 = perfect, 0 = random, <0 = worse than random")
        with col2:
            st.metric("NMI", f"{nmi:.3f}",
                      help="Normalized Mutual Information: 1 = perfect, 0 = no correlation")
        with col3:
            st.metric("Evaluated", f"{n_valid:,}",
                      help=f"Rows with non-null '{color_by}'")
        if n_excluded > 0:
            st.caption(f"{n_excluded:,} rows with N/A '{color_by}' excluded from evaluation")

        # Tree-style breakdown
        tree_output = _build_cluster_tree(df_plot, kmeans_col, color_by)
        st.code(tree_output, language="text")
    else:
        st.info(f"No valid '{color_by}' values to compare with KMeans clusters.")


def render_cluster_representatives():
    """Render representative images per KMeans cluster for the precalculated app.

    Representatives are the members closest to each cluster centroid (computed
    on the full-dimensional embeddings). Images are fetched from each record's
    URL column; URLs that fail to load are skipped and the next-closest
    candidate is tried (fallback), so transient/broken URLs don't leave gaps.
    """
    df_plot = st.session_state.get("data", None)
    embeddings = st.session_state.get("embeddings", None)
    if df_plot is None or embeddings is None:
        return

    kmeans_cols = sorted(
        [c for c in df_plot.columns if c.startswith("KMeans (k=")],
        key=lambda c: int(c.split("=")[1].rstrip(")")),
    )
    if not kmeans_cols:
        return  # nothing to show until a KMeans run exists

    st.markdown("### Representative Images")
    st.caption(
        "Members closest to each cluster centroid. Images load from each "
        "record's URL; unreachable images are skipped automatically."
    )

    selected_col = st.selectbox(
        "KMeans result",
        options=kmeans_cols,
        index=len(kmeans_cols) - 1,
        key="representatives_kmeans_selector",
        help="Which KMeans run to show representatives for.",
    )

    # Guard: embeddings must align row-for-row with df_plot.
    if len(embeddings) != len(df_plot):
        st.info("Re-run projection and KMeans to view representatives.")
        return

    n_per_cluster = 3
    representatives = find_cluster_representatives(
        embeddings, df_plot[selected_col].values, n_per_cluster=n_per_cluster
    )

    # Warm the cache concurrently. Representatives are oversampled for fallback,
    # but we only need a few successes per cluster — prefetch a prefix (2x the
    # display count) in parallel. Deeper fallback candidates (rare) resolve
    # on-demand below.
    prefetch_per_cluster = n_per_cluster * 2
    prefetch_urls = [
        resolve_record_image_url(df_plot.iloc[idx])
        for idxs in representatives.values()
        for idx in idxs[:prefetch_per_cluster]
    ]
    with st.spinner("Loading representative images..."):
        fetch_images_concurrent([u for u in prefetch_urls if u])

    def _resolve(idx):
        url = resolve_record_image_url(df_plot.iloc[idx])
        if not url:
            return None
        # Prefetched URLs hit the process cache; anything deeper falls back to
        # a single synchronous fetch (also cached).
        if url in _IMAGE_CACHE:
            return _IMAGE_CACHE[url]
        return get_image_from_url(url)

    def _caption(idx):
        row = df_plot.iloc[idx]
        for col in ("scientific_name", "species", "common_name", "uuid"):
            if col in row.index and pd.notna(row[col]):
                return str(row[col])
        return None

    render_representative_images(
        representatives,
        resolve_image=_resolve,
        n_per_cluster=n_per_cluster,
        caption_fn=_caption,
    )
