"""
Data preview components for the precalculated embeddings application.
Dynamically displays all available metadata fields.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from typing import Optional
from PIL import Image
from io import BytesIO

from shared.utils.logging_config import get_logger

logger = get_logger(__name__)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_image_from_url_cached(url: str, timeout: int = 5) -> Optional[bytes]:
    """Internal cached function to fetch image bytes."""
    if not url or not isinstance(url, str):
        return None

    try:
        if not url.startswith(('http://', 'https://')):
            return None

        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            return None

        return response.content

    except Exception:
        return None


def fetch_image_from_url(url: str, timeout: int = 5) -> Optional[bytes]:
    """
    Fetch an image from a URL with logging.
    Uses caching internally but logs the request.
    """
    if not url or not isinstance(url, str):
        return None

    if not url.startswith(('http://', 'https://')):
        logger.warning(f"[Image] Invalid URL scheme: {url[:50]}...")
        return None

    logger.info(f"[Image] Fetching: {url[:80]}...")
    start_time = time.time()

    result = _fetch_image_from_url_cached(url, timeout)

    elapsed = time.time() - start_time
    if result:
        logger.info(f"[Image] Loaded: {len(result)/1024:.1f}KB in {elapsed:.3f}s")
    else:
        logger.warning(f"[Image] Failed to load: {url[:50]}...")

    return result


def get_image_from_url(url: str) -> Optional[Image.Image]:
    """Get image from URL with caching and logging."""
    image_bytes = fetch_image_from_url(url)
    if image_bytes:
        try:
            image = Image.open(BytesIO(image_bytes))
            logger.info(f"[Image] Opened: {image.size[0]}x{image.size[1]} {image.mode}")
            return image
        except Exception as e:
            logger.error(f"[Image] Failed to open: {e}")
            return None
    return None


def render_data_preview():
    """Render the data preview panel (record details on point click)."""
    _render_record_details()


def render_cluster_analysis():
    """Render cluster analysis section (call from full-width bottom area)."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)

    taxonomic_info = st.session_state.get("taxonomic_clustering", {})
    evaluation_column = taxonomic_info.get("evaluation_column")

    if (
        evaluation_column
        and df_plot is not None
        and evaluation_column in df_plot.columns
        and labels is not None
    ):
        _render_cluster_analysis(df_plot, labels, evaluation_column)


def _render_record_details():
    """Render the record details panel (existing functionality)."""
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
        labels is not None and
        selection_valid and
        0 <= selected_idx < len(df_plot) and
        filtered_df is not None
    ):
        # Get the selected record
        selected_uuid = df_plot.iloc[selected_idx]['uuid']
        cluster = labels[selected_idx] if labels is not None else "?"

        # Use cluster_name if available
        if 'cluster_name' in df_plot.columns:
            cluster_display = df_plot.iloc[selected_idx]['cluster_name']
        else:
            cluster_display = cluster

        # Find the full record
        record = filtered_df[filtered_df['uuid'] == selected_uuid].iloc[0]

        st.markdown("### Record Details")

        # Try to display image if identifier/url column exists (cached to prevent re-fetch)
        image_cols = ['identifier', 'image_url', 'url', 'img_url', 'image']
        for img_col in image_cols:
            if img_col in record.index and pd.notna(record[img_col]):
                url = record[img_col]
                image = get_image_from_url(url)
                if image is not None:
                    st.image(image, width=280)
                    break

        # Display Cluster and UUID prominently (not in table)
        st.markdown(f"**Cluster:** `{cluster_display}`")
        st.markdown(f"**UUID:** `{selected_uuid}`")

        # Build metadata table for remaining fields
        skip_fields = {'emb', 'embedding', 'embeddings', 'vector', 'idx', 'uuid', 'cluster', 'cluster_name'}

        metadata_rows = []
        for field, value in record.items():
            if field.lower() in skip_fields or field in skip_fields:
                continue
            if pd.isna(value):
                continue

            # Format value
            if isinstance(value, float):
                display_val = f"{value:.4f}"
            elif isinstance(value, (list, tuple)):
                display_val = f"[{len(value)} items]"
            else:
                display_val = str(value)

            metadata_rows.append({"Field": field, "Value": display_val})

        # Display remaining metadata as table
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
        if df_plot is not None and labels is not None:
            st.info("Click a point in the scatter plot to view its details.")
        else:
            st.info("Run clustering first, then click a point to view details.")

        # Show dataset summary
        filtered_df = st.session_state.get("filtered_df", None)
        if filtered_df is not None and len(filtered_df) > 0:
            st.markdown("### Dataset Summary")
            st.markdown(f"**Records:** {len(filtered_df):,}")

            # Show column stats
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


def _build_cluster_tree(df_plot, evaluation_column):
    """Build a tree-style string summarizing cluster composition against ground truth."""
    unique_clusters = sorted(df_plot['cluster'].unique(), key=lambda x: int(x))
    n_total = len(df_plot)
    n_clusters = len(unique_clusters)

    lines = []
    lines.append(f'KMeans Clustering Summary ({n_total} points, {n_clusters} clusters)')
    lines.append(f'Evaluation column: {evaluation_column}')
    lines.append('')

    for ci, cluster_id in enumerate(unique_clusters):
        is_last_cluster = (ci == n_clusters - 1)
        mask = df_plot['cluster'] == cluster_id
        cluster_df = df_plot[mask]
        n = len(cluster_df)

        gt_counts = cluster_df[evaluation_column].value_counts()
        purity = gt_counts.iloc[0] / n if n > 0 else 0
        dominant = gt_counts.index[0]
        entropy = _compute_entropy(gt_counts.values)

        prefix = '\u2514\u2500\u2500 ' if is_last_cluster else '\u251c\u2500\u2500 '
        lines.append(f'{prefix}Cluster {cluster_id}  [{n} pts]  purity: {purity:.0%}  entropy: {entropy:.2f}')

        child_prefix = '    ' if is_last_cluster else '\u2502   '
        for ji, (cat, count) in enumerate(gt_counts.items()):
            is_last_cat = (ji == len(gt_counts) - 1)
            pct = count / n * 100
            cat_connector = '\u2514\u2500 ' if is_last_cat else '\u251c\u2500 '
            lines.append(f'{child_prefix}{cat_connector}{cat:<20s} {count:>4d}  {pct:>5.1f}%')

    return '\n'.join(lines)


def _render_cluster_analysis(df_plot, labels, evaluation_column):
    """Render cluster analysis with per-cluster breakdown against ground truth."""
    eval_metrics = st.session_state.get('evaluation_metrics')

    # Overall metrics
    if eval_metrics:
        st.markdown(f"### Evaluation vs `{evaluation_column}`")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ARI", f"{eval_metrics['ari']:.3f}",
                      help="Adjusted Rand Index: 1 = perfect, 0 = random, <0 = worse than random")
        with col2:
            st.metric("NMI", f"{eval_metrics['nmi']:.3f}",
                      help="Normalized Mutual Information: 1 = perfect, 0 = no correlation")
        with col3:
            st.metric("Evaluated", f"{eval_metrics['n_evaluated']:,}",
                      help="Rows with non-null ground truth")
        if eval_metrics.get('n_null_excluded', 0) > 0:
            st.caption(f"{eval_metrics['n_null_excluded']:,} null rows excluded")

    st.markdown("---")

    # Tree-style cluster summary
    tree_output = _build_cluster_tree(df_plot, evaluation_column)
    st.code(tree_output, language="text")
