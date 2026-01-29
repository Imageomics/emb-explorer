"""
Data preview components for the precalculated embeddings application.
Dynamically displays all available metadata fields.
"""

import streamlit as st
import pandas as pd
import requests
from typing import Optional
from PIL import Image
from io import BytesIO


@st.cache_data(ttl=300, show_spinner=False)
def fetch_image_from_url(url: str, timeout: int = 5) -> Optional[bytes]:
    """Try to fetch an image from a URL. Returns bytes to be cacheable."""
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


def get_image_from_url(url: str) -> Optional[Image.Image]:
    """Get image from URL with caching."""
    image_bytes = fetch_image_from_url(url)
    if image_bytes:
        return Image.open(BytesIO(image_bytes))
    return None


def render_data_preview():
    """Render the data preview panel with dynamic field display."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    selected_idx = st.session_state.get("selected_image_idx", None)  # Default to None, not 0
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

        st.markdown("### 📋 Record Details")

        # Try to display image if identifier/url column exists (cached to prevent re-fetch)
        image_cols = ['identifier', 'image_url', 'url', 'img_url', 'image']
        for img_col in image_cols:
            if img_col in record.index and pd.notna(record[img_col]):
                url = record[img_col]
                image = get_image_from_url(url)
                if image is not None:
                    st.image(image, width=280)
                    break

        # Build metadata table
        skip_fields = {'emb', 'embedding', 'embeddings', 'vector', 'idx'}

        # Collect all metadata as rows
        metadata_rows = []

        # Always show cluster and UUID first
        metadata_rows.append({"Field": "Cluster", "Value": str(cluster_display)})
        metadata_rows.append({"Field": "UUID", "Value": str(selected_uuid)})

        # Add remaining fields
        for field, value in record.items():
            if field.lower() in skip_fields or field in ['uuid', 'cluster', 'cluster_name']:
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

        # Display as table with full values
        if metadata_rows:
            metadata_df = pd.DataFrame(metadata_rows)
            st.dataframe(
                metadata_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Field": st.column_config.TextColumn("Field", width="small"),
                    "Value": st.column_config.TextColumn("Value", width="large"),
                }
            )

    else:
        # Show appropriate message based on state
        if df_plot is not None and labels is not None:
            st.info("📋 Click a point in the scatter plot to view its details.")
        else:
            st.info("📋 Run clustering first, then click a point to view details.")

        # Show dataset summary
        filtered_df = st.session_state.get("filtered_df", None)
        if filtered_df is not None and len(filtered_df) > 0:
            st.markdown("### 📈 Dataset Summary")
            st.markdown(f"**Records:** {len(filtered_df):,}")

            # Show column stats
            column_info = st.session_state.get("column_info", {})
            if column_info:
                with st.expander("Column overview"):
                    for col, info in list(column_info.items())[:10]:
                        unique = len(info['unique_values']) if info['unique_values'] else "many"
                        st.caption(f"• **{col}** ({info['type']}): {unique} unique")
