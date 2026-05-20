"""
Image preview component for the embed_explore application.
"""

import streamlit as st
import os

from shared.utils.logging_config import get_logger

logger = get_logger(__name__)

# Track last displayed image to avoid duplicate logging
_last_displayed_path = None


def render_image_preview():
    """Render the image preview panel for local image files."""
    global _last_displayed_path

    valid_paths = st.session_state.get("valid_paths", None)
    labels = st.session_state.get("labels", None)
    kmeans_col = st.session_state.get("kmeans_column", None)
    selected_idx = st.session_state.get("selected_image_idx", None)

    if (
        valid_paths is not None and
        selected_idx is not None and
        0 <= selected_idx < len(valid_paths)
    ):
        img_path = valid_paths[selected_idx]
        cluster = labels[selected_idx] if labels is not None else None

        if _last_displayed_path != img_path:
            log_msg = f"[Image] Loading local file: {os.path.basename(img_path)}"
            if cluster is not None:
                log_msg += f" (cluster={cluster})"
            logger.info(log_msg)
            _last_displayed_path = img_path

        caption = os.path.basename(img_path)
        if cluster is not None and kmeans_col:
            caption = f"{kmeans_col}={cluster}: {caption}"

        st.image(img_path, caption=caption, width='stretch')
        st.markdown(f"**File:** `{os.path.basename(img_path)}`")
        if cluster is not None and kmeans_col:
            st.markdown(f"**{kmeans_col}:** `{cluster}`")
    else:
        st.info("Image preview will appear here after you select a point in the scatter.")
