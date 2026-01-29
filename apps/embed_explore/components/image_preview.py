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
    selected_idx = st.session_state.get("selected_image_idx", 0)

    if (
        valid_paths is not None and
        labels is not None and
        selected_idx is not None and
        0 <= selected_idx < len(valid_paths)
    ):
        img_path = valid_paths[selected_idx]
        cluster = labels[selected_idx] if labels is not None else "?"

        # Log only when image changes
        if _last_displayed_path != img_path:
            logger.info(f"[Image] Loading local file: {os.path.basename(img_path)} (cluster={cluster})")
            _last_displayed_path = img_path

        st.image(img_path, caption=f"Cluster {cluster}: {os.path.basename(img_path)}", width='stretch')
        st.markdown(f"**File:** `{os.path.basename(img_path)}`")
        st.markdown(f"**Cluster:** `{cluster}`")
    else:
        st.info("Image preview will appear here after you select a cluster point.")
