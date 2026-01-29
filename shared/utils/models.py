"""
Model listing utilities.

Uses lazy loading to avoid importing open_clip at startup.
"""

import pandas as pd

# Lazy-loaded module reference
_open_clip = None


def _get_open_clip():
    """Lazy load open_clip module."""
    global _open_clip
    if _open_clip is None:
        import open_clip
        _open_clip = open_clip
    return _open_clip


def list_available_models():
    """List all available models."""
    open_clip = _get_open_clip()

    # Create list of all models
    models_data = []

    # Add special models first
    models_data.extend([
        {"name": "hf-hub:imageomics/bioclip-2", "pretrained": None},
        {"name": "hf-hub:imageomics/bioclip", "pretrained": None}
    ])

    # OpenCLIP models
    openclip_models = open_clip.list_pretrained()
    for model_name, pretrained in openclip_models:
        models_data.append({
            "name": model_name,
            "pretrained": pretrained
        })

    return models_data
