"""
Visualization components for the embed_explore application.

This module re-exports from shared for backwards compatibility.
"""

# Re-export scatter plot from shared module
from shared.components.visualization import render_scatter_plot

# Re-export image preview from local module
from apps.embed_explore.components.image_preview import render_image_preview

__all__ = ['render_scatter_plot', 'render_image_preview']
