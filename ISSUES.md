# Known Issues

## Issue #1: Clustering summary recomputes on every render

**Status:** Fixed
**Branch:** `feature/app-separation`
**Date:** 2026-01-29

### Problem

The clustering summary statistics are being recomputed on every Streamlit render cycle (point selection, density mode change, etc.) instead of only when the "Run Clustering" button is clicked.

### Evidence from logs

```
[2026-01-29 11:10:53] INFO [shared.components.visualization] [Visualization] Rendering chart: 1000 points, density=Opacity, bins=N/A
[2026-01-29 11:10:53] INFO [shared.services.clustering_service] Generating clustering summary statistics
[2026-01-29 11:10:53] INFO [shared.components.visualization] [Visualization] Rendering chart: 1000 points, density=Opacity, bins=N/A
[2026-01-29 11:10:53] INFO [shared.services.clustering_service] Generating clustering summary statistics
[2026-01-29 11:10:54] INFO [shared.components.visualization] [Visualization] Point selected: idx=589, cluster=9
[2026-01-29 11:10:54] INFO [shared.services.clustering_service] Generating clustering summary statistics
```

### Expected behavior

- `Generating clustering summary statistics` should only appear once after clicking "Run Clustering"
- Subsequent renders (zoom, pan, point selection, density mode change) should use cached results from session state

### Current implementation

The fix attempted in commit `07a66a9` stores summary in session state (`clustering_summary`, `clustering_representatives`) but there appears to be another code path still calling `ClusteringService.generate_clustering_summary()`.

### Files to investigate

- `shared/components/summary.py` - `render_clustering_summary()` function
- `apps/embed_explore/components/sidebar.py` - clustering execution
- Check if there are other places calling `generate_clustering_summary`

### Impact

- **Performance:** unnecessary computation on every render
- **User experience:** potential lag during interactions

### Root cause

The `apps/embed_explore/app.py` was importing from its local `apps.embed_explore.components.summary` instead of the shared `shared.components.summary`. The local version was calling `ClusteringService.generate_clustering_summary()` directly on every render.

### Fix

- Updated `apps/embed_explore/app.py` to import from `shared.components.summary`
- Updated local `summary.py` to re-export from shared for backwards compatibility
