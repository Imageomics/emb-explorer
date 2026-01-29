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

---

## Issue #2: Slow app startup due to heavy library imports

**Status:** Fixed
**Branch:** `feature/viz-altair-interactive`
**Date:** 2026-01-29

### Problem

The app had a long startup time because heavy libraries (FAISS, torch, open_clip, cuML) were being imported at module load time, even when they weren't needed immediately.

### Evidence from logs

```
[2026-01-29 11:14:11] INFO [faiss.loader] Loading faiss with AVX512 support.
[2026-01-29 11:14:11] INFO [faiss.loader] Successfully loaded faiss with AVX512 support.
```

These messages appeared during app startup before any user action.

### Expected behavior

Heavy libraries should only be loaded when explicitly needed:
- FAISS: only when user selects FAISS backend or auto-resolution chooses it
- torch/open_clip: only when user runs embedding generation
- cuML: only when user selects cuML backend

### Root cause

Multiple files had module-level imports of heavy libraries:
- `shared/utils/clustering.py` - imported sklearn, UMAP at module level
- `shared/utils/models.py` - imported `open_clip` at module level
- `shared/services/embedding_service.py` - imported `torch` and `open_clip` at module level
- `shared/components/clustering_controls.py` - imported `faiss` and `cuml` for availability check
- `shared/utils/backend.py` - availability checks weren't cached

### Fix

Implemented lazy loading pattern across all affected files:

1. **`shared/utils/clustering.py`**: Converted module-level imports to lazy-load functions (`_get_sklearn_modules()`, `_get_umap_module()`, `_check_faiss_available()`, `_check_cuml_available()`)

2. **`shared/utils/models.py`**: Added `_get_open_clip()` lazy loader

3. **`shared/services/embedding_service.py`**: Added `_get_torch()` and `_get_open_clip()` lazy loaders, moved `@torch.no_grad()` decorator to context manager

4. **`shared/components/clustering_controls.py`**: Added `_get_backend_availability()` with caching, only checks when user expands backend controls

5. **`shared/utils/backend.py`**: Added caching for `check_faiss_available()` and `check_cuml_available()`

### Verification

```python
# Before imports: []
# After module imports: []  # No heavy libraries loaded!
# After calling check_faiss_available(): ['faiss']  # Only loaded when needed
```
