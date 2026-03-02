# Backend Pipeline

A quick walkthrough of what happens to your embeddings from the moment you click
"Run Clustering" to the scatter plot on screen.

## The Pipeline at a Glance

```
Raw Embeddings (from parquet or model)
  │
  ├─ Validate: check for NaN/Inf, cast to float32
  ├─ L2 Normalize: project onto unit hypersphere
  │
  ├─► Step 1: KMeans Clustering (high-dimensional)
  │     Backend: cuML → FAISS → sklearn
  │
  ├─► Step 2: Dimensionality Reduction to 2D
  │     Method:  PCA / t-SNE / UMAP
  │     Backend: cuML → sklearn
  │
  └─► Scatter Plot (Altair)
        Color = cluster, position = 2D projection
```

## Step 0: Embedding Preparation

Before any computation, every embedding goes through `_prepare_embeddings()`:

1. **Cast to float32** — GPU backends require it; keeps memory predictable.
2. **NaN/Inf check** — replaces bad values with 0 and logs a warning.
3. **L2 normalization** — divides each vector by its magnitude so every point
   sits on the unit hypersphere. This is critical for two reasons:
   - Prevents cuML UMAP's NN-descent from crashing with SIGFPE on
     large-magnitude vectors (see `investigation/cuml_umap_sigfpe/`).
   - Appropriate for contrastive embeddings (CLIP, BioCLIP) whose training
     objective is cosine-similarity based — magnitude isn't a learned signal.

Input norms are logged so you can always verify what came in.

## Step 1: KMeans Clustering

Clusters the full high-dimensional embeddings (e.g., 768-d for BioCLIP 2).
Runs *before* dimensionality reduction so clusters are based on the full
feature space, not a lossy 2D projection.

| Backend | When It's Used | How It Works |
|---------|---------------|--------------|
| **cuML** | GPU available + >500 samples | GPU-accelerated KMeans via RAPIDS. Runs on CuPy arrays. Falls back to sklearn on any error. |
| **FAISS** | No GPU + >500 samples | Facebook's optimized CPU KMeans using L2 index. Fast for medium datasets. Falls back to sklearn on error. |
| **sklearn** | Small datasets or fallback | Standard scikit-learn KMeans. Always works, no special dependencies. |

**Auto-selection priority:** cuML > FAISS > sklearn. You can override in the sidebar.

## Step 2: Dimensionality Reduction

Projects embeddings from high-dimensional space down to 2D for visualization.
This is purely for the scatter plot — clustering uses the full-dimensional data.

### PCA (Principal Component Analysis)

The fastest option. Linear projection onto the two directions of maximum variance.
Good for getting a quick overview; doesn't capture nonlinear structure.

| Backend | Notes |
|---------|-------|
| **cuML** | GPU-accelerated, near-instant even on large datasets |
| **sklearn** | CPU-based, still fast since PCA is O(n) |

### t-SNE

Nonlinear method that preserves local neighborhoods. Good at revealing clusters
but slow on large datasets. Perplexity is auto-adjusted based on sample size.

| Backend | Notes |
|---------|-------|
| **cuML** | GPU-accelerated, handles thousands of samples well |
| **sklearn** | CPU-based, can be slow above ~5k samples |

### UMAP

The recommended default. Nonlinear like t-SNE but faster and better at
preserving global structure. Neighbor count is auto-adjusted.

| Backend | Notes |
|---------|-------|
| **cuML** | Runs in an **isolated subprocess** so a crash doesn't kill the app. The subprocess verifies L2 normalization as a safety net. Falls back to sklearn on failure. |
| **sklearn** | CPU-based `umap-learn`. Slower but numerically stable. |

**Why the subprocess?** cuML UMAP's NN-descent algorithm can occasionally trigger
a SIGFPE (floating-point exception) that kills the process instantly — no Python
try/except can catch it. The subprocess isolates this risk.

## Backend Selection

When you select "auto" (the default), the app picks the fastest available backend:

| Operation | Auto Logic |
|-----------|-----------|
| KMeans | cuML if GPU + >500 samples, else FAISS if available + >500 samples, else sklearn |
| Dim. Reduction | cuML if GPU + >5000 samples, else sklearn |

Any GPU error (architecture mismatch, missing libraries, out of memory (OOM)) triggers an
automatic retry with sklearn. OOM errors are surfaced to the user with guidance.

## Logging

Every step is logged to `logs/emb_explorer.log` (DEBUG level) and console (INFO):

- Embedding extraction: shape, dtype
- Preparation: input norms (min/max/mean), non-finite count, L2 normalization
- Backend selection: which backend was chosen and why
- KMeans: cluster count, sample count, elapsed time
- Reduction: method, sample count, elapsed time
- Fallbacks: what failed and what we fell back to
- Visualization: point selection events, density mode changes

Check the log file for the full picture when debugging.

## GPU Fallback Chain

```
cuML (GPU)
  │ error?
  ▼
FAISS (CPU, optimized)     ← KMeans only
  │ error?
  ▼
sklearn (CPU, always works)
```

The app is designed to *always produce a result*. GPU acceleration is a
nice-to-have, never a hard requirement.
