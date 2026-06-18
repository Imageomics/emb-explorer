---
title: Image Embedding Explorer (Precalculated Demo)
emoji: 🔍
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Filter, project, and cluster precalculated image embeddings
tags:
  - biodiversity
  - embeddings
  - bioclip
  - clustering
  - dimensionality-reduction
  - umap
  - tsne
  - visualization
  - imageomics
datasets:
  - imageomics/TreeOfLife-200M-Embeddings
models:
  - imageomics/bioclip-2
---

# Image Embedding Explorer — Precalculated Demo

Hosted demo of the [emb-explorer](https://github.com/Imageomics/emb-explorer)
precalculated embeddings app. Pick a curated BioCLIP 2 dataset (Darwin's
finches or wolves), project it to 2D, color by metadata, and cluster.

## How it works

- The app code (`apps/` + `shared/`) is deployed manually from the
  `feature/hf-space-precalculated-demo` branch of
  [emb-explorer](https://github.com/Imageomics/emb-explorer) — no GitHub
  clone or CI sync. The Dockerfile builds straight from the pushed files.
- Dependencies are a precalc-only subset (`requirements-space.txt`); the
  embedding-generation stack (torch / open-clip) is intentionally excluded.
- The curated demo data lives in the [`netzhang/demo`](https://huggingface.co/datasets/netzhang/demo)
  dataset, **mounted read-only at `/data`** via a Space volume. Files are
  fetched lazily, so the full TreeOfLife-200M embeddings can be mounted
  without consuming disk.

## Volume mount (one-time setup, out of band)

```bash
hf spaces volumes set netzhang/emb-explorer-demo \
  -v hf://datasets/netzhang/demo:/data
```

The app reads `/data/demo_subset/<dataset>/bioclip-2_float16/emb_*.parquet`
(controlled by the `EMB_EXPLORER_DEMO_DATA_ROOT` env var, default `/data`).
