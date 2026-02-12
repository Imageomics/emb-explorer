# Test Suite

Hey! Welcome to the emb-explorer test suite. This doc is for humans *and* AI coding agents (hi Claude) — so it's kept concise and structured.

## Quick Start

See the main [README](../README.md) for environment setup. Once your venv is activated:

```bash
# Run everything (CPU tests)
pytest tests/ -v

# Run a specific file
pytest tests/test_backend.py -v

# CPU tests only (skip GPU-marked tests)
pytest tests/ -m "not gpu"
```

> **Heads up:** TSNE/UMAP tests are slow on CPU-only nodes (~12 min total). PCA and everything else is fast. On GPU nodes the full suite runs much quicker.

## Running on GPU Nodes

All current tests run on CPU, but some (UMAP, t-SNE) are significantly faster on a GPU node. If your cluster uses SLURM:

```bash
# Interactive
salloc --partition=gpu --gpus-per-node=1 --time=00:30:00
# activate venv, then:
pytest tests/ -v

# Or submit via the batch script (set VENV_DIR first)
VENV_DIR=/path/to/venvs sbatch tests/run_gpu_tests.sh
```

> The `@pytest.mark.gpu` marker is registered for future GPU-specific tests (e.g. real cuML/FAISS-GPU code paths). No tests use it yet — all 98 tests pass on CPU-only nodes.

## What's Tested

| File | Target Module | Tests | What It Covers |
|---|---|---|---|
| `test_clustering.py` | `shared/utils/clustering.py` | 23 | L2 normalization, dim reduction (sklearn), KMeans (sklearn), GPU fallback via mocked cupy |
| `test_backend.py` | `shared/utils/backend.py` | 29 | Error classifiers (`is_gpu_error`, `is_oom_error`, `is_cuda_arch_error`), backend resolution priority, CUDA cache |
| `test_clustering_service.py` | `shared/services/clustering_service.py` | 8 | `generate_clustering_summary()` correctness, `run_clustering_safe()` fallback chain |
| `test_filters.py` | `apps/precalculated/components/sidebar.py` | 16 | PyArrow filter logic (categorical/numeric/text/AND), column type detection, embedding extraction |
| `test_taxonomy_tree.py` | `shared/utils/taxonomy_tree.py` | 12 | Tree building, NaN handling, depth/count filtering, statistics |
| `test_logging_config.py` | `shared/utils/logging_config.py` | 5 | Logger naming, handler setup, idempotency, file handler creation |
| `conftest.py` | — | — | Shared fixtures (embeddings, paths, PyArrow tables, reset helpers) |

**Total: 98 tests across 6 files.**

## What's NOT Tested (and why)

- **Streamlit UI components** (`shared/components/visualization.py`, `summary.py`) — mostly Altair chart rendering. Testing visual output has low ROI.
- **Image fetching** (`data_preview.py`) — requires HTTP mocking for external URLs. Low priority.

## Design Principles

- **CPU tests need no GPU.** All 98 tests pass on login/compute nodes without CUDA.
- **GPU fallback is tested by mocking** — we patch `HAS_CUML`, `HAS_CUDA`, `cp` (cupy), and `subprocess.run` to simulate GPU failures and verify the fallback chain.
- **GPU execution on real hardware** — `@pytest.mark.gpu` is registered for future tests that exercise actual cuML/FAISS-GPU code paths.
- **Pure functions are tested directly** — `_prepare_embeddings()`, `apply_filters_arrow()`, `build_taxonomic_tree()`, error classifiers, etc. No mocking needed.
- **Small data** — fixtures use 10-100 samples to keep tests fast.

## For AI Agents

If you're adding new utility functions to `shared/utils/` or `shared/services/`:

1. **Add tests.** Check if an existing test file covers the module, or create a new one.
2. **Use the fixtures** in `conftest.py` — `sample_embeddings`, `sample_embeddings_small`, `sample_arrow_table`, etc.
3. **Mock GPU code**, don't try to call it. Patch module-level flags like `HAS_CUML` or inject mock objects for `cp` (cupy).
4. **Run `pytest tests/ -v`** after changes to verify nothing broke.
5. The `reset_cuda_cache` and `reset_logging` fixtures exist because those modules use global state — use them when testing `backend.py` or `logging_config.py`.
6. **GPU tests** (future) use `@pytest.mark.gpu`. These only run on GPU nodes — don't expect them to pass on CPU-only nodes.

## Markers

| Marker | Purpose |
|---|---|
| `@pytest.mark.gpu` | Requires CUDA GPU. Reserved for future GPU-specific tests. Run via `pytest -m gpu`. |

Registered in `pyproject.toml` under `[tool.pytest.ini_options]`.
