# Test Suite

Hey! Welcome to the emb-explorer test suite. This doc is for humans *and* AI coding agents (hi Claude) — so it's kept concise and structured.

## Quick Start

Once your venv is activated:

```bash
pytest tests/ -v                    # all tests
pytest tests/test_backend.py -v     # specific file
pytest tests/ -m "not gpu"          # skip GPU-marked tests
```

> **Heads up:** TSNE/UMAP tests are slow on CPU (~1 min). Everything else is fast. Much quicker on GPU nodes.

## Test Organization

| File | What It Covers |
|---|---|
| `test_backend.py` (29) | Error classifiers, backend resolution priority, CUDA cache |
| `test_clustering.py` (23) | L2 normalization, dim reduction, KMeans, GPU fallback (mocked) |
| `test_filters.py` (16) | PyArrow filter logic, column type detection, embedding extraction |
| `test_taxonomy_tree.py` (12) | Tree building, NaN handling, depth/count filtering |
| `test_clustering_service.py` (8) | Clustering summary, `run_clustering_safe()` fallback chain |
| `test_logging_config.py` (5) | Logger naming, handler setup, idempotency |
| `conftest.py` | Shared fixtures (embeddings, paths, PyArrow tables, reset helpers) |

**98 tests total.** All pass on CPU-only machines — no GPU required. GPU fallback behavior is tested via mocking (`HAS_CUML`, `HAS_CUDA`, `subprocess.run`). The `@pytest.mark.gpu` marker is registered for future tests that exercise real GPU code paths.

## Running on a SLURM Cluster

Two batch scripts are provided in `tests/`. Before using them, edit the `#SBATCH` headers to match your cluster (account, partition names, venv path):

```bash
sbatch tests/run_cpu_tests.sh             # CPU partition — runs non-GPU tests
sbatch tests/run_gpu_tests.sh             # GPU partition — runs full suite
sbatch tests/run_gpu_tests.sh --gpu       # GPU partition — GPU-marked tests only
```

The GPU script sets `LD_LIBRARY_PATH` for cuML/CuPy nvidia libs automatically.

## For AI Agents

If you're adding new utility functions to `shared/utils/` or `shared/services/`:

1. **Add tests.** Check if an existing test file covers the module, or create a new one.
2. **Use the fixtures** in `conftest.py` — `sample_embeddings`, `sample_embeddings_small`, `sample_arrow_table`, etc.
3. **Mock GPU code**, don't try to call it. Patch module-level flags like `HAS_CUML` or inject mock objects for `cp` (cupy).
4. **Run `pytest tests/ -v`** after changes to verify nothing broke.
5. The `reset_cuda_cache` and `reset_logging` fixtures exist because those modules use global state — use them when testing `backend.py` or `logging_config.py`.
6. **GPU tests** (future) use `@pytest.mark.gpu`. These only run on GPU nodes — don't expect them to pass on CPU-only nodes.
