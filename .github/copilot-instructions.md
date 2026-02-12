# Copilot Code Review Instructions

## Project context

This is a Streamlit-based image-embedding explorer that runs on HPC GPU
clusters (Ohio Supercomputer Center, SLURM). It has an automatic backend
fallback chain: cuML (GPU) → FAISS (CPU) → scikit-learn (CPU). Optional
GPU dependencies (cuML, CuPy, PyTorch, FAISS-GPU) may or may not be
installed — the app detects them at runtime and degrades gracefully.

## Review focus

Prioritise **logic bugs, security issues, and correctness problems** over
style or lint. We run linters separately. A review comment should tell
us something a linter cannot.

## Patterns to accept (do NOT flag these)

- **`except (ImportError, Exception): pass` with an inline comment** —
  These are intentional graceful-degradation paths for optional GPU
  dependencies. If the comment explains the intent, do not suggest adding
  logging or replacing the bare pass.

- **Self-referencing extras in `pyproject.toml`** — e.g.
  `gpu = ["emb-explorer[gpu-cu12]"]`. This is a supported pip feature
  for aliasing optional-dependency groups. It is not a circular dependency.

- **`faiss-gpu-cu12` inside a `[gpu-cu13]` extra** — There is no
  `faiss-gpu-cu13` package on PyPI. CUDA forward-compatibility means the
  cu12 build works on CUDA 13 drivers. If a comment explains this, accept it.

- **Streamlit `st.rerun(scope="app")`** — The `scope` parameter has been
  available since Streamlit 1.33 (2024). `scope="app"` from inside a
  `@st.fragment` triggers a full page rerun. This is intentional.

- **PID-based temp files under `/dev/shm`** — Used for subprocess IPC in
  cuML UMAP isolation. The subprocess is short-lived and files are cleaned
  up in a `finally` block. This is acceptable for a single-user HPC app.

## Things worth flagging

- Version-specifier bugs in `pyproject.toml` (e.g. `<=X.Y.0` excluding
  valid patch releases when the real constraint is `<X.Z`).
- Incorrect error handling that swallows exceptions *without* a comment.
- Security issues: command injection, unsanitised user input, secrets in code.
- Race conditions or state bugs in Streamlit session state.
- GPU memory leaks (cupy/torch tensors not freed).
