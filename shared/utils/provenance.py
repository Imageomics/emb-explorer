"""Analysis provenance: projection tags, analysis-column conventions, and the
export builders behind "Export analysis" (#48).

Data model
----------
``df_plot`` keeps ``x``/``y`` as the *current* view and additionally retains
every projection run of the session as a ``<tag>_x`` / ``<tag>_y`` pair, where
``tag = <method>_s<seed>`` for seeded runs (``tsne_s614``) and
``<method>_r<n>`` for unseeded ones (``tsne_r2``; each unseeded run is a
legitimately different view, so they never overwrite each other). Retained
projection pairs and ``KMeans (k=...)`` columns are carried across re-projection
by :func:`carry_over_analysis_columns`.

Alongside the columns, a session-level *registry* records how each analysis
column was produced (method, backend actually used, seed, key params,
timestamp). :func:`build_export_table` and :func:`build_sidecar` turn both into
the CSV + JSON export.

Streamlit-free so it can be unit-tested and reused outside the apps.
"""

import hashlib
import importlib.metadata
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

KMEANS_PREFIX = "KMeans (k="
# Columns holding embedding vectors; never exported.
VECTOR_COLUMNS = {"emb", "embedding", "embeddings", "vector"}
# Chart-internal columns; never exported (the current view is duplicated by
# its tagged projection pair).
INTERNAL_COLUMNS = {"x", "y", "idx"}

_PROJECTION_COL_RE = re.compile(r"^(?P<tag>[a-z0-9]+_(?:s-?\d+|r\d+))_(?P<axis>[xy])$")
_PROJECTION_TAG_RE = re.compile(r"^[a-z0-9]+_(?:s-?\d+|r\d+)$")


# ---------------------------------------------------------------------------
# Column conventions
# ---------------------------------------------------------------------------

def make_projection_tag(method: str, seed: Optional[int], existing: Iterable[str]) -> str:
    """Tag for a projection run: ``<method>_s<seed>`` or ``<method>_r<n>``.

    Seeded runs of the same method share a tag (re-running replaces the view);
    unseeded runs get the next free run counter among ``existing`` tags.
    """
    m = method.lower()
    if seed is not None:
        return f"{m}_s{seed}"
    used = {t for t in existing if re.fullmatch(rf"{re.escape(m)}_r\d+", t)}
    n = 1
    while f"{m}_r{n}" in used:
        n += 1
    return f"{m}_r{n}"


def projection_tag_of(column: str) -> Optional[str]:
    """``'tsne_s614_x'`` -> ``'tsne_s614'``; None for non-projection columns."""
    match = _PROJECTION_COL_RE.match(column)
    return match.group("tag") if match else None


def is_projection_column(column: str) -> bool:
    return _PROJECTION_COL_RE.match(column) is not None


def is_kmeans_column(column: str) -> bool:
    return column.startswith(KMEANS_PREFIX)


def is_analysis_column(column: str) -> bool:
    return is_projection_column(column) or is_kmeans_column(column)


def projection_tags(df: pd.DataFrame) -> List[str]:
    """Tags with both ``_x`` and ``_y`` present, in column order."""
    tags: List[str] = []
    for col in df.columns:
        tag = projection_tag_of(col)
        if tag and tag not in tags and f"{tag}_x" in df.columns and f"{tag}_y" in df.columns:
            tags.append(tag)
    return tags


def projection_columns(df: pd.DataFrame) -> List[str]:
    return [f"{tag}_{axis}" for tag in projection_tags(df) for axis in ("x", "y")]


def kmeans_k_of(column: str) -> int:
    """``'KMeans (k=5)'`` -> ``5``."""
    return int(column[len(KMEANS_PREFIX):].rstrip(")"))


def kmeans_columns(df: pd.DataFrame) -> List[str]:
    """``KMeans (k=...)`` columns sorted by k."""
    return sorted((c for c in df.columns if is_kmeans_column(c)), key=kmeans_k_of)


def export_column_name(column: str) -> str:
    """Tidy, tool-friendly name for a column in the exported CSV.

    In-app KMeans columns keep their display label (``KMeans (k=5)``), which
    is awkward as a CSV header (spaces, parentheses, ``=``); they export as
    ``kmeans_k5``, matching the projection tag style. Every other column
    exports under its own name. The sidecar records the mapping.
    """
    if is_kmeans_column(column):
        return f"kmeans_k{kmeans_k_of(column)}"
    return column


def carry_over_analysis_columns(prev_df: Optional[pd.DataFrame], new_df: pd.DataFrame) -> pd.DataFrame:
    """Copy retained projection pairs and KMeans columns from the previous
    ``df_plot`` onto a freshly built one (in place; also returned).

    Positional, like the KMeans carry-over it generalizes: skipped when the
    row count differs (e.g. the precalculated app's filters changed). Columns
    already on ``new_df`` win, so a re-run of the same seeded tag replaces the
    old view.
    """
    if prev_df is None or len(prev_df) != len(new_df):
        return new_df
    for col in prev_df.columns:
        if is_analysis_column(col) and col not in new_df.columns:
            new_df[col] = prev_df[col].values
    return new_df


# ---------------------------------------------------------------------------
# Registry (session-level record of how each analysis column was produced)
# ---------------------------------------------------------------------------

def new_registry() -> Dict[str, Dict[str, Any]]:
    return {"projections": {}, "kmeans": {}}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_projection(
    registry: Dict[str, Dict[str, Any]],
    tag: str,
    *,
    method: str,
    backend: str,
    requested_backend: str,
    seed: Optional[int],
    params: Optional[Dict[str, Any]] = None,
    n_samples: int,
    elapsed_seconds: float,
    fallback_from: Optional[str] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "method": method.upper(),
        "backend": backend,
        "requested_backend": requested_backend,
        "seed": seed,
        "params": dict(params or {}),
        "n_samples": int(n_samples),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "run_at": _now_iso(),
    }
    if fallback_from:
        entry["fallback_from"] = fallback_from
    registry["projections"][tag] = entry
    return entry


def record_kmeans(
    registry: Dict[str, Dict[str, Any]],
    column: str,
    *,
    k: int,
    backend: str,
    requested_backend: str,
    seed: Optional[int],
    n_workers: int,
    elapsed_seconds: float,
    fallback_from: Optional[str] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "k": int(k),
        "backend": backend,
        "requested_backend": requested_backend,
        "seed": seed,
        "n_workers": int(n_workers),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "run_at": _now_iso(),
    }
    if fallback_from:
        entry["fallback_from"] = fallback_from
    registry["kmeans"][column] = entry
    return entry


def prune_registry(registry: Dict[str, Dict[str, Any]], df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Drop registry entries whose columns are no longer on ``df`` (e.g. after
    a row-count change made the carry-over skip them)."""
    present_tags = set(projection_tags(df))
    registry["projections"] = {t: e for t, e in registry["projections"].items() if t in present_tags}
    registry["kmeans"] = {c: e for c, e in registry["kmeans"].items() if c in df.columns}
    return registry


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def record_set_hash(keys: Iterable[Any]) -> str:
    """Order-independent fingerprint of the *set of records* in an export
    (their uuids / paths), not of the data contents.

    Same parquet + same filters gives the same hash; a changed filter, or an
    added/dropped record, changes it. Row order does not matter. Embedding
    values and metadata contents are not covered.
    """
    joined = "\n".join(sorted(str(k) for k in keys))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


def library_versions() -> Dict[str, Optional[str]]:
    """Installed versions of the libraries that produce projections/clusters.

    Uses package metadata only; importing cuml is slow and needs a GPU.
    """
    def _version(*dists: str) -> Optional[str]:
        for dist in dists:
            try:
                return importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                continue
        return None

    return {
        "emb-explorer": _version("emb-explorer"),
        "scikit-learn": _version("scikit-learn"),
        "umap-learn": _version("umap-learn"),
        "cuml": _version("cuml-cu12", "cuml-cu13", "cuml"),
    }


def metadata_columns(df: pd.DataFrame, key_col: str) -> List[str]:
    """Original metadata columns: everything that is not the key, an analysis
    column, chart-internal, or an embedding vector."""
    return [
        c for c in df.columns
        if c != key_col
        and not is_analysis_column(c)
        and c not in INTERNAL_COLUMNS
        and c not in VECTOR_COLUMNS
    ]


def build_export_table(df_plot: pd.DataFrame, key_col: str, include_metadata: bool = True) -> pd.DataFrame:
    """Flat export table: key, projection pairs, KMeans columns, [metadata]."""
    if key_col not in df_plot.columns:
        raise KeyError(f"key column {key_col!r} not in df_plot")
    cols = [key_col] + projection_columns(df_plot) + kmeans_columns(df_plot)
    if include_metadata:
        cols += metadata_columns(df_plot, key_col)
    table = df_plot.loc[:, cols].copy()
    return table.rename(columns={c: export_column_name(c) for c in cols})


def build_sidecar(
    df_plot: pd.DataFrame,
    registry: Dict[str, Dict[str, Any]],
    *,
    app: str,
    key_col: str,
    source: str,
    include_metadata: bool,
    exported_columns: List[str],
    filters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Provenance JSON for an export: environment, dataset (including the
    filters that selected its records, when the app has any), and one entry
    per projection tag / KMeans column present in the CSV. KMeans entries are
    keyed by the exported column name and carry ``source_column`` (the in-app
    label) so the two can be mapped back."""
    present_tags = projection_tags(df_plot)
    present_kmeans = kmeans_columns(df_plot)
    return {
        "format": "emb-explorer-analysis-export/1",
        "app": app,
        "exported_at": _now_iso(),
        "libraries": library_versions(),
        "dataset": {
            "source": source,
            "key_column": key_col,
            "n_records": int(len(df_plot)),
            "filters": dict(filters or {}),
            "record_set_hash": record_set_hash(df_plot[key_col].tolist()),
        },
        "include_metadata": bool(include_metadata),
        "columns": list(exported_columns),
        "projections": {t: registry.get("projections", {}).get(t, {}) for t in present_tags},
        "kmeans": {
            export_column_name(c): {"source_column": c, **registry.get("kmeans", {}).get(c, {})}
            for c in present_kmeans
        },
    }


def export_basename(source: str, exported_at: str) -> str:
    """``emb-explorer_<dataset-stem>_<YYYYmmdd-HHMMSS>``."""
    stem = Path(str(source)).stem or "analysis"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-") or "analysis"
    stamp = datetime.fromisoformat(exported_at).strftime("%Y%m%d-%H%M%S")
    return f"emb-explorer_{stem}_{stamp}"
