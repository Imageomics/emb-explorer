"""Shared "Export analysis" section (#48).

Downloads the session's analysis as a flat CSV (record key, every retained
projection run as ``<tag>_x``/``<tag>_y``, every ``KMeans (k=...)`` column,
optionally the original metadata) plus a provenance JSON sidecar describing
how each column was produced. Column selection and the sidecar live in
``shared/utils/provenance.py``; this module is only the Streamlit surface.
"""

import json

import streamlit as st

from shared.utils.provenance import (
    build_export_table,
    build_sidecar,
    export_basename,
    kmeans_columns,
    new_registry,
    projection_tags,
)


def render_export_section(app: str) -> None:
    """Render the export expander. ``app`` is "embed_explore" or "precalculated"."""
    with st.expander("Export analysis", expanded=False):
        df_plot = st.session_state.get("data")
        if df_plot is None:
            st.info("Run projection first to enable export.")
            return

        key_col = "uuid" if "uuid" in df_plot.columns else "image_path"
        source = str(
            st.session_state.get("parquet_file_path")
            or st.session_state.get("last_image_dir")
            or "unknown"
        )

        include_metadata = st.checkbox(
            "Include original metadata",
            value=True,
            key="export_include_metadata",
            help=(
                "On: the CSV is self-contained (metadata columns included). "
                "Off: only the key, projection and KMeans columns; join back "
                "to your source data on the key column."
            ),
        )

        table = build_export_table(df_plot, key_col, include_metadata)
        registry = st.session_state.get("provenance") or new_registry()
        sidecar = build_sidecar(
            df_plot,
            registry,
            app=app,
            key_col=key_col,
            source=source,
            include_metadata=include_metadata,
            exported_columns=list(table.columns),
            filters=st.session_state.get("active_filters") or {},
        )
        base = export_basename(source, sidecar["exported_at"])

        st.caption(
            f"{len(table):,} rows x {len(table.columns)} columns; "
            f"{len(projection_tags(df_plot))} projection run(s), "
            f"{len(kmeans_columns(df_plot))} KMeans run(s). "
            "Embedding vectors are never exported."
        )

        st.download_button(
            "Download table (CSV)",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"{base}.csv",
            mime="text/csv",
            key="export_csv_btn",
            width="stretch",
        )
        st.download_button(
            "Download provenance (JSON)",
            data=json.dumps(sidecar, indent=2, default=str).encode("utf-8"),
            file_name=f"{base}.json",
            mime="application/json",
            key="export_json_btn",
            width="stretch",
        )
