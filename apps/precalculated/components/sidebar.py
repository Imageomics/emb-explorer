"""
Sidebar components for the precalculated embeddings application.
Features dynamic cascading filter generation based on parquet columns.
"""

import streamlit as st
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
import os
import time
import hashlib
from typing import Dict, Any, Optional, Tuple, List

from shared.services.clustering_service import ClusteringService
from shared.components.clustering_controls import render_clustering_backend_controls
from shared.utils.backend import check_cuda_available, resolve_backend, is_oom_error
from shared.utils.logging_config import get_logger

logger = get_logger(__name__)


# Technical columns that should never be shown as filters
EXCLUDED_COLUMNS = {'uuid', 'emb', 'embedding', 'embeddings', 'vector'}


def get_column_info_dynamic(table: pa.Table) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically analyze all columns in a PyArrow table for filtering.

    Args:
        table: PyArrow Table to analyze

    Returns:
        Dictionary mapping column names to their info (type, unique_values, etc.)
    """
    column_info = {}

    for col_name in table.column_names:
        # Skip technical/excluded columns
        if col_name.lower() in EXCLUDED_COLUMNS:
            continue

        col_array = table.column(col_name)

        # Handle null values
        non_null_mask = pc.is_valid(col_array)
        non_null_count = pc.sum(non_null_mask).as_py()
        total_count = len(col_array)
        null_count = total_count - non_null_count

        if non_null_count == 0:
            col_type = 'empty'
            unique_values = []
            value_counts = {}
        else:
            # Check data type
            arrow_type = col_array.type

            if (pa.types.is_integer(arrow_type) or
                pa.types.is_floating(arrow_type) or
                pa.types.is_decimal(arrow_type)):
                col_type = 'numeric'
                unique_values = None
                value_counts = None
            elif pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
                # Skip list/array columns (like embeddings)
                continue
            else:
                # Get unique values for categorical determination
                try:
                    unique_array = pc.unique(col_array)
                    unique_count = len(unique_array)

                    if unique_count <= 100:  # Categorical if <= 100 unique values
                        col_type = 'categorical'
                        unique_values = sorted([v.as_py() for v in unique_array if v.is_valid])

                        # Get value counts
                        value_counts_result = pc.value_counts(col_array)
                        value_counts = {}
                        for i in range(len(value_counts_result)):
                            struct = value_counts_result[i].as_py()
                            if struct['values'] is not None:
                                value_counts[struct['values']] = struct['counts']
                    else:
                        col_type = 'text'
                        unique_values = None
                        value_counts = None
                except Exception:
                    col_type = 'text'
                    unique_values = None
                    value_counts = None

        column_info[col_name] = {
            'type': col_type,
            'unique_values': unique_values,
            'value_counts': value_counts,
            'null_count': null_count,
            'total_count': total_count,
            'null_percentage': (null_count / total_count) * 100 if total_count > 0 else 0
        }

    return column_info


def get_cascading_options(
    table: pa.Table,
    target_column: str,
    current_filters: Dict[str, Any],
    column_info: Dict[str, Dict[str, Any]]
) -> List[str]:
    """
    Get available options for a column based on other active filters.
    This enables cascading/dependent filter behavior.

    Args:
        table: Full PyArrow table
        target_column: Column to get options for
        current_filters: Currently selected filter values (excluding target_column)
        column_info: Column metadata

    Returns:
        List of unique values available for the target column given other filters
    """
    # Build filters excluding the target column
    other_filters = {k: v for k, v in current_filters.items() if k != target_column and v}

    if not other_filters:
        # No other filters, return original unique values
        info = column_info.get(target_column, {})
        return info.get('unique_values', []) or []

    # Apply other filters to get subset
    filtered_table = apply_filters_arrow(table, other_filters)

    if target_column not in filtered_table.column_names:
        return []

    # Get unique values from filtered subset
    try:
        col_array = filtered_table.column(target_column)
        unique_array = pc.unique(col_array)
        return sorted([v.as_py() for v in unique_array if v.is_valid])
    except Exception:
        return column_info.get(target_column, {}).get('unique_values', []) or []


def render_file_section() -> Tuple[bool, Optional[str]]:
    """
    Render the file loading section.

    Returns:
        Tuple of (file_loaded, file_path)
    """
    with st.expander("📁 Load Parquet", expanded=True):
        file_path = st.text_input(
            "Parquet file or directory path",
            value=st.session_state.get("parquet_file_path", ""),
            help="Path to a parquet file or directory of parquet files containing embeddings and metadata"
        )

        load_button = st.button("Load File", type="primary")

        if load_button and file_path and os.path.exists(file_path):
            try:
                logger.info(f"Loading parquet file: {file_path}")
                with st.spinner("Loading parquet file..."):
                    table = pq.read_table(file_path)
                    df = table.to_pandas()

                logger.info(f"Loaded {len(df):,} records, {len(table.column_names)} columns, "
                            f"schema: {[f'{c.name}({c.type})' for c in table.schema]}")

                # Validate required columns
                if 'uuid' not in table.column_names:
                    st.error("Missing required 'uuid' column")
                    logger.error("Parquet validation failed: missing 'uuid' column")
                    return False, file_path
                if 'emb' not in table.column_names:
                    st.error("Missing required 'emb' column")
                    logger.error("Parquet validation failed: missing 'emb' column")
                    return False, file_path

                emb_dim = len(df['emb'].iloc[0])
                logger.info(f"Embedding dimension: {emb_dim}")

                # Dynamically analyze all columns
                column_info = get_column_info_dynamic(table)
                logger.info(f"Column analysis: {len(column_info)} filterable columns "
                            f"({sum(1 for v in column_info.values() if v['type'] == 'categorical')} categorical, "
                            f"{sum(1 for v in column_info.values() if v['type'] == 'numeric')} numeric, "
                            f"{sum(1 for v in column_info.values() if v['type'] == 'text')} text)")

                # Store in session state
                st.session_state.parquet_table = table
                st.session_state.parquet_df = df
                st.session_state.parquet_file_path = file_path
                st.session_state.column_info = column_info

                # Reset downstream state
                st.session_state.filtered_df = None
                st.session_state.embeddings = None
                st.session_state.data = None
                st.session_state.labels = None
                st.session_state.selected_image_idx = None
                st.session_state.active_filters = {}
                st.session_state.pending_filters = {}

                st.success(f"Loaded {len(df):,} records with {len(column_info)} filterable columns")
                st.info(f"Embedding dimension: {emb_dim}")

                return True, file_path

            except Exception as e:
                st.error(f"Error loading file: {e}")
                logger.exception(f"Failed to load parquet file: {file_path}")
                return False, file_path

        elif load_button and file_path:
            st.error(f"File not found: {file_path}")
            return False, file_path
        elif load_button:
            st.error("Please provide a file path")
            return False, None

    return False, file_path


def render_dynamic_filters() -> Dict[str, Any]:
    """
    Render dynamically generated cascading filters based on parquet columns.
    Filter options update based on other selected filters (AND logic).

    Returns:
        Dictionary of applied filters
    """
    with st.expander("🔍 Filter Data", expanded=True):
        df = st.session_state.get("parquet_df", None)
        table = st.session_state.get("parquet_table", None)
        column_info = st.session_state.get("column_info", {})

        if df is None or table is None:
            st.info("Load a parquet file first to enable filtering.")
            return {}

        st.markdown(f"**Total records:** {len(df):,}")

        # Separate columns by type for better organization
        categorical_cols = [(k, v) for k, v in column_info.items() if v['type'] == 'categorical']
        numeric_cols = [(k, v) for k, v in column_info.items() if v['type'] == 'numeric']
        text_cols = [(k, v) for k, v in column_info.items() if v['type'] == 'text']

        # Sort categorical columns by number of unique values (fewer first)
        categorical_cols.sort(key=lambda x: len(x[1].get('unique_values', []) or []))

        # Let user select which columns to filter on
        all_filterable = [col for col, _ in categorical_cols + numeric_cols + text_cols]

        selected_columns = st.multiselect(
            "Select columns to filter on",
            options=all_filterable,
            default=st.session_state.get("selected_filter_columns", []),
            help="Choose columns for filtering. Options cascade based on selections (AND logic).",
            key="filter_column_selector"
        )
        st.session_state.selected_filter_columns = selected_columns

        if not selected_columns:
            st.caption("Select columns above to create filters")

            # Show column summary with consistent string types to avoid Arrow errors
            with st.expander("📊 Available columns", expanded=False):
                col_summary = []
                for col, info in column_info.items():
                    unique_count = len(info['unique_values']) if info['unique_values'] else -1
                    col_summary.append({
                        "Column": col,
                        "Type": info['type'],
                        "Unique": str(unique_count) if unique_count >= 0 else "many",
                        "Null %": f"{info['null_percentage']:.1f}%"
                    })
                st.dataframe(pd.DataFrame(col_summary), hide_index=True, width="stretch")

            return {}

        st.markdown("---")
        st.markdown("**🎯 Cascading Filters** *(AND logic - options update based on selections)*")

        # Initialize pending filters from session state
        pending_filters = st.session_state.get("pending_filters", {})

        # Render filters for selected columns (max 4 per row)
        cols_per_row = 4
        for row_start in range(0, len(selected_columns), cols_per_row):
            row_cols = selected_columns[row_start:row_start + cols_per_row]
            cols = st.columns(len(row_cols))

            for i, col_name in enumerate(row_cols):
                info = column_info.get(col_name, {})
                col_type = info.get('type', 'text')

                with cols[i]:
                    st.markdown(f"**{col_name}**")

                    if col_type == 'categorical':
                        # Get cascading options based on other filters
                        available_options = get_cascading_options(
                            table, col_name, pending_filters, column_info
                        )

                        # Get current selection, filter to only valid options
                        current_selection = pending_filters.get(col_name, [])
                        if isinstance(current_selection, list):
                            current_selection = [v for v in current_selection if v in available_options]

                        selected_values = st.multiselect(
                            f"Select values",
                            options=available_options,
                            default=current_selection,
                            key=f"filter_{col_name}",
                            help=f"{len(available_options)} options available"
                        )

                        # Update pending filters
                        if selected_values:
                            pending_filters[col_name] = selected_values
                        elif col_name in pending_filters:
                            del pending_filters[col_name]

                    elif col_type == 'numeric':
                        # For numeric, apply other filters first to get valid range
                        other_filters = {k: v for k, v in pending_filters.items() if k != col_name and v}
                        if other_filters:
                            filtered_table = apply_filters_arrow(table, other_filters)
                            filtered_df = filtered_table.to_pandas()
                        else:
                            filtered_df = df

                        col_data = filtered_df[col_name].dropna()
                        if len(col_data) > 0:
                            min_val, max_val = float(col_data.min()), float(col_data.max())
                            if min_val != max_val:
                                # Get current range or use full range
                                current_range = pending_filters.get(col_name, {})
                                default_min = current_range.get('min', min_val) if isinstance(current_range, dict) else min_val
                                default_max = current_range.get('max', max_val) if isinstance(current_range, dict) else max_val

                                # Clamp to available range
                                default_min = max(min_val, min(default_min, max_val))
                                default_max = min(max_val, max(default_max, min_val))

                                range_values = st.slider(
                                    f"Range",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=(default_min, default_max),
                                    key=f"filter_{col_name}"
                                )
                                if range_values != (min_val, max_val):
                                    pending_filters[col_name] = {'min': range_values[0], 'max': range_values[1]}
                                elif col_name in pending_filters:
                                    del pending_filters[col_name]

                    elif col_type == 'text':
                        current_text = pending_filters.get(col_name, "")
                        if not isinstance(current_text, str):
                            current_text = ""

                        search_text = st.text_input(
                            f"Search",
                            value=current_text,
                            key=f"filter_{col_name}",
                            help="Case-insensitive contains search"
                        )
                        if search_text.strip():
                            pending_filters[col_name] = search_text.strip()
                        elif col_name in pending_filters:
                            del pending_filters[col_name]

        # Store pending filters
        st.session_state.pending_filters = pending_filters

        st.markdown("---")

        # Show preview of filtered count
        if pending_filters:
            try:
                preview_table = apply_filters_arrow(table, pending_filters)
                preview_count = len(preview_table)
                st.info(f"📊 Preview: **{preview_count:,}** records match current filters")
            except Exception:
                pass

        # Apply filters button
        col1, col2 = st.columns([1, 1])
        with col1:
            apply_button = st.button("Apply Filters", type="primary")
        with col2:
            clear_button = st.button("Clear All")

        if clear_button:
            st.session_state.filtered_df = df
            st.session_state.active_filters = {}
            st.session_state.pending_filters = {}
            st.session_state.selected_filter_columns = []
            st.rerun()

        if apply_button:
            if pending_filters:
                with st.spinner("Applying filters..."):
                    logger.info(f"Applying filters: {list(pending_filters.keys())}")
                    filtered_table = apply_filters_arrow(table, pending_filters)
                    filtered_df = filtered_table.to_pandas()

                    logger.info(f"Filter result: {len(df):,} -> {len(filtered_df):,} records "
                                f"({len(filtered_df)/len(df)*100:.1f}% retained)")

                    st.session_state.filtered_df = filtered_df
                    st.session_state.active_filters = pending_filters.copy()

                    # Reset downstream state
                    st.session_state.embeddings = None
                    st.session_state.data = None
                    st.session_state.labels = None
                    st.session_state.selected_image_idx = None

                    st.success(f"Filtered to {len(filtered_df):,} records")
            else:
                st.session_state.filtered_df = df
                st.session_state.active_filters = {}
                st.info("No filters applied, using full dataset")

        # Show active filter summary
        active_filters = st.session_state.get("active_filters", {})
        if active_filters:
            with st.expander("📋 Applied filters", expanded=False):
                for col, val in active_filters.items():
                    if isinstance(val, list):
                        st.caption(f"• **{col}**: {', '.join(str(v) for v in val[:3])}{'...' if len(val) > 3 else ''}")
                    elif isinstance(val, dict):
                        st.caption(f"• **{col}**: {val['min']:.2f} to {val['max']:.2f}")
                    else:
                        st.caption(f"• **{col}**: contains '{val}'")

        return pending_filters


def apply_filters_arrow(table: pa.Table, filters: Dict[str, Any]) -> pa.Table:
    """
    Apply filters to PyArrow Table with AND logic.

    Args:
        table: PyArrow Table to filter
        filters: Dictionary of column_name -> filter_value pairs

    Returns:
        Filtered PyArrow Table
    """
    filter_expressions = []

    for col, filter_value in filters.items():
        if col not in table.column_names or filter_value is None:
            continue

        col_ref = pc.field(col)

        if isinstance(filter_value, dict):
            # Numeric range filter
            if 'min' in filter_value and filter_value['min'] is not None:
                filter_expressions.append(pc.greater_equal(col_ref, filter_value['min']))
            if 'max' in filter_value and filter_value['max'] is not None:
                filter_expressions.append(pc.less_equal(col_ref, filter_value['max']))
        elif isinstance(filter_value, list):
            # Categorical filter (multiple values)
            if len(filter_value) > 0:
                filter_expressions.append(pc.is_in(col_ref, pa.array(filter_value)))
        elif isinstance(filter_value, str):
            # Text filter (case-insensitive contains)
            if filter_value.strip():
                pattern = f".*{filter_value.lower()}.*"
                filter_expressions.append(
                    pc.match_substring_regex(pc.utf8_lower(col_ref), pattern)
                )

    # Combine all filters with AND
    if filter_expressions:
        from functools import reduce
        try:
            combined = reduce(lambda a, b: pc.and_kleene(a, b), filter_expressions)
            return table.filter(combined)
        except AttributeError:
            # Fallback for older PyArrow
            result = table
            for expr in filter_expressions:
                result = result.filter(expr)
            return result

    return table


def extract_embeddings_safe(df: pd.DataFrame) -> np.ndarray:
    """
    Safely extract embeddings from DataFrame using zero-copy where possible.

    Args:
        df: DataFrame with 'emb' column

    Returns:
        numpy array of embeddings
    """
    if 'emb' not in df.columns:
        raise ValueError("DataFrame does not contain 'emb' column")

    # Use np.stack for efficient conversion
    embeddings = np.stack(df['emb'].values)

    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings should be 2D, got shape {embeddings.shape}")

    return embeddings.astype(np.float32)


def render_clustering_section() -> Tuple[bool, int, str, str, str, int, Optional[int]]:
    """
    Render the clustering section with VRAM error handling.

    Returns:
        Tuple of (cluster_button_clicked, n_clusters, reduction_method,
                  dim_reduction_backend, clustering_backend, n_workers, seed)
    """
    with st.expander("🎯 Cluster Embeddings", expanded=False):
        filtered_df = st.session_state.get("filtered_df", None)

        if filtered_df is None or len(filtered_df) == 0:
            st.info("Apply filters first to enable clustering.")
            return False, 5, "TSNE", "auto", "auto", 8, None

        st.markdown(f"**Ready to cluster:** {len(filtered_df):,} records")

        # Estimate memory requirements
        emb_dim = len(filtered_df['emb'].iloc[0])
        n_samples = len(filtered_df)
        est_memory_mb = (n_samples * emb_dim * 4) / (1024 * 1024)  # float32

        if est_memory_mb > 1000:
            st.warning(f"⚠️ Large dataset: ~{est_memory_mb:.0f} MB for embeddings. Consider filtering further if GPU memory is limited.")

        # Cluster count options
        cluster_method = st.radio(
            "Cluster count method:",
            ["Specify number", "Use column values"],
            horizontal=True
        )

        if cluster_method == "Specify number":
            n_clusters = st.slider("Number of clusters", 2, min(100, len(filtered_df)//2), 5)
            cluster_column = None
        else:
            # Get categorical columns for clustering
            column_info = st.session_state.get("column_info", {})
            categorical_cols = [k for k, v in column_info.items() if v['type'] == 'categorical']

            if categorical_cols:
                cluster_column = st.selectbox(
                    "Use unique values from column:",
                    categorical_cols,
                    help="Number of clusters = unique values in selected column"
                )
                if cluster_column in filtered_df.columns:
                    n_clusters = filtered_df[cluster_column].nunique()
                    st.info(f"Using **{n_clusters}** clusters from {cluster_column}")
                else:
                    n_clusters = 5
            else:
                st.warning("No categorical columns available")
                n_clusters = 5
                cluster_column = None

        reduction_method = st.selectbox(
            "Dimensionality Reduction",
            ["TSNE", "PCA", "UMAP"],
            help="For 2D visualization only. Clustering uses full embeddings."
        )

        # Backend controls
        dim_reduction_backend, clustering_backend, n_workers, seed = render_clustering_backend_controls()

        cluster_button = st.button("Run Clustering", type="primary")

        if cluster_button:
            run_clustering_with_error_handling(
                filtered_df, n_clusters, reduction_method,
                dim_reduction_backend, clustering_backend, n_workers, seed,
                cluster_column if cluster_method == "Use column values" else None
            )

        return cluster_button, n_clusters, reduction_method, dim_reduction_backend, clustering_backend, n_workers, seed


def run_clustering_with_error_handling(
    filtered_df: pd.DataFrame,
    n_clusters: int,
    reduction_method: str,
    dim_reduction_backend: str,
    clustering_backend: str,
    n_workers: int,
    seed: Optional[int],
    cluster_column: Optional[str] = None
):
    """
    Run clustering with comprehensive error handling for VRAM and CUDA issues.
    """
    try:
        # Check CUDA availability
        cuda_available, device_info = check_cuda_available()

        # Resolve auto backends
        actual_dim_backend = resolve_backend(dim_reduction_backend, "reduction")
        actual_cluster_backend = resolve_backend(clustering_backend, "clustering")

        # Log clustering start
        logger.info("=" * 60)
        logger.info("CLUSTERING START")
        logger.info("=" * 60)
        logger.info(f"Device: {device_info} (CUDA: {'Yes' if cuda_available else 'No'})")
        logger.info(f"Dim Reduction Backend: {actual_dim_backend} (requested: {dim_reduction_backend})")
        logger.info(f"Clustering Backend: {actual_cluster_backend} (requested: {clustering_backend})")

        # Extract embeddings
        t_start = time.time()
        with st.spinner("Extracting embeddings..."):
            embeddings = extract_embeddings_safe(filtered_df)
            st.session_state.embeddings = embeddings
        t_extract = time.time() - t_start

        n_samples, emb_dim = embeddings.shape
        mem_mb = (n_samples * emb_dim * 4) / (1024 * 1024)

        logger.info(f"Records: {n_samples:,} | Embedding dim: {emb_dim}")
        logger.info(f"Memory: ~{mem_mb:.1f} MB | Clusters: {n_clusters}")
        logger.info(f"Embeddings extracted ({t_extract:.2f}s)")

        # Run clustering with automatic GPU fallback
        t_cluster_start = time.time()
        with st.spinner(f"Running {reduction_method} + KMeans..."):
            df_plot, labels = ClusteringService.run_clustering_safe(
                embeddings,
                filtered_df['uuid'].tolist(),
                n_clusters,
                reduction_method,
                n_workers,
                actual_dim_backend,
                actual_cluster_backend,
                seed
            )

        t_cluster = time.time() - t_cluster_start
        t_total = time.time() - t_start

        # Log clustering completion to console
        logger.info(f"{reduction_method} + KMeans completed ({t_cluster:.2f}s)")
        logger.info(f"Total time: {t_total:.2f}s")

        # Create enhanced plot dataframe
        df_plot = create_cluster_dataframe(filtered_df.reset_index(drop=True), df_plot[['x', 'y']].values, labels)

        # Handle column-based cluster names
        if cluster_column and cluster_column in filtered_df.columns:
            filtered_reset = filtered_df.reset_index(drop=True)
            unique_taxa = sorted(filtered_df[cluster_column].dropna().unique())
            taxon_to_id = {taxon: str(i) for i, taxon in enumerate(unique_taxa)}

            taxonomic_names = []
            numeric_clusters = []

            for idx in range(len(df_plot)):
                taxon_value = filtered_reset.iloc[idx][cluster_column]
                if pd.notna(taxon_value) and taxon_value in taxon_to_id:
                    taxonomic_names.append(str(taxon_value))
                    numeric_clusters.append(taxon_to_id[taxon_value])
                else:
                    taxonomic_names.append("Unknown")
                    numeric_clusters.append(str(len(unique_taxa)))

            df_plot['cluster'] = numeric_clusters
            df_plot['cluster_name'] = taxonomic_names
            st.session_state.taxonomic_clustering = {'is_taxonomic': True, 'column': cluster_column}
        else:
            df_plot['cluster_name'] = df_plot['cluster'].copy()
            st.session_state.taxonomic_clustering = {'is_taxonomic': False}

        # Store results with data version tracking
        data_hash = hashlib.md5(f"{len(df_plot)}_{n_clusters}_{reduction_method}".encode()).hexdigest()[:8]

        st.session_state.data = df_plot
        st.session_state.labels = labels
        st.session_state.data_version = data_hash  # Track data version for selection validation
        st.session_state.selected_image_idx = None  # User must click to select (not auto-select)
        st.session_state.filtered_df_for_clustering = filtered_df.reset_index(drop=True)

        # Final log with success
        logger.info(f"Clustering complete: {n_clusters} clusters found")
        logger.info("=" * 60)

        st.success(f"Clustering complete! {n_clusters} clusters found.")

    except (RuntimeError, OSError) as e:
        if is_oom_error(e):
            st.error("**GPU Out of Memory**")
            st.info("Try: Reduce dataset size with more filters, use 'sklearn' backend, or use PCA")
            logger.exception("GPU OOM error during clustering")
        else:
            st.error(f"Error during clustering: {e}")
            logger.exception("Clustering error")

    except MemoryError:
        st.error("**System Out of Memory** - Reduce dataset size")
        logger.exception("System memory exhausted during clustering")

    except Exception as e:
        st.error(f"Error: {e}")
        logger.exception("Unexpected clustering error")


def create_cluster_dataframe(df: pd.DataFrame, embeddings_2d: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Create a dataframe for clustering visualization."""
    df_plot = pd.DataFrame({
        "x": embeddings_2d[:, 0],
        "y": embeddings_2d[:, 1],
        "cluster": labels.astype(str),
        "uuid": df['uuid'].values,
        "idx": range(len(df))
    })

    # Add available metadata columns for tooltips
    for col in df.columns:
        if col not in ['uuid', 'emb', 'embedding', 'embeddings'] and col not in df_plot.columns:
            df_plot[col] = df[col].values

    return df_plot
