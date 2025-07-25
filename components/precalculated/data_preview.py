"""
Data preview components for the precalculated embeddings page.
"""

import streamlit as st
import pandas as pd
import requests
from typing import Optional
from PIL import Image
from io import BytesIO


def fetch_image_from_url(url: str, timeout: int = 5) -> Optional[Image.Image]:
    """
    Try to fetch an image from a URL.
    
    Args:
        url: The image URL
        timeout: Request timeout in seconds
        
    Returns:
        PIL Image object if successful, None otherwise
    """
    if not url or not isinstance(url, str):
        return None
        
    try:
        # Add common image URL patterns if needed
        if not url.startswith(('http://', 'https://')):
            return None
            
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Check if content type is an image
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            return None
            
        # Try to open as image
        image = Image.open(BytesIO(response.content))
        return image
        
    except Exception:
        return None


def render_data_preview():
    """Render the data preview panel (replaces image preview)."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    selected_idx = st.session_state.get("selected_image_idx", 0)
    filtered_df = st.session_state.get("filtered_df_for_clustering", None)
    
    if (
        df_plot is not None and
        labels is not None and
        selected_idx is not None and
        0 <= selected_idx < len(df_plot) and
        filtered_df is not None
    ):
        # Get the selected record
        selected_idx = st.session_state.get("selected_image_idx", 0)
        selected_uuid = df_plot.iloc[selected_idx]['uuid']
        cluster = labels[selected_idx] if labels is not None else "?"
        
        # Use cluster_name if available (for taxonomic clustering)
        if 'cluster_name' in df_plot.columns:
            cluster_display = df_plot.iloc[selected_idx]['cluster_name']
        else:
            cluster_display = cluster
        
        # Find the full record in the original filtered dataframe
        record = filtered_df[filtered_df['uuid'] == selected_uuid].iloc[0]
        
        st.markdown(f"### 📋 Record Details")
        
        # Create tabs for different types of information
        tab_overview, tab_details = st.tabs(["🔍 Overview", "📊 Details"])
        
        with tab_overview:
            # Basic information
            st.markdown(f"**Cluster:** `{cluster_display}`")
            st.markdown(f"**UUID:** `{selected_uuid}`")
            
            # Try to fetch and display image if identifier exists
            if 'identifier' in record.index and pd.notna(record['identifier']):
                identifier_url = record['identifier']
                st.markdown("**Image:**")
                
                with st.spinner("Fetching image..."):
                    image = fetch_image_from_url(identifier_url)
                    
                if image is not None:
                    st.image(image, caption=f"Image from: {identifier_url}", use_container_width=True)
                else:
                    st.info(f"Could not fetch image from: {identifier_url}")
                    with st.expander("🔗 Image URL"):
                        st.code(identifier_url)
        
        with tab_details:
            # Taxonomy section
            st.markdown("#### 🧬 Taxonomy")
            
            # Show scientific and common names first
            id_fields = ['scientific_name', 'common_name']
            for field in id_fields:
                if field in record.index and pd.notna(record[field]):
                    st.markdown(f"**{field.replace('_', ' ').title()}:** {record[field]}")
            
            # Show taxonomic hierarchy
            taxonomic_fields = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
            hierarchy_parts = []
            for field in taxonomic_fields:
                if field in record.index and pd.notna(record[field]):
                    hierarchy_parts.append(f"{field.title()}: {record[field]}")
            
            if hierarchy_parts:
                st.markdown("**Taxonomic Hierarchy:**")
                hierarchy_text = "\n".join([f"• {part}" for part in hierarchy_parts])
                st.code(hierarchy_text, language="text")
            
            # Display source information
            st.markdown("#### 📊 Source Information")
            source_fields = ['source_dataset', 'publisher', 'basisOfRecord', 'img_type']
            for field in source_fields:
                if field in record.index and pd.notna(record[field]):
                    value = record[field]
                    if len(str(value)) > 50:  # Truncate long values
                        value = str(value)[:47] + "..."
                    st.markdown(f"**{field.replace('_', ' ').title()}:** {value}")
            
            # Display additional metadata in an expander
            with st.expander("🔍 All Metadata"):
                # Create a clean dataframe for display
                display_data = []
                for field, value in record.items():
                    if field not in ['uuid', 'emb']:  # Skip technical fields
                        display_data.append({
                            'Field': field.replace('_', ' ').title(),
                            'Value': value if pd.notna(value) else 'null'
                        })
                
                if display_data:
                    metadata_df = pd.DataFrame(display_data)
                    st.dataframe(metadata_df, hide_index=True, use_container_width=True)
    
    else:
        st.info("📋 Record details will appear here after you select a point in the cluster plot.")
        
        # Show dataset summary if we have filtered data
        filtered_df = st.session_state.get("filtered_df", None)
        if filtered_df is not None and len(filtered_df) > 0:
            st.markdown("### 📈 Dataset Summary")
            st.markdown(f"**Total records:** {len(filtered_df):,}")
            
            # Show distribution of key fields
            summary_fields = ['kingdom', 'family', 'source_dataset', 'img_type']
            for field in summary_fields:
                if field in filtered_df.columns:
                    non_null_count = filtered_df[field].notna().sum()
                    unique_count = filtered_df[field].nunique()
                    st.markdown(f"**{field.replace('_', ' ').title()}:** {unique_count} unique values ({non_null_count:,} non-null)")


def render_cluster_statistics():
    """Render cluster-level statistics."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    filtered_df = st.session_state.get("filtered_df_for_clustering", None)
    
    if df_plot is not None and labels is not None and filtered_df is not None:
        st.markdown("### 📊 Cluster Statistics")
        
        # Create cluster summary
        cluster_summary = []
        
        # Check if we have taxonomic clustering with cluster names
        if 'cluster_name' in df_plot.columns:
            # Use cluster names for display, but group by cluster ID for consistency
            unique_cluster_ids = sorted(df_plot['cluster'].unique(), key=lambda x: int(x))
            
            for cluster_id in unique_cluster_ids:
                cluster_mask = df_plot['cluster'] == cluster_id
                cluster_size = cluster_mask.sum()
                cluster_percentage = (cluster_size / len(df_plot)) * 100
                
                # Get the cluster name for this cluster ID
                cluster_name = df_plot[cluster_mask]['cluster_name'].iloc[0] if cluster_size > 0 else str(cluster_id)
                
                cluster_summary.append({
                    'Cluster': cluster_name,
                    'Size': cluster_size,
                    'Percentage': f"{cluster_percentage:.1f}%"
                })
        else:
            # Standard numeric clustering
            for cluster_id in sorted(df_plot['cluster'].unique(), key=int):
                cluster_mask = df_plot['cluster'] == cluster_id
                cluster_size = cluster_mask.sum()
                cluster_percentage = (cluster_size / len(df_plot)) * 100
                
                cluster_summary.append({
                    'Cluster': int(cluster_id),
                    'Size': cluster_size,
                    'Percentage': f"{cluster_percentage:.1f}%"
                })
        
        summary_df = pd.DataFrame(cluster_summary)
        st.dataframe(summary_df, hide_index=True, use_container_width=True)
