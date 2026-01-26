"""
Clustering summary components for the embed_explore application.
"""

import streamlit as st
import os
import pandas as pd
from shared.services.clustering_service import ClusteringService


def render_clustering_summary():
    """Render the clustering summary panel with statistics and representative images."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    embeddings = st.session_state.get("embeddings", None)

    if df_plot is not None and labels is not None and embeddings is not None:
        # Check if this is image data
        has_images = 'image_path' in df_plot.columns

        if has_images:
            # For image data, show the full clustering summary
            st.subheader("Clustering Summary")

            try:
                summary_df, representatives = ClusteringService.generate_clustering_summary(
                    embeddings, labels, df_plot
                )

                st.dataframe(summary_df, hide_index=True, width='stretch')

                st.markdown("#### Representative Images")
                for row in summary_df.itertuples():
                    k = row.Cluster
                    st.markdown(f"**Cluster {k}**")
                    img_cols = st.columns(3)
                    for i, img_idx in enumerate(representatives[k]):
                        img_path = df_plot.iloc[img_idx]["image_path"]
                        img_cols[i].image(
                            img_path,
                            width='stretch',
                            caption=os.path.basename(img_path)
                        )

            except Exception as e:
                st.error(f"Error generating clustering summary: {e}")
        else:
            st.info("No image data available for summary visualization.")

    else:
        st.info("Clustering summary will appear here after clustering.")
