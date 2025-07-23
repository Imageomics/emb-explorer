"""
Clustering summary components.
"""

import streamlit as st
import os
from server.clustering_service import ClusteringService


def render_clustering_summary():
    """Render the clustering summary panel."""
    df_plot = st.session_state.get("data", None)
    labels = st.session_state.get("labels", None)
    embeddings = st.session_state.get("embeddings", None)

    if df_plot is not None and labels is not None and embeddings is not None:
        st.subheader("Clustering Summary")

        try:
            summary_df, representatives = ClusteringService.generate_clustering_summary(
                embeddings, labels, df_plot
            )
            
            st.dataframe(summary_df, hide_index=True, use_container_width=True)

            st.markdown("#### Representative Images")
            for row in summary_df.itertuples():
                k = row.Cluster
                st.markdown(f"**Cluster {k}**")
                img_cols = st.columns(3)
                for i, img_idx in enumerate(representatives[k]):
                    img_path = df_plot.iloc[img_idx]["image_path"]
                    img_cols[i].image(
                        img_path, 
                        use_container_width=True, 
                        caption=os.path.basename(img_path)
                    )
                    
        except Exception as e:
            st.error(f"Error generating clustering summary: {e}")
    else:
        st.info("Clustering summary table will appear here after clustering.")
