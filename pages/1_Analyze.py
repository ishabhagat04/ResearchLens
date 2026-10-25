# import streamlit as st
# import plotly.express as px
# import pandas as pd
# import numpy as np
# from sklearn.decomposition import PCA
# from src.preprocessing import load_papers
# from src.embeddings import embed_texts
# from src.clustering import cluster_papers

# st.set_page_config(page_title="Clustering Visualizer", layout="wide")

# # Inject basic dark CSS for this page
# st.markdown("""
# <style>
#     .stApp { background: #0E1117; color: #E2E8F0; }
#     h1, h2, h3 { color: #60A5FA !important; }
# </style>
# """, unsafe_allow_html=True)

# st.title("🌌 Corpus Galaxy: Embedding Clusters")
# st.markdown("This visualizes how our from-scratch Transformer groups the papers. Papers closer together are semantically similar.")

# try:
#     frame = load_papers("C:/ResearchLens/data/sample_papers.csv")
# except Exception as e:
#     st.error(f"Could not load data: {e}")
#     st.stop()

# with st.spinner("Mapping the embedding space..."):
#     vectors = embed_texts(frame["analysis_text"].tolist())
#     labels, _, _ = cluster_papers(vectors, frame["analysis_text"].tolist())
    
#     # Reduce dimensions for visualization
#     pca = PCA(n_components=2)
#     coords = pca.fit_transform(vectors)
    
#     df = pd.DataFrame({
#         "x": coords[:, 0],
#         "y": coords[:, 1],
#         "Title": frame["title"],
#         "Cluster": [f"Cluster {l+1}" for l in labels]
#     })

#     fig = px.scatter(
#         df, x="x", y="y", color="Cluster", hover_data=["Title"],
#         title="2D PCA of Paper Embeddings",
#         template="plotly_dark",
#         color_discrete_sequence=px.colors.qualitative.Pastel
#     )
#     fig.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')))
#     fig.update_layout(
#         plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
#         title_font_size=20, legend_title_font_size=14
#     )
#     st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from src.preprocessing import load_papers
from src.embeddings import embed_texts
from src.clustering import cluster_papers

st.set_page_config(page_title="Clustering Visualizer", layout="wide")

# Full Dark Theme CSS
st.markdown("""
<style>
    .stApp { background: #0E1117; color: #E2E8F0; }
    header[data-testid="stHeader"] { background-color: transparent !important; }
    [data-testid="stSidebar"] { background-color: #0E1117; border-right: 1px solid #2D3748; }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    h1, h2, h3 { color: #60A5FA !important; }
    p { font-size: 1.1rem; color: #A0AEC0; }
</style>
""", unsafe_allow_html=True)

st.title("🌌 Corpus Galaxy: Embedding Clusters")
st.markdown("""
> **What are you looking at?** 
> Our Deep Learning model converts every paper into a mathematical coordinate (an embedding). This chart plots them on a 2D map. 
> **Papers that are close together** discuss similar topics. The different colors represent the **clusters** the AI found.
""")

try:
    frame = load_papers("C:/ResearchLens/data/sample_papers.csv")
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

with st.spinner("Mapping the embedding space..."):
    vectors = embed_texts(frame["analysis_text"].tolist())
    labels, _, _ = cluster_papers(vectors, frame["analysis_text"].tolist())
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    
    df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "Title": frame["title"],
        "Cluster": [f"Cluster {l+1}" for l in labels]
    })

    fig = px.scatter(
        df, x="x", y="y", color="Cluster", hover_data=["Title"],
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(marker=dict(size=15, line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        title_font_size=20, legend_title_font_size=14,
        xaxis_title="Semantic Dimension 1", yaxis_title="Semantic Dimension 2"
    )
    st.plotly_chart(fig, use_container_width=True)