
# import streamlit as st
# import pandas as pd
# from src.preprocessing import load_papers, DatasetError
# from src.embeddings import embed_texts
# from src.similarity import rank_similar
# from src.clustering import cluster_papers
# from src.reading_priority import reading_priority
# from src.saturation import saturation_map
# from src.student_brief import student_brief

# # --- Premium AI Dark Theme ---
# st.set_page_config(page_title="ResearchLens", page_icon="🔎", layout="wide")

# st.markdown("""
# <style>
#     /* Deep Dark AI Background */
#     .stApp {
#         background: #0E1117;
#         color: #E2E8F0;
#     }
#     [data-testid="stSidebar"] {
#         background-color: #0E1117;
#         border-right: 1px solid #2D3748;
#     }
    
#     /* Typography */
#     h1, h2, h3 {
#         color: #F8FAFC !important;
#         font-family: 'Inter', sans-serif;
#         font-weight: 700;
#     }
    
#     /* Hero Banner (Home) */
#     .hero-banner {
#         background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
#         padding: 3rem 2rem;
#         border-radius: 16px;
#         color: white;
#         text-align: center;
#         box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
#         margin-bottom: 2rem;
#     }
#     .hero-banner h1 {
#         color: white !important;
#         font-size: 3.5rem;
#         margin-bottom: 0.5rem;
#     }
#     .hero-banner p {
#         font-size: 1.25rem;
#         opacity: 0.9;
#     }
    
#     /* Glassmorphism Cards (Home) */
#     .feature-card {
#         background: #1A202C;
#         padding: 1.5rem;
#         border-radius: 12px;
#         box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
#         border: 1px solid #2D3748;
#         border-top: 4px solid #3b82f6;
#         height: 100%;
#         transition: transform 0.2s;
#     }
#     .feature-card:hover {
#         transform: translateY(-5px);
#         border-color: #60A5FA;
#     }
#     .feature-card h3 {
#         font-size: 1.5rem;
#         margin-bottom: 0.5rem;
#         color: #60A5FA !important;
#     }
#     .feature-card p {
#         font-size: 1.1rem;
#         color: #A0AEC0;
#         line-height: 1.6;
#     }
    
#     /* Formatting for Analyze Page Inputs */
#     .stTextInput>div>div>input {
#         font-size: 1.1rem;
#         padding: 10px;
#         background-color: #2D3748;
#         color: white;
#         border-radius: 8px;
#     }
#     .stDataFrame {
#         font-size: 1.05rem;
#     }
    
#     /* Make standard text bigger */
#     p, label, .stMarkdown {
#         font-size: 1.1rem !important;
#         color: #E2E8F0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # --- Sidebar ---
# st.sidebar.markdown("<h2 style='color: #60A5FA;'>🔎 ResearchLens</h2>", unsafe_allow_html=True)
# st.sidebar.markdown("---")
# page = st.sidebar.radio("Navigation", ["Home", "Analyze Corpus"])

# # --- HOME PAGE ---
# if page == "Home":
#     st.markdown("""
#         <div class="hero-banner">
#             <h1>ResearchLens</h1>
#             <p>An Evidence-Aware Deep Learning Framework for Scientific Literature Analysis</p>
#         </div>
#     """, unsafe_allow_html=True)
    
#     st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Stop reading randomly. Start reading strategically.</h2>", unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.markdown("""
#             <div class="feature-card">
#                 <h3>📌 Reading Priority</h3>
#                 <p>Identifies the most central papers in a cluster using custom Transformer embeddings. Never waste time reading irrelevant papers again.</p>
#             </div>
#         """, unsafe_allow_html=True)
        
#     with col2:
#         st.markdown("""
#             <div class="feature-card">
#                 <h3>✅ Strong Areas</h3>
#                 <p>Maps the saturation of your corpus. Know which topics are already heavily researched before you start writing your literature review.</p>
#             </div>
#         """, unsafe_allow_html=True)
        
#     with col3:
#         st.markdown("""
#             <div class="feature-card">
#                 <h3>🕳️ Silent Gaps</h3>
#                 <p>Finds keywords that appear in only 1 paper. These are your potential research gaps that you can focus your project on.</p>
#             </div>
#         """, unsafe_allow_html=True)
        
#     st.write("")
#     st.write("")
#     st.info("👈 **Click on 'Analyze Corpus' in the sidebar to get started.** Designed for final-year UG and PG students.")

# # --- ANALYZE PAGE (Wide, full-screen formatting) ---
# else:
#     st.title("📊 Analyze Corpus")
#     st.markdown("<p style='color: #A0AEC0;'>Upload your paper metadata CSV and enter a topic to run the deep learning pipeline.</p>", unsafe_allow_html=True)
#     st.divider()
    
#     # Full width input fields
#     topic = st.text_input("Research Topic", placeholder="e.g., plant disease detection")
#     upload = st.file_uploader("Upload Paper Metadata (CSV)", type=["csv"])
    
#     if upload is None:
#         st.info("Please upload a CSV to begin. You can use the sample at `data/sample_papers.csv`.")
#         st.stop()
        
#     try: 
#         frame = load_papers(upload)
#     except DatasetError as exc: 
#         st.error(str(exc))
#         st.stop()
        
#     st.success(f"Successfully loaded {len(frame)} records.")
    
#     st.dataframe(frame[["paper_id", "title", "year", "keywords"]], use_container_width=True)
    
#     if not topic.strip(): 
#         st.warning("⚠️ Please enter a research topic to begin the analysis.")
#         st.stop()
        
#     analyzable = frame[frame["abstract"].str.strip().ne("")].reset_index(drop=True)
    
#     with st.spinner("🧠 Encoding abstracts with custom Transformer model..."):
#         vectors = embed_texts(analyzable["analysis_text"].tolist())
#         query_vector = embed_texts([topic])[0]
        
#     labels, _, metrics = cluster_papers(vectors, analyzable["analysis_text"].tolist())
    
#     st.divider()
    
#     # Full width results
#     st.subheader("🎯 Most Similar Papers")
#     for idx, score in rank_similar(vectors, query_vector, limit=min(5, len(vectors))):
#         st.markdown(f"**[{analyzable.iloc[idx]['paper_id']}] {analyzable.iloc[idx]['title']}** &nbsp;—&nbsp; Similarity Score: `{score:.3f}`")
        
#     st.divider()
    
#     st.subheader("📌 Read These First")
#     for i, r in enumerate(reading_priority(analyzable, vectors, labels)[:5], 1):
#         st.markdown(f"**{i}. [{r['paper_id']}] {r['title']}** &nbsp;—&nbsp; Centrality: `{r['centrality']}`")
        
#     st.divider()
    
#     st.subheader("📄 One-Screen Brief")
#     brief = student_brief(analyzable, labels, vectors, topic)
#     st.code(brief, language="text")

import streamlit as st
import pandas as pd
from src.preprocessing import load_papers, DatasetError
from src.embeddings import embed_texts
from src.similarity import rank_similar
from src.clustering import cluster_papers
from src.reading_priority import reading_priority
from src.saturation import saturation_map
from src.student_brief import student_brief

# --- Premium AI Dark Theme ---
st.set_page_config(page_title="ResearchLens", page_icon="🔎", layout="wide")

st.markdown("""
<style>
    /* Deep Dark AI Background */
    .stApp {
        background: #0E1117;
        color: #E2E8F0;
    }
    
    /* Hide the ugly white header bar */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* Sidebar Background & Text */
    [data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid #2D3748;
    }
    [data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    
    /* Hero Banner (Home) */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
        margin-bottom: 2rem;
    }
    .hero-banner h1 {
        color: white !important;
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    .hero-banner p {
        font-size: 1.25rem;
        opacity: 0.9;
    }
    
    /* Glassmorphism Cards (Home) */
    .feature-card {
        background: #1A202C;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        border: 1px solid #2D3748;
        border-top: 4px solid #3b82f6;
        height: 100%;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: #60A5FA;
    }
    .feature-card h3 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #60A5FA !important;
    }
    .feature-card p {
        font-size: 1.1rem;
        color: #A0AEC0;
        line-height: 1.6;
    }
    
    /* Formatting for Analyze Page Inputs */
    .stTextInput>div>div>input {
        font-size: 1.1rem;
        padding: 10px;
        background-color: #2D3748;
        color: white;
        border-radius: 8px;
    }
    
    /* Make standard text bigger */
    p, label, .stMarkdown {
        font-size: 1.1rem !important;
        color: #E2E8F0;
    }

    /* --- FIX THE BRIGHT WHITE BOXES --- */
    
    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {
        background-color: #1A202C !important; /* Dark Grey */
        border: 2px dashed #4A5568 !important;
    }
    [data-testid="stFileUploadDropzone"] * {
        color: #E2E8F0 !important; /* White text */
    }
    
    /* Success Message */
    .stSuccess {
        background-color: #1C4532 !important; /* Dark Green */
        color: #9AE6B4 !important;
        border: 1px solid #276749 !important;
    }
    
    /* One-Screen Brief Code Block */
    .stCodeBlock, pre {
        background-color: #1A202C !important; /* Dark Grey */
        border: 1px solid #2D3748 !important;
        border-radius: 8px;
    }
    .stCodeBlock code, pre code {
        color: #A0AEC0 !important; /* Soft white text */
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("<h2 style='color: #60A5FA;'>🔎 ResearchLens</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Home", "Analyze Corpus"])

# --- HOME PAGE ---
if page == "Home":
    st.markdown("""
        <div class="hero-banner">
            <h1>ResearchLens</h1>
            <p>An Evidence-Aware Deep Learning Framework for Scientific Literature Analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h2 style='text-align: center; margin-bottom: 2rem;'>Stop reading randomly. Start reading strategically.</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3>📌 Reading Priority</h3>
                <p>Identifies the most central papers in a cluster using custom Transformer embeddings. Never waste time reading irrelevant papers again.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3>✅ Strong Areas</h3>
                <p>Maps the saturation of your corpus. Know which topics are already heavily researched before you start writing your literature review.</p>
            </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3>🕳️ Silent Gaps</h3>
                <p>Finds keywords that appear in only 1 paper. These are your potential research gaps that you can focus your project on.</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    st.write("")
    st.info("👈 **Click on 'Analyze Corpus' in the sidebar to get started.** Designed for final-year UG and PG students.")

# --- ANALYZE PAGE ---
else:
    st.title("📊 Analyze Corpus")
    st.markdown("<p style='color: #A0AEC0;'>Upload your paper metadata CSV and enter a topic to run the deep learning pipeline.</p>", unsafe_allow_html=True)
    st.divider()
    
    topic = st.text_input("Research Topic", placeholder="e.g., plant disease detection")
    upload = st.file_uploader("Upload Paper Metadata (CSV)", type=["csv"])
    
    if upload is None:
        st.info("Please upload a CSV to begin. You can use the sample at `data/sample_papers.csv`.")
        st.stop()
        
    try: 
        frame = load_papers(upload)
    except DatasetError as exc: 
        st.error(str(exc))
        st.stop()
        
    st.success(f"Successfully loaded {len(frame)} records.")
    
    st.dataframe(frame[["paper_id", "title", "year", "keywords"]], use_container_width=True)
    
    if not topic.strip(): 
        st.warning("⚠️ Please enter a research topic to begin the analysis.")
        st.stop()
        
    analyzable = frame[frame["abstract"].str.strip().ne("")].reset_index(drop=True)
    
    with st.spinner("🧠 Encoding abstracts with custom Transformer model..."):
        vectors = embed_texts(analyzable["analysis_text"].tolist())
        query_vector = embed_texts([topic])[0]
        
    labels, _, metrics = cluster_papers(vectors, analyzable["analysis_text"].tolist())
    
    st.divider()
    
    st.subheader("🎯 Most Similar Papers")
    for idx, score in rank_similar(vectors, query_vector, limit=min(5, len(vectors))):
        st.markdown(f"**[{analyzable.iloc[idx]['paper_id']}] {analyzable.iloc[idx]['title']}** &nbsp;—&nbsp; Similarity Score: `{score:.3f}`")
        
    st.divider()
    
    st.subheader("📌 Read These First")
    for i, r in enumerate(reading_priority(analyzable, vectors, labels)[:5], 1):
        st.markdown(f"**{i}. [{r['paper_id']}] {r['title']}** &nbsp;—&nbsp; Centrality: `{r['centrality']}`")
        
    st.divider()
    
    st.subheader("📄 One-Screen Brief")
    brief = student_brief(analyzable, labels, vectors, topic)
    st.code(brief, language="text")