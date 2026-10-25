
# import streamlit as st
# from src.preprocessing import load_papers
# from src.evidence_store import EvidenceIndex
# from src.agent_loop import answer_with_verification
# from src.llm_client import health

# st.set_page_config(page_title="Ask ResearchLens", layout="wide")

# # Full Dark Theme CSS
# st.markdown("""
# <style>
#     .stApp { background: #0E1117; color: #E2E8F0; }
#     header[data-testid="stHeader"] { background-color: transparent !important; }
#     [data-testid="stSidebar"] { background-color: #0E1117; border-right: 1px solid #2D3748; }
#     [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
#     h1, h2, h3 { color: #60A5FA !important; }
#     p { font-size: 1.1rem; color: #A0AEC0; }
#     .stTextInput>div>div>input { background-color: #2D3748; color: white; border: 1px solid #4A5568; }
#     .stSuccess { background-color: #1C4532 !important; color: #9AE6B4 !important; border: 1px solid #276749 !important; }
#     .stWarning { background-color: #742A2A !important; color: #FEB2B2 !important; border: 1px solid #9B2C2C !important; }
# </style>
# """, unsafe_allow_html=True)

# st.title("🤖 Agentic Q&A")
# st.markdown("""
# > **What is this?** 
# > This is our Advanced AI layer. It uses a local LLM to answer your questions. 
# > The **Retriever** finds relevant text, the **Generator** writes an answer, and the **Critic** verifies that the answer is actually grounded in your papers (so it doesn't hallucinate!).
# """)

# if not health():
#     st.error("⚠️ Ollama is not running. Please open a terminal and run `ollama serve` to use the AI.")
#     st.stop()

# @st.cache_resource
# def load_index():
#     frame = load_papers("C:/ResearchLens/data/sample_papers.csv")
#     return EvidenceIndex(frame)

# evidx = load_index()

# q = st.text_input("Your question:", placeholder="e.g., What limitations are reported for edge devices?")
# if not q.strip():
#     st.info("Type a question above to begin.")
#     st.stop()

# with st.spinner("🔍 Agent is retrieving evidence, generating, and verifying..."):
#     result = answer_with_verification(q, evidx)

# if result["status"] == "grounded":
#     st.success("✅ Answer verified against corpus evidence")
#     st.markdown(f"### Answer\n\n{result['final']}")
#     st.markdown(f"**Cited Papers:** `{', '.join(result['cited'])}`")
    
#     with st.expander("🔬 View Evidence & Critic Report"):
#         for c, m in result["report"]["evidence_map"].items():
#             st.markdown(f"**Claim:** {c}")
#             st.caption(f"↳ *Verified against:* [{m['paper_id']}] \"{m['evidence']}\"")
# else:
#     st.warning("⚠️ The Critic Agent rejected the generated answer because it could not be verified against the corpus.")

import streamlit as st
from src.preprocessing import load_papers
from src.evidence_store import EvidenceIndex
from src.agent_loop import answer_with_verification
from src.llm_client import health

st.set_page_config(page_title="Ask ResearchLens", layout="wide")

# Full Dark Theme CSS
st.markdown("""
<style>
    .stApp { background: #0E1117; color: #E2E8F0; }
    header[data-testid="stHeader"] { background-color: transparent !important; }
    [data-testid="stSidebar"] { background-color: #0E1117; border-right: 1px solid #2D3748; }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    h1, h2, h3 { color: #60A5FA !important; }
    p { font-size: 1.1rem; color: #A0AEC0; }
    .stTextInput>div>div>input { background-color: #2D3748; color: white; border: 1px solid #4A5568; }
    .stSuccess { background-color: #1C4532 !important; color: #9AE6B4 !important; border: 1px solid #276749 !important; }
    .stWarning { background-color: #742A2A !important; color: #FEB2B2 !important; border: 1px solid #9B2C2C !important; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Agentic Q&A")
st.markdown("""
> **What is this?** 
> This is our Advanced AI layer. It uses a local LLM to answer your questions. 
> The **Retriever** finds relevant text, the **Generator** writes an answer, and the **Critic** verifies that the answer is actually grounded in your papers (so it doesn't hallucinate!).
""")

if not health():
    st.error("⚠️ Ollama is not running. Please open a terminal and run `ollama serve` to use the AI.")
    st.stop()

@st.cache_resource
def load_index():
    frame = load_papers("C:/ResearchLens/data/sample_papers.csv")
    return EvidenceIndex(frame)

evidx = load_index()

q = st.text_input("Your question:", placeholder="e.g., What limitations are reported for edge devices?")
if not q.strip():
    st.info("Type a question above to begin.")
    st.stop()

with st.spinner("🔍 Agent is retrieving evidence, generating, and verifying..."):
    result = answer_with_verification(q, evidx)

if result["status"] == "grounded":
    st.success("✅ Answer verified against corpus evidence")
    st.markdown(f"### Answer\n\n{result['final']}")
    st.markdown(f"**Cited Papers:** `{', '.join(result['cited'])}`")
    
    with st.expander("🔬 View Evidence & Critic Report"):
        for c, m in result["report"]["evidence_map"].items():
            st.markdown(f"**Claim:** {c}")
            st.caption(f"↳ *Verified against:* [{m['paper_id']}] \"{m['evidence']}\"")
else:
    st.warning("⚠️ The Critic Agent rejected the generated answer because it could not be verified against the corpus.")