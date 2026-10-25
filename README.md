# ResearchLens

**An Evidence-Aware Deep Learning Framework for Scientific Literature Analysis and Research Gap Discovery.**

Reads a student's paper folder and answers: *Which papers do I read first? What is this corpus strong on? Where is it silent?* Every recommendation shows the exact sentences from the papers that justify it.

## Who this helps
Final-year UG and first-year PG students writing their first literature review.

## Tech Stack
- **From-Scratch Transformer:** 4-layer encoder trained with InfoNCE contrastive loss (PyTorch).
- **Analytics:** K-Means clustering, Centrality ranking, and Saturation mapping.
- **Agentic RAG:** Local LLM (Qwen2.5 via Ollama) with a Critic Agent to eliminate hallucination.
- **Frontend:** Streamlit + Plotly.

## How to Run Locally

1. Clone the repository:
   `git clone https://github.com/ishabhagat04/ResearchLens.git`
2. Create a virtual environment and install dependencies:
   `python -m venv .venv`
   `.\.venv\Scripts\Activate.ps1`
   `pip install -r requirements.txt`
3. **Train the Deep Learning model** (Required, as the `.pt` file is not uploaded to GitHub):
   `python -m src.train_scratch`
4. Install Ollama and pull the model:
   `ollama pull qwen2.5:1.5b`
5. Run the app:
   `streamlit run app.py`

## Evaluation
- **Retrieval:** P@1, P@5, MRR, nDCG@10 (Our from-scratch model beats the TF-IDF baseline).
- **Gap Discovery:** Identifies saturated and silent areas of the corpus.
