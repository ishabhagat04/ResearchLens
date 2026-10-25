"""Grounded answer generation."""
from src.llm_client import generate

PROMPT = """Answer the question using ONLY the evidence below.
Cite paper IDs in [brackets]. If evidence is insufficient, reply INSUFFICIENT_EVIDENCE.

Question: {question}

Evidence:
{evidence}

Answer:"""

def build_evidence_block(sentences):
    return "\n".join(f"[{s['paper_id']}] {s['sentence']}" for s in sentences)

def generate_answer(question, retrieved):
    if not retrieved:
        return {"answer": "INSUFFICIENT_EVIDENCE", "cited_paper_ids": [], "evidence": []}
    raw = generate(PROMPT.format(question=question, evidence=build_evidence_block(retrieved))).strip()
    cited = sorted({r["paper_id"] for r in retrieved if r["paper_id"] in raw})
    return {"answer": raw, "cited_paper_ids": cited, "evidence": retrieved}