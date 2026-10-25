"""Sentence-level embedding index for evidence retrieval."""
from __future__ import annotations
import re
import numpy as np
from src.embeddings import embed_texts

def split_sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 25]

class EvidenceIndex:
    def __init__(self, frame):
        self.meta, self.sentences = [], []
        for _, row in frame.iterrows():
            for s in split_sentences(row["abstract"]):
                self.sentences.append(s)
                self.meta.append({"paper_id": row["paper_id"], "title": row["title"], "sentence": s, "source_url": row["source_url"]})
        self.vectors = embed_texts(self.sentences) if self.sentences else np.zeros((0, 256))

    def retrieve(self, query: str, k: int = 3):
        if len(self.vectors) == 0: return []
        q = embed_texts([query])[0]
        scores = self.vectors @ q
        return [{**self.meta[i], "score": float(scores[i])} for i in np.argsort(-scores)[:k]]