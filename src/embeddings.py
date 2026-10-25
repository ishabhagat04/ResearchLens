"""Embeddings from scratch model."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np, torch
from src.tokenizer_scratch import ScratchTokenizer
from src.model_scratch import ScratchEncoder
MODEL_DIR = Path("models/scratch-encoder")
CACHE_DIR = Path(".cache/embeddings"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "scratch-transformer"
_TOKENIZER, _MODEL = None, None
def get_model():
    global _TOKENIZER, _MODEL
    if _MODEL is not None: return _TOKENIZER, _MODEL
    if not (MODEL_DIR / "encoder.pt").exists(): raise RuntimeError("Train first: python -m src.train_scratch")
    tok = ScratchTokenizer.load(MODEL_DIR / "tokenizer.json")
    ckpt = torch.load(MODEL_DIR / "encoder.pt", map_location="cpu")
    mdl = ScratchEncoder(**ckpt["config"])
    mdl.load_state_dict(ckpt["state_dict"])
    mdl.eval()
    _TOKENIZER, _MODEL = tok, mdl
    return tok, mdl
def _hash(texts):
    h = hashlib.sha256()
    for t in texts: h.update(t.encode())
    return h.hexdigest()[:16]
@torch.no_grad()
def embed_texts(texts, use_cache=True):
    key = _hash(texts)
    fp = CACHE_DIR / f"{MODEL_NAME}_{key}.npy"
    if use_cache and fp.exists(): return np.load(fp)
    tok, mdl = get_model()
    ids = torch.tensor([tok.encode(t) for t in texts], dtype=torch.long)
    vecs = mdl(ids).cpu().numpy().astype(np.float32)
    if use_cache: np.save(fp, vecs)
    return vecs
