"""Critic: verify claims against evidence."""
import re
import numpy as np
from src.embeddings import embed_texts

CLAIM_RE = re.compile(r"(?<=[.!?])\s+")

def split_claims(text):
    return [c.strip() for c in CLAIM_RE.split(text) if len(c.strip()) > 15]

def verify(answer, evidence, threshold=0.55):
    if not evidence or answer.strip() == "INSUFFICIENT_EVIDENCE":
        return {"grounded": False, "unsupported_claims": split_claims(answer), "supported_claims": [], "evidence_map": {}, "score": 0.0}
    claims = split_claims(answer)
    if not claims:
        return {"grounded": False, "unsupported_claims": [], "supported_claims": [], "evidence_map": {}, "score": 0.0}
    
    ev_texts = [e["sentence"] for e in evidence]
    cv = embed_texts(claims, use_cache=False)
    ev = embed_texts(ev_texts, use_cache=False)
    sims = cv @ ev.T
    
    supported, unsupported, emap = [], [], {}
    for i, c in enumerate(claims):
        best = int(np.argmax(sims[i]))
        if sims[i][best] >= threshold:
            supported.append(c)
            emap[c] = {"paper_id": evidence[best]["paper_id"], "evidence": evidence[best]["sentence"], "score": float(sims[i][best])}
        else:
            unsupported.append(c)
            
    return {"grounded": len(unsupported) == 0, "unsupported_claims": unsupported, "supported_claims": supported, "evidence_map": emap, "score": float(np.mean([emap[c]["score"] for c in supported])) if supported else 0.0}