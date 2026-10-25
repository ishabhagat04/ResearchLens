import numpy as np
def rank_similar(vectors, query_embedding, limit=5):
    scores = vectors @ query_embedding
    return [(int(i), float(scores[i])) for i in np.argsort(-scores)[:limit]]
def rank_ids(vectors, paper_ids, query_embedding, k=10):
    scores = vectors @ query_embedding
    return [paper_ids[i] for i in np.argsort(-scores)[:k]]
