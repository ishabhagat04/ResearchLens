from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
def cluster_papers(vectors, texts, k=None):
    n = len(vectors)
    if n < 2: return None, [], {}
    k = k or max(2, min(n // 2, 4))
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(vectors)
    metrics = {}
    if len(set(km.labels_)) > 1: metrics["silhouette"] = float(silhouette_score(vectors, km.labels_))
    return km.labels_.tolist(), [], metrics
