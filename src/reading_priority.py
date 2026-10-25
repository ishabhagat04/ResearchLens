import numpy as np
def reading_priority(frame, vectors, labels, top_per_cluster=1):
    df = frame[["paper_id", "title"]].copy().reset_index(drop=True)
    df["cluster"] = list(labels)
    recs = []
    for cid, group in df.groupby("cluster"):
        idxs = group.index.tolist()
        if len(idxs) < 2:
            recs.append({"cluster": int(cid), "paper_id": group.iloc[0]["paper_id"], "title": group.iloc[0]["title"], "centrality": 1.0, "reason": "Only paper in this cluster."})
            continue
        sub = vectors[idxs]
        sim = sub @ sub.T
        centrality = sim.mean(axis=1)
        best = int(np.argmax(centrality))
        recs.append({"cluster": int(cid), "paper_id": df.iloc[idxs[best]]["paper_id"], "title": df.iloc[idxs[best]]["title"], "centrality": round(float(centrality[best]), 3), "reason": f"Most central in cluster of {len(idxs)} papers."})
    return sorted(recs, key=lambda r: -r["centrality"])
