from collections import Counter

def _keyword_list(raw):
    return [w.strip().lower() for w in str(raw).replace("|", ",").replace(";", ",").split(",") if w.strip()]

def saturation_map(frame, labels, vectors):
    df = frame[["paper_id", "title", "year", "keywords"]].copy().reset_index(drop=True)
    df["cluster"] = list(labels)
    n_total = len(df)
    out = {"saturated": [], "emerging": [], "silent": []}
    for cid, group in df.groupby("cluster"):
        n = len(group)
        label = group["title"].iloc[0].split(":")[-1].strip()[:48]
        if n >= max(2, int(n_total * 0.4)): out["saturated"].append({"cluster": int(cid), "label": label, "papers": n})
        elif n <= 1: out["emerging"].append({"cluster": int(cid), "label": label, "papers": n})
    kw_counts = Counter()
    for k in df["keywords"]:
        for w in _keyword_list(k): kw_counts[w] += 1
    out["silent"] = [{"keyword": k, "papers": c} for k, c in kw_counts.items() if c == 1][:12]
    return out
