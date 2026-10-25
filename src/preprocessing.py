"""Load and clean paper metadata CSV."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
REQUIRED = {"title", "abstract", "year", "keywords", "source_url"}
class DatasetError(ValueError): pass
def _coerce(source):
    df = pd.read_csv(source) if hasattr(source, "read") else pd.read_csv(Path(source))
    df.columns = [c.strip().lower() for c in df.columns]
    return df
def load_papers(source) -> pd.DataFrame:
    try: df = _coerce(source)
    except Exception as e: raise DatasetError(f"Could not read CSV: {e}") from e
    missing = REQUIRED - set(df.columns)
    if missing: raise DatasetError(f"Missing required columns: {sorted(missing)}")
    if "paper_id" not in df.columns: df["paper_id"] = [f"P{i+1}" for i in range(len(df))]
    df["title"] = df["title"].astype(str).str.strip()
    df["abstract"] = df["abstract"].fillna("").astype(str).str.strip()
    df["keywords"] = df["keywords"].fillna("").astype(str)
    df["source_url"] = df["source_url"].fillna("").astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df.loc[(df["year"] < 1900) | (df["year"] > 2100), "year"] = pd.NA
    df = df[df["title"].str.len() > 0]
    df = df.drop_duplicates(subset=["title"], keep="first").reset_index(drop=True)
    df["analysis_text"] = (df["title"] + " . " + df["abstract"] + " . " + df["keywords"]).str.strip()
    return df
