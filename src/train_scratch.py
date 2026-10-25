"""Train the from-scratch encoder."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import load_papers
from src.tokenizer_scratch import ScratchTokenizer
from src.model_scratch import ScratchEncoder
from src.loss_scratch import MultiNegRankingLoss
OUT_DIR = Path("models/scratch-encoder")
EVAL_PATH = Path("data/eval_pairs.json")
class PairDataset(Dataset):
    def __init__(self, pairs, tok):
        self.pairs, self.tok = pairs, tok
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, p = self.pairs[i]
        return (torch.tensor(self.tok.encode(a), dtype=torch.long), torch.tensor(self.tok.encode(p), dtype=torch.long))
def build_pairs(frame, eval_pairs):
    id_to_text = dict(zip(frame["paper_id"], frame["analysis_text"]))
    out = []
    for e in eval_pairs:
        for pid in e["relevant_ids"]:
            if pid in id_to_text: out.append((e["query"], id_to_text[pid]))
    return out
def train(csv_path="data/sample_papers.csv", epochs=15, batch_size=8, lr=1e-3, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    frame = load_papers(csv_path)
    pairs = build_pairs(frame, json.loads(EVAL_PATH.read_text()))
    if len(pairs) < 4: raise RuntimeError("Need >=4 pairs in data/eval_pairs.json")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = ScratchTokenizer(max_len=128)
    tok.build_vocab(frame["analysis_text"].tolist(), min_freq=1, max_vocab=20000)
    tok.save(OUT_DIR / "tokenizer.json")
    print(f"Vocab: {len(tok.vocab)} | pairs: {len(pairs)}")
    dl = DataLoader(PairDataset(pairs, tok), batch_size=batch_size, shuffle=True, drop_last=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ScratchEncoder(vocab_size=len(tok.vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = MultiNegRankingLoss()
    for ep in range(1, epochs + 1):
        model.train(); total, n = 0.0, 0
        for a_ids, p_ids in dl:
            a_ids, p_ids = a_ids.to(device), p_ids.to(device)
            loss = loss_fn(model(a_ids), model(p_ids))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item(); n += 1
        sched.step()
        print(f"epoch {ep:02d}  loss={total/max(n,1):.4f}")
    torch.save({"state_dict": model.state_dict(), "config": {"vocab_size": len(tok.vocab), "d_model": 256, "nhead": 4, "num_layers": 4, "max_len": 128}}, OUT_DIR / "encoder.pt")
    print(f"Saved -> {OUT_DIR/'encoder.pt'}")
if __name__ == "__main__": train()
