"""From-scratch word-level tokenizer."""
from __future__ import annotations
import json, re
from collections import Counter
from pathlib import Path
PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9]+")
class ScratchTokenizer:
    def __init__(self, vocab=None, max_len=128):
        self.vocab = vocab or {PAD: 0, UNK: 1}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.max_len = max_len
    @staticmethod
    def _norm(t): return t.lower().strip()
    def tokenize(self, text): return TOKEN_RE.findall(self._norm(text))
    def build_vocab(self, texts, min_freq=2, max_vocab=20000):
        counter = Counter()
        for t in texts: counter.update(self.tokenize(t))
        kept = [(w, c) for w, c in counter.most_common(max_vocab) if c >= min_freq]
        for word, _ in kept:
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
                self.inv_vocab[self.vocab[word]] = word
    def encode(self, text):
        ids = [self.vocab.get(w, self.vocab[UNK]) for w in self.tokenize(text)]
        ids = ids[: self.max_len]
        return ids + [self.vocab[PAD]] * (self.max_len - len(ids))
    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"vocab": self.vocab, "max_len": self.max_len}))
    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(vocab=d["vocab"], max_len=d["max_len"])
