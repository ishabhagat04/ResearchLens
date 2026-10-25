"""InfoNCE / Multiple-Negatives Ranking Loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiNegRankingLoss(nn.Module):
    def __init__(self, scale=20.0):
        super().__init__()
        self.scale = scale
    def forward(self, anchor, positive):
        logits = self.scale * anchor @ positive.T
        labels = torch.arange(anchor.size(0), device=anchor.device)
        return F.cross_entropy(logits, labels)
