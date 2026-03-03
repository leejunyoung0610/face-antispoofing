import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    False Negative (Spoof→Live) 큰 페널티
    False Positive (Live→Spoof) 작은 페널티
    """

    def __init__(self, fn_weight: float = 5.0, fp_weight: float = 1.0):
        super().__init__()
        self.fn_weight = float(fn_weight)
        self.fp_weight = float(fp_weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        preds = logits.argmax(dim=1)

        fn_mask = (labels == 1) & (preds == 0)  # Spoof(1)를 Live(0)로 판단
        fp_mask = (labels == 0) & (preds == 1)  # Live(0)를 Spoof(1)로 판단

        weights = torch.ones_like(ce_loss)
        weights[fn_mask] = self.fn_weight
        weights[fp_mask] = self.fp_weight

        return (ce_loss * weights).mean()

