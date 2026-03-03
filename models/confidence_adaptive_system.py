import torch
import torch.nn.functional as F
from torch import nn

from models.texture2_expert import Texture2Expert
from models.texture_expert import TextureExpert


class ConfidenceAdaptiveSystem(nn.Module):
    """Texture confidence gate: use Texture when confident, delegate to Frequency otherwise."""

    def __init__(self, threshold: float = 0.8):
        super().__init__()
        self.texture = TextureExpert()
        self.texture2 = Texture2Expert()
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_out = self.texture(x)
        t_probs = F.softmax(t_out, dim=1)
        conf, _ = t_probs.max(dim=1)
        low_conf = conf < self.threshold
        result = t_out.clone()
        if low_conf.any():
            freq_out = self.texture2(x[low_conf])
            result[low_conf] = freq_out
        return result
