import torch
import torch.nn.functional as F
from torch import nn


class CascadingSystem(nn.Module):
    """Texture가 확신하지 못할 때 Frequency를 보완하여 결합."""

    def __init__(self, texture_model: nn.Module, texture2_model: nn.Module, threshold: float = 0.95):
        super().__init__()
        self.texture = texture_model
        self.texture2 = texture2_param_model
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_out = self.texture(x)
        t_probs = F.softmax(t_out, dim=1)
        t_conf, _ = t_probs.max(dim=1)
        result = torch.zeros_like(t_out)
        for i in range(x.size(0)):
            if t_conf[i] > self.threshold:
                result[i] = t_out[i]
            else:
                f_out = self.texture2(x[i : i + 1])
                result[i] = 0.7 * t_out[i] + 0.3 * f_out[0]
        return result
