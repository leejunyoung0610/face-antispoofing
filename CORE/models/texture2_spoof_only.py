import torch
import torch.nn.functional as F
from torch import nn


class FrequencySpoofDetector(nn.Module):
    """Frequency expert가 확신하는 Spoof만 최종적으로 가져오는 래퍼."""

    def __init__(self, texture2_model: nn.Module, spoof_threshold: float = 0.9):
        super().__init__()
        self.texture2 = texture2_param_model
        self.spoof_threshold = spoof_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.texture2(x)
        probs = F.softmax(out, dim=1)
        spoof_confidence = probs[:, 1]
        result = torch.empty_like(out)
        spoof_override = torch.tensor([-10.0, 10.0], device=out.device)
        live_override = torch.tensor([10.0, -10.0], device=out.device)
        high_conf_mask = spoof_confidence > self.spoof_threshold
        result[high_conf_mask] = out[high_conf_mask]
        result[~high_conf_mask] = live_override
        return result
