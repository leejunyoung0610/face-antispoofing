import torch
import torch.nn.functional as F

from models.texture_expert import TextureExpert
from models.texture2_expert import Texture2Expert


class ImprovedAdaptiveSystem(torch.nn.Module):
    """
    Dual-stage adaptive operator:
    - Texture confidence < low_threshold: use Frequency only (100%)
    - low_threshold <= Texture < high_threshold: Frequency-leaning fusion (0.3T + 0.7F)
    - Texture >= high_threshold: Texture only
    """

    def __init__(self, low_threshold: float = 0.40, high_threshold: float = 0.55):
        super().__init__()
        self.texture = TextureExpert()
        self.texture2 = Texture2Expert()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_out = self.texture(x)
        t_probs = F.softmax(t_out, dim=1)
        t_conf = t_probs.max(dim=1).values

        results = []
        for i in range(len(x)):
            conf = t_conf[i].item()
            if conf < self.low_threshold:
                f_out = self.texture2(x[i : i + 1])
                results.append(f_out[0])
            elif conf < self.high_threshold:
                f_out = self.texture2(x[i : i + 1])
                results.append(0.3 * t_out[i] + 0.7 * f_out[0])
            else:
                results.append(t_out[i])

        return torch.stack(results, dim=0)
