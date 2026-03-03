import torch
import torch.nn as nn
import torch.nn.functional as F

from models.texture2_expert import Texture2Expert
from models.texture_expert import TextureExpert


class AdaptiveSystem(nn.Module):
    """
    Texture 확신도 기반 동적 Expert 선택
    - 확신도 높음(>threshold): Texture만 사용
    - 확신도 낮음(<=threshold): Texture + Frequency 앙상블(0.7/0.3)
    """

    def __init__(self, confidence_threshold: float = 0.90):
        super().__init__()
        self.texture = TextureExpert()
        self.texture2 = Texture2Expert()
        self.threshold = float(confidence_threshold)

    def forward(self, x: torch.Tensor):
        t_logits = self.texture(x)
        t_probs = F.softmax(t_logits, dim=1)
        t_conf, _ = t_probs.max(dim=1)  # (B,)

        use_texture2 = t_conf <= self.threshold
        output = t_logits.clone()

        if use_texture2.any():
            x_low = x[use_texture2]
            f_logits_low = self.texture2(x_low)
            output[use_texture2] = 0.7 * t_logits[use_texture2] + 0.3 * f_logits_low

        freq_usage = use_texture2.float().mean()
        return output, freq_usage

