import torch
import torch.nn as nn

from models.frequency_expert import FrequencyExpert
from models.gating import GatingModule
from models.texture_expert import TextureExpert


class FinalSystem(nn.Module):
    """
    Texture + Frequency 2-Expert system (Baseline 제외).
    입력: raw (0~1) RGB 이미지 (B, 3, 224, 224)
    출력: (combined_logits, weights)
    """

    def __init__(self):
        super().__init__()
        self.texture = TextureExpert()
        self.frequency = FrequencyExpert()
        self.gating = GatingModule(num_experts=2)

        # 기본은 균등 결합에 가깝게 시작(가중치/신뢰도 네트워크 0 초기화)
        for m in self.gating.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        t_out = self.texture(x)
        f_out = self.frequency(x)
        combined, weights = self.gating([t_out, f_out])
        return combined, weights

