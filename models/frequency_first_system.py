import torch
import torch.nn.functional as F
from torch import nn

from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert


class FrequencyFirstSystem(nn.Module):
    """
    Frequency-first two-stage system: Frequency makes the fast decision, Texture rechecks only when Frequency suspects Spoof.
    """

    def __init__(self):
        super().__init__()
        self.frequency = FrequencyExpert()
        self.texture = TextureExpert()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_out = self.frequency(x)
        f_pred = f_out.argmax(dim=1)
        result = torch.zeros_like(f_out)
        for i in range(len(x)):
            if f_pred[i] == 0:
                result[i] = torch.tensor([100.0, -100.0], device=x.device)
            else:
                t_out = self.texture(x[i : i + 1])
                result[i] = t_out[0]
        return result
