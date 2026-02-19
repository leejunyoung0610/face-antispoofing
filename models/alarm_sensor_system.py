import torch
import torch.nn.functional as F
from torch import nn

from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert


class AlarmSensorSystem(nn.Module):
    """
    Texture expert serves as the primary decision model.
    Frequency expert acts as an alarm sensor that only overrides when it is highly confident about Spoof.
    """

    def __init__(self, alarm_threshold: float = 0.9):
        super().__init__()
        self.texture = TextureExpert()
        self.frequency = FrequencyExpert()
        self.alarm_threshold = alarm_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_out = self.texture(x)
        f_out = self.frequency(x)

        t_probs = F.softmax(t_out, dim=1)
        f_probs = F.softmax(f_out, dim=1)

        result = torch.zeros_like(t_out)
        alarm = f_probs[:, 1] > self.alarm_threshold
        result[alarm] = torch.tensor([-100.0, 100.0], device=x.device)
        result[~alarm] = t_out[~alarm]
        return result


if __name__ == "__main__":
    sample = torch.randn(4, 3, 224, 224)
    model = AlarmSensorSystem()
    out = model(sample)
    print("Output shape:", out.shape)
