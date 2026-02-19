import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FrequencyExpert(nn.Module):
    """CNN expert using grayscale + FFT + laplacian channels."""

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    @staticmethod
    def preprocess(x: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        fft = torch.fft.fft2(gray)
        mag = torch.log1p(torch.abs(torch.fft.fftshift(fft)))
        mag = (mag - mag.amin(dim=[1, 2], keepdim=True)) / (
            mag.amax(dim=[1, 2], keepdim=True) - mag.amin(dim=[1, 2], keepdim=True) + 1e-8
        )
        laplacian_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=x.device
        ).view(1, 1, 3, 3)
        edge = F.conv2d(gray.unsqueeze(1), laplacian_kernel, padding=1).squeeze(1)
        edge = torch.abs(edge)
        edge = (edge - edge.amin(dim=[1, 2], keepdim=True)) / (
            edge.amax(dim=[1, 2], keepdim=True) - edge.amin(dim=[1, 2], keepdim=True) + 1e-8
        )
        return torch.stack([gray, mag, edge], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = self.preprocess(x)
        return self.backbone(inp)


if __name__ == "__main__":
    import torch

    model = FrequencyExpert()
    sample = torch.randn(2, 3, 224, 224)
    logits = model(sample)
    print("Output shape:", logits.shape)
