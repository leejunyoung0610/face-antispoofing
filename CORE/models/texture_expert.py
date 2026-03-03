import torch
import torch.nn as nn
import timm


class TextureExpert(nn.Module):
    """EfficientNet-B0 기반 texture expert."""

    def __init__(self, dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    import torch

    model = TextureExpert()
    x = torch.randn(4, 3, 224, 224)
    logits = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", logits.shape)
    print("Sample logits:", logits[0])
