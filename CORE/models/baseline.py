import torch
import torch.nn as nn
from torchvision import models


class BaselineModel(nn.Module):
    """ResNet18 기반 이진 분류기 (Live / Spoof)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


if __name__ == "__main__":
    model = BaselineModel()
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print("logits shape", out.shape)
