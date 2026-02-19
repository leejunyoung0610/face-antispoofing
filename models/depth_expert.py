from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class DepthExpert(nn.Module):
    """MiDaS 기반 depth map 통계를 이용한 Expert."""

    def __init__(self, dropout: float = 0.3, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.midas_transforms.small_transform
        self.midas.to(self.device).eval()
        self.classifier = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def _extract_depth_features(self, image: torch.Tensor) -> torch.Tensor:
        img_np = (image.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        input_batch = self.transform(img_np).to(self.device)
        with torch.no_grad():
            depth = self.midas(input_batch)
        if depth.dim() == 1:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 2:
            depth = depth.unsqueeze(0).unsqueeze(0)
        elif depth.dim() == 3:
            depth = depth.unsqueeze(1)

        depth = F.interpolate(
            depth,
            size=(image.shape[1], image.shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        depth = depth - depth.min()
        if depth.max() > 0:
            depth = depth / depth.max()

        grad_y, grad_x = torch.gradient(depth)
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)

        features = torch.tensor(
            [
                depth.var().item(),
                (depth.max() - depth.min()).item(),
                grad_mag.mean().item(),
                grad_mag.std().item(),
                grad_mag.max().item(),
                grad_mag.min().item(),
                depth.mean().item(),
                depth.max().item(),
                depth.min().item(),
            ],
            dtype=torch.float32,
            device=self.device,
        )
        if features.norm(p=2) > 0:
            features = features / features.norm(p=2)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = torch.stack([self._extract_depth_features(img) for img in x], dim=0)
        return self.classifier(feats)
