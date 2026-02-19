import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence, Tuple, Union


class GatingModule(nn.Module):
    """Soft gating을 활용해 여러 expert를 결합."""

    def __init__(self, num_experts: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.num_experts = num_experts
        self.score_net = nn.Sequential(
            nn.Linear(num_experts * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )
        self.confidence_net = nn.Sequential(
            nn.Linear(num_experts * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        logits: Union[Sequence[torch.Tensor], torch.Tensor],
        texture_logits: torch.Tensor = None,
        frequency_logits: torch.Tensor = None,
        depth_logits: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(logits, Sequence):
            logits_list = logits
        else:
            if None in (texture_logits, frequency_logits, depth_logits):
                raise ValueError("Provide either logits list or all individual tensors.")
            logits_list = [logits, texture_logits, frequency_logits]
        if len(logits_list) != self.num_experts:
            raise ValueError("Number of logits must match num_experts.")
        probs_list = [F.softmax(l, dim=-1) for l in logits_list]
        stacked = torch.stack(probs_list, dim=1)
        flattened = stacked.view(stacked.size(0), -1)
        weights = torch.softmax(self.score_net(flattened), dim=1)
        weighted_logits = torch.sum(weights.unsqueeze(-1) * stacked, dim=1)
        confidence = torch.sigmoid(self.confidence_net(flattened)).squeeze(-1)
        fallback = stacked.mean(dim=1)
        combined = (
            confidence.unsqueeze(-1) * weighted_logits
            + (1 - confidence).unsqueeze(-1) * fallback
        )
        return combined, weights
