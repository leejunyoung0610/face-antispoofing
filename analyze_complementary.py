import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert


def _find_checkpoint(preferred: str, fallback: str) -> str:
    return preferred if os.path.exists(preferred) else fallback


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ComplementarySystem(nn.Module):
    """Texture/Frequency 실패 케이스가 겹치지 않는다고 가정하고 단순 보정."""

    def __init__(self, texture: nn.Module, frequency: nn.Module):
        super().__init__()
        self.texture = texture
        self.frequency = frequency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t_out = self.texture(x)
        f_out = self.frequency(x)
        t_preds = t_out.argmax(dim=1)
        f_preds = f_out.argmax(dim=1)
        result = torch.empty_like(t_out)
        spoof_logit = torch.tensor([-10.0, 10.0], device=t_out.device)
        live_logit = torch.tensor([10.0, -10.0], device=t_out.device)
        for idx in range(x.size(0)):
            if t_preds[idx] == 1 or f_preds[idx] == 1:
                result[idx] = spoof_logit
            else:
                result[idx] = live_logit
        return result


def load_experts(device):
    texture = TextureExpert().to(device)
    freq = FrequencyExpert().to(device)
    texture_ckpt = _find_checkpoint(
        "checkpoints/cross/texture.pth", "checkpoints/texture_expert/best_model.pth"
    )
    freq_ckpt = _find_checkpoint(
        "checkpoints/cross/frequency_asymmetric.pth", "checkpoints/frequency_expert/best_model.pth"
    )
    texture.load_state_dict(torch.load(texture_ckpt, map_location=device))
    freq.load_state_dict(torch.load(freq_ckpt, map_location=device))
    texture.eval()
    freq.eval()
    return texture, freq


def collect_spoof_failures(model, dataset: ReplayAttackDataset, device):
    failures = set()
    model.eval()
    with torch.inference_mode():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            path = dataset.samples[idx]["video_path"]
            raw = sample["raw"].unsqueeze(0).to(device)
            label = sample["label"]
            pred = model(raw).argmax(1).item()
            if label == 1 and pred == 0:
                failures.add(os.path.basename(path))
    return failures


def evaluate_model(model, dataset, device):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    correct = total = tp = tn = fp = fn = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            raw = batch["raw"].to(device)
            labels = batch["label"].to(device)
            preds = model(raw).argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    acc = correct / total if total else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    return {"accuracy": acc, "far": far, "frr": frr}


def main():
    device = get_device()
    print(f"Analyzing on device {device}")
    texture, frequency = load_experts(device)
    dataset = ReplayAttackDataset(
        "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
        split="test",
        transform=raw_transforms(),
        raw_transform=raw_transforms(),
    )

    texture_fails = collect_spoof_failures(texture, dataset, device)
    freq_fails = collect_spoof_failures(frequency, dataset, device)
    print(f"Texture False Negatives: {len(texture_fails)}")
    print(f"Frequency False Negatives: {len(freq_fails)}")
    print("겹치는 케이스:", texture_fails & freq_fails)
    print("Texture만:", texture_fails - freq_fails)
    print("Frequency만:", freq_fails - texture_fails)

    complementary = ComplementarySystem(texture, frequency).to(device)
    metrics = evaluate_model(complementary, dataset, device)
    print("\nComplementary System Metrics:")
    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")


if __name__ == "__main__":
    main()
