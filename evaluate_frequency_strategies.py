import os

import torch
from torch.utils.data import DataLoader

from analyze_complementary import ComplementarySystem
from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.cascading_system import CascadingSystem
from models.frequency_expert import FrequencyExpert
from models.frequency_spoof_only import FrequencySpoofDetector
from models.texture_expert import TextureExpert


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_checkpoint(preferred: str, fallback: str) -> str:
    return preferred if os.path.exists(preferred) else fallback


def load_texture(device):
    texture = TextureExpert().to(device)
    texture_ckpt = _find_checkpoint(
        "checkpoints/cross/texture.pth", "checkpoints/texture_expert/best_model.pth"
    )
    texture.load_state_dict(torch.load(texture_ckpt, map_location=device))
    texture.eval()
    return texture


def load_frequency(device):
    frequency = FrequencyExpert().to(device)
    freq_ckpt = _find_checkpoint(
        "checkpoints/cross/frequency_asymmetric.pth", "checkpoints/frequency_expert/best_model.pth"
    )
    frequency.load_state_dict(torch.load(freq_ckpt, map_location=device))
    frequency.eval()
    return frequency


def evaluate_model(model, loader, device):
    model.eval()
    correct = total = tp = tn = fp = fn = 0
    with torch.inference_mode():
        for batch in loader:
            raw = batch["raw"].to(device)
            labels = batch["label"].to(device)
            logits = model(raw)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    acc = correct / total if total else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    hter = (far + frr) / 2
    return {"accuracy": acc, "far": far, "frr": frr, "hter": hter}


def main():
    device = get_device()
    print(f"Evaluating Frequency strategies on {device}")
    dataset = ReplayAttackDataset(
        "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
        split="test",
        transform=raw_transforms(),
        raw_transform=raw_transforms(),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    strategies = []
    texture = load_texture(device)
    frequency = load_frequency(device)
    strategies.append(("Texture Only", texture))
    strategies.append(("Frequency Only", frequency))
    strategies.append(("Frequency Spoof Only", FrequencySpoofDetector(load_frequency(device))))
    strategies.append(
        (
            "Cascading (Texture + Frequency)",
            CascadingSystem(load_texture(device), load_frequency(device)),
        )
    )
    strategies.append(
        (
            "Complementary (Disjoint Failures)",
            ComplementarySystem(load_texture(device), load_frequency(device)),
        )
    )

    print("\nStrategy Results:")
    print(f"{'Strategy':<35} {'Acc':>7} {'FAR':>7} {'FRR':>7} {'HTER':>7}")
    for name, model in strategies:
        results = evaluate_model(model, loader, device)
        print(
            f"{name:<35} "
            f"{results['accuracy']*100:6.2f}% "
            f"{results['far']*100:6.2f}% "
            f"{results['frr']*100:6.2f}% "
            f"{results['hter']*100:6.2f}%"
        )


if __name__ == "__main__":
    main()
