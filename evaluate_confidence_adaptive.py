import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.confidence_adaptive_system import ConfidenceAdaptiveSystem


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_pretrained(system, texture_path, freq_path, device):
    texture_state = torch.load(texture_path, map_location=device)
    freq_state = torch.load(freq_path, map_location=device)
    system.texture.load_state_dict(texture_state)
    system.frequency.load_state_dict(freq_state)


def compute_metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    live = tn / (tn + fp) if (tn + fp) else 0.0
    spoof = tp / (tp + fn) if (tp + fn) else 0.0
    overall = (tp + tn) / (tp + tn + fp + fn)
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    return {"live": live, "spoof": spoof, "overall": overall, "far": far, "frr": frr}


def evaluate(system, loader, device):
    system.eval()
    preds = []
    labels = []
    freq_calls = 0
    with torch.inference_mode():
        for batch in loader:
            frame = batch["frame"].to(device)
            labels.append(batch["label"])
            out = system(frame)
            preds.append(out.argmax(1).cpu())
            texture_out = system.texture(frame)
            mask = F.softmax(texture_out, dim=1).max(1).values < system.threshold
            freq_calls += mask.sum().item()
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    metrics = compute_metrics(preds, labels)
    return metrics, freq_calls / len(labels)


def main():
    device = get_device()
    loader = DataLoader(
        ReplayAttackDataset(
            os.path.join("data", "replay-attack", "datasets", "fas_pure_data", "Idiap-replayattack"),
            split="test",
            transform=raw_transforms(),
            raw_transform=raw_transforms(),
            freq_crop=False,
        ),
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    system = ConfidenceAdaptiveSystem(threshold=0.8).to(device)
    load_pretrained(
        system,
        "checkpoints/cross/texture.pth",
        "checkpoints/cross/frequency_asymmetric.pth",
        device,
    )
    metrics, freq_ratio = evaluate(system, loader, device)
    print("=== Confidence Adaptive System ===")
    print(f"Threshold: {system.threshold}")
    print(f"Frequency used for {freq_ratio*100:.2f}% of samples.")
    print(
        f"Live: {metrics['live']*100:.2f}% | Spoof: {metrics['spoof']*100:.2f}% | Overall: {metrics['overall']*100:.2f}% | FAR: {metrics['far']*100:.2f}% | FRR: {metrics['frr']*100:.2f}%"
    )


if __name__ == "__main__":
    main()
