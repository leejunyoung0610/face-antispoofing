import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.frequency_expert import FrequencyExpert


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_checkpoint(preferred: str, fallback: str) -> str:
    return preferred if os.path.exists(preferred) else fallback


def load_frequency(device):
    frequency = FrequencyExpert().to(device)
    freq_ckpt = _find_checkpoint(
        "checkpoints/cross/frequency_asymmetric.pth",
        "checkpoints/frequency_expert/best_model.pth",
    )
    frequency.load_state_dict(torch.load(freq_ckpt, map_location=device))
    frequency.eval()
    return frequency


def preprocess_channels(x):
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
    return {"Gray": gray, "FFT": mag, "Lap": edge}


def build_input(channels, include):
    zeros = torch.zeros_like(channels["Gray"])
    mapping = {
        "Gray": channels["Gray"] if "Gray" in include else zeros,
        "FFT": channels["FFT"] if "FFT" in include else zeros,
        "Lap": channels["Lap"] if "Lap" in include else zeros,
    }
    stack = torch.stack([mapping["Gray"], mapping["FFT"], mapping["Lap"]], dim=1)
    return stack


def evaluate_combination(frequency, loader, device, include, label):
    correct = total = tp = tn = fp = fn = 0
    with torch.inference_mode():
        for batch in loader:
            raw = batch["raw"].to(device)
            labels = batch["label"].to(device)
            channels = preprocess_channels(raw)
            inp = build_input(channels, include).to(device)
            logits = frequency.backbone(inp)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    live = tn / (tn + fp) if (tn + fp) else 0.0
    spoof = tp / (tp + fn) if (tp + fn) else 0.0
    overall = correct / total if total else 0.0
    return {
        "label": label,
        "live": live,
        "spoof": spoof,
        "overall": overall,
        "fp": fp,
        "fn": fn,
    }


def main():
    device = get_device()
    dataset = ReplayAttackDataset(
        "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
        split="test",
        transform=raw_transforms(),
        raw_transform=raw_transforms(),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    frequency = load_frequency(device)

    combos = [
        (["Gray", "FFT"], "Gray+FFT (no Lap)"),
        (["Gray", "Lap"], "Gray+Lap (no FFT)"),
        (["FFT", "Lap"], "FFT+Lap (no Gray)"),
    ]
    results = []
    for include, label in combos:
        metrics = evaluate_combination(frequency, loader, device, include, label)
        results.append(metrics)
    all_metrics = evaluate_combination(frequency, loader, device, ["Gray", "FFT", "Lap"], "All 3 channels")

    print("=== Channel Ablation Study ===")
    print(f"{'Channels':20} {'Live':>7} {'Spoof':>7} {'Overall':>8} {'Live_FP':>8} {'Spoof_FN':>8}")
    for metrics in results:
        print(
            f"{metrics['label']:20} "
            f"{metrics['live']*100:6.2f}% "
            f"{metrics['spoof']*100:6.2f}% "
            f"{metrics['overall']*100:7.2f}% "
            f"{metrics['fp']:8} "
            f"{metrics['fn']:8}"
        )
    print(
        f"{all_metrics['label']:20} "
        f"{all_metrics['live']*100:6.2f}% "
        f"{all_metrics['spoof']*100:6.2f}% "
        f"{all_metrics['overall']*100:7.2f}% "
        f"{all_metrics['fp']:8} "
        f"{all_metrics['fn']:8}"
    )

    # Analysis
    live_drops = {
        metrics["label"]: all_metrics["live"] - metrics["live"] for metrics in results
    }
    spoof_drops = {
        metrics["label"]: all_metrics["spoof"] - metrics["spoof"] for metrics in results
    }
    important_live = max(live_drops, key=live_drops.get)
    important_spoof = max(spoof_drops, key=spoof_drops.get)
    print("\nAnalysis:")
    print(f"- Most important channel combo: {important_live}")
    print(f"- Removing {important_live.split()[0]} hurts Live most")
    print(f"- Removing {important_spoof.split()[0]} hurts Spoof most")


if __name__ == "__main__":
    main()
