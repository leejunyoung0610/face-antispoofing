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


def evaluate(model, loader, device):
    model.eval()
    correct = total = tp = tn = fp = fn = 0
    with torch.inference_mode():
        for batch in loader:
            raw = batch["raw"].to(device)
            labels = batch["label"].to(device)
            logits = model(raw)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    live_acc = tn / (tn + fp) if (tn + fp) else 0.0
    spoof_acc = tp / (tp + fn) if (tp + fn) else 0.0
    overall = correct / total if total else 0.0
    return {"live": live_acc, "spoof": spoof_acc, "overall": overall}


def clone_channels(tensor):
    return tensor.repeat(1, 3, 1, 1)


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

    def channel_input(x, channel):
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

        channels = {
            "Grayscale": gray,
            "FFT": mag,
            "Laplacian": edge,
            "All (3ch)": torch.stack([gray, mag, edge], dim=1),
        }

        selected = channels[channel]
        if channel == "All (3ch)":
            return selected
        return selected.unsqueeze(1).repeat(1, 3, 1, 1)

    def evaluate_channel(channel):
        correct = total = tp = tn = fp = fn = 0
        frequency.eval()
        with torch.inference_mode():
            for batch in loader:
                raw = batch["raw"].to(device)
                labels = batch["label"].to(device)
                inp = channel_input(raw, channel).to(device)
                logits = frequency.backbone(inp)
                preds = logits.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                tp += ((preds == 1) & (labels == 1)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()
        live_acc = tn / (tn + fp) if (tn + fp) else 0.0
        spoof_acc = tp / (tp + fn) if (tp + fn) else 0.0
        overall = correct / total if total else 0.0
        return {
            "live": live_acc,
            "spoof": spoof_acc,
            "overall": overall,
            "live_fp": fp,
            "spoof_fn": fn,
        }

    results = []
    for channel in ["Grayscale", "FFT", "Laplacian", "All (3ch)"]:
        metrics = evaluate_channel(channel)
        results.append((channel, metrics))

    print("=== Channel Contribution Analysis ===")
    print(f"{'Channel':12} {'Live':>7} {'Spoof':>7} {'Overall':>8} {'Notes':>8}")
    for channel, metrics in results:
        notes = "Current" if channel == "All (3ch)" else ""
        print(
            f"{channel:12} "
            f"{metrics['live']*100:6.2f}% "
            f"{metrics['spoof']*100:6.2f}% "
            f"{metrics['overall']*100:7.2f}% "
            f"{notes:>8}"
        )
        print(
            f"   → Live False Positives: {metrics['live_fp']} | Spoof False Negatives: {metrics['spoof_fn']}"
        )


if __name__ == "__main__":
    main()
