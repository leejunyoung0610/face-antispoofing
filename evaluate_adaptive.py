import argparse
import os

import numpy as np
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_cross_dataset import ReplayAttackDataset
from models.adaptive_system import AdaptiveSystem
from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert


def get_device() -> torch.device:
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[idx]
    return float(eer) * 100.0


def compute_metrics(labels: torch.Tensor, logits: torch.Tensor):
    labels = labels.detach().cpu()
    logits = logits.detach().cpu()

    probs = torch.softmax(logits, dim=1)[:, 1]
    preds = logits.argmax(dim=1)

    live_mask = labels == 0
    spoof_mask = labels == 1
    live_acc = (preds[live_mask] == 0).float().mean().item() if live_mask.any() else 0.0
    spoof_acc = (preds[spoof_mask] == 1).float().mean().item() if spoof_mask.any() else 0.0
    overall = (preds == labels).float().mean().item()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    hter = (far + frr) / 2 * 100.0

    eer = compute_eer(labels.numpy(), probs.numpy())
    return live_acc * 100.0, spoof_acc * 100.0, overall * 100.0, hter, eer


def eval_texture_only(model: TextureExpert, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval Texture", leave=False):
            x = batch["raw"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            all_logits.append(logits)
            all_labels.append(y)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    live, spoof, overall, hter, eer = compute_metrics(labels, logits)
    return live, spoof, overall, hter, eer, 0.0


def eval_frequency_only(model: FrequencyExpert, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval Frequency", leave=False):
            x = batch["raw"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            all_logits.append(logits)
            all_labels.append(y)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    live, spoof, overall, hter, eer = compute_metrics(labels, logits)
    return live, spoof, overall, hter, eer, 100.0


def eval_static_ensemble(texture: TextureExpert, frequency: FrequencyExpert, loader: DataLoader, device: torch.device):
    texture.eval()
    frequency.eval()
    all_logits = []
    all_labels = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval Static 2-Expert", leave=False):
            x = batch["raw"].to(device)
            y = batch["label"].to(device)
            t = texture(x)
            f = frequency(x)
            logits = 0.7 * t + 0.3 * f
            all_logits.append(logits)
            all_labels.append(y)
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    live, spoof, overall, hter, eer = compute_metrics(labels, logits)
    return live, spoof, overall, hter, eer, 100.0


def eval_adaptive(system: AdaptiveSystem, loader: DataLoader, device: torch.device):
    system.eval()
    all_logits = []
    all_labels = []
    freq_usage_sum = 0.0
    n = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Eval Adaptive", leave=False):
            x = batch["raw"].to(device)
            y = batch["label"].to(device)
            logits, freq_usage = system(x)
            all_logits.append(logits)
            all_labels.append(y)
            bs = y.size(0)
            freq_usage_sum += float(freq_usage.item()) * bs
            n += bs
    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    live, spoof, overall, hter, eer = compute_metrics(labels, logits)
    freq_usage_pct = (freq_usage_sum / max(n, 1)) * 100.0
    return live, spoof, overall, hter, eer, freq_usage_pct


def load_ckpt(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def main():
    parser = argparse.ArgumentParser(description="Adaptive expert selection evaluation (Replay-Attack).")
    parser.add_argument(
        "--data_root",
        default="data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
        help="Replay-Attack root.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.85, 0.90, 0.95])
    args = parser.parse_args()

    device = get_device()
    ds = ReplayAttackDataset(args.data_root, split="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    texture = TextureExpert().to(device)
    frequency = FrequencyExpert().to(device)

    texture_ckpt = (
        "checkpoints/cross/texture.pth"
        if os.path.exists("checkpoints/cross/texture.pth")
        else "checkpoints/texture_expert/best_model.pth"
    )
    frequency_ckpt = (
        "checkpoints/cross/frequency_asymmetric.pth"
        if os.path.exists("checkpoints/cross/frequency_asymmetric.pth")
        else "checkpoints/cross/frequency.pth"
        if os.path.exists("checkpoints/cross/frequency.pth")
        else "checkpoints/frequency_expert/best_model.pth"
    )

    print(f"Using texture ckpt: {texture_ckpt}")
    print(f"Using frequency ckpt: {frequency_ckpt}")
    texture.load_state_dict(torch.load(texture_ckpt, map_location=device))
    frequency.load_state_dict(torch.load(frequency_ckpt, map_location=device))

    print("\n=== System Comparison (Replay-Attack) ===")
    print(f"{'':22} {'Live':>8} {'Spoof':>8} {'Overall':>8} {'HTER':>8} {'EER':>8} {'Freq_Usage':>10}")

    live, spoof, overall, hter, eer, fu = eval_texture_only(texture, loader, device)
    print(f"{'Texture Only:':22} {live:7.2f}% {spoof:7.2f}% {overall:7.2f}% {hter:7.2f}% {eer:7.2f}% {fu:9.1f}%")

    live, spoof, overall, hter, eer, fu = eval_frequency_only(frequency, loader, device)
    print(f"{'Frequency Only:':22} {live:7.2f}% {spoof:7.2f}% {overall:7.2f}% {hter:7.2f}% {eer:7.2f}% {fu:9.1f}%")

    live, spoof, overall, hter, eer, fu = eval_static_ensemble(texture, frequency, loader, device)
    print(f"{'2-Expert (Static):':22} {live:7.2f}% {spoof:7.2f}% {overall:7.2f}% {hter:7.2f}% {eer:7.2f}% {fu:9.1f}%")

    print("\n--- Confidence Threshold Ablation ---")
    for thr in args.thresholds:
        adaptive = AdaptiveSystem(confidence_threshold=thr).to(device)
        adaptive.texture.load_state_dict(texture.state_dict())
        adaptive.frequency.load_state_dict(frequency.state_dict())
        live, spoof, overall, hter, eer, fu = eval_adaptive(adaptive, loader, device)
        print(f"Adaptive (τ={thr:.2f}): Live={live:.2f}% Spoof={spoof:.2f}% Overall={overall:.2f}% HTER={hter:.2f}% EER={eer:.2f}% Freq_Usage={fu:.1f}%")


if __name__ == "__main__":
    main()

