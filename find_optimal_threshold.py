import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.dataset import ReplayAttackDataset, raw_transforms
from models.texture_expert import TextureExpert
from models.frequency_expert import FrequencyExpert


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model, preferred, fallback, device):
    path = preferred if os.path.exists(preferred) else fallback
    model.load_state_dict(torch.load(path, map_location=device))
    return path


def compute_metrics(preds: torch.Tensor, labels: torch.Tensor):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    live = tn / (tn + fp) if (tn + fp) else 0.0
    spoof = tp / (tp + fn) if (tp + fn) else 0.0
    overall = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    return {"overall": overall, "live": live, "spoof": spoof, "far": far, "frr": frr}


def evaluate_threshold(texture, frequency, loader, device, threshold: float):
    preds_all = []
    labels_all = []
    freq_calls = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            x = batch["raw"].to(device)
            labels = batch["label"].to(device)
            t_logits = texture(x)
            t_probs = F.softmax(t_logits, dim=1)
            t_conf = t_probs.max(dim=1).values
            use_freq = t_conf < threshold
            out = t_logits.clone()
            if use_freq.any():
                f_logits = frequency(x[use_freq])
                out[use_freq] = 0.7 * t_logits[use_freq] + 0.3 * f_logits
            freq_calls += use_freq.sum().item()
            total += labels.size(0)
            preds_all.append(out.argmax(dim=1).cpu())
            labels_all.append(labels.cpu())
    preds = torch.cat(preds_all)
    labels = torch.cat(labels_all)
    m = compute_metrics(preds, labels)
    m["freq_call"] = freq_calls / max(total, 1)
    return m


def main():
    device = get_device()
    seed = 42
    data_root = os.path.join("data", "replay-attack", "datasets", "fas_pure_data", "Idiap-replayattack")
    dev_ds = ReplayAttackDataset(
        data_root,
        split="dev",
        transform=raw_transforms(),
        raw_transform=raw_transforms(),
        freq_crop=False,
        seed=seed,
    )
    loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=0)

    texture = TextureExpert().to(device)
    frequency = FrequencyExpert().to(device)
    tex_path = load_checkpoint(texture, "checkpoints/cross/texture.pth", "checkpoints/texture_expert/best_model.pth", device)
    freq_path = load_checkpoint(frequency, "checkpoints/cross/frequency_asymmetric.pth", "checkpoints/frequency_expert/best_model.pth", device)

    texture.eval()
    frequency.eval()

    print(f"Using texture: {tex_path}")
    print(f"Using frequency: {freq_path}")
    print(f"DEV size: {len(dev_ds)} videos (seed={seed})")

    thresholds = np.round(np.arange(0.50, 1.00, 0.05), 2)
    results = []
    print("=== Threshold Search (DEV) ===")
    print(f"{'Threshold':<10} {'Overall':>8} {'Live':>7} {'Spoof':>7} {'FAR':>7} {'FRR':>7} {'Freq_Call':>10}")
    for thr in thresholds:
        m = evaluate_threshold(texture, frequency, loader, device, float(thr))
        results.append({"threshold": float(thr), **m})
        print(
            f"{thr:<10.2f} {m['overall']*100:7.2f}% {m['live']*100:6.2f}% {m['spoof']*100:6.2f}% "
            f"{m['far']*100:6.2f}% {m['frr']*100:6.2f}% {m['freq_call']*100:9.2f}%"
        )

    best = max(results, key=lambda r: r["overall"])
    print(f"\nOptimal: {best['threshold']:.2f} (Overall {best['overall']*100:.2f}%)")

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "optimal_threshold.json"
    out_path.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

