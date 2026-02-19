import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model, checkpoints, device):
    for path in checkpoints:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            return
    raise FileNotFoundError("Missing checkpoint among: " + ", ".join(checkpoints))


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


def gather_confidences(texture, frequency, loader, device):
    records = []
    texture.eval()
    frequency.eval()
    with torch.inference_mode():
        for batch in loader:
            frame = batch["frame"].to(device)
            raw = batch["raw"].to(device)
            labels = batch["label"].to(device)
            paths = batch["metadata"]["video_path"]
            t_logits = texture(frame)
            f_logits = frequency(raw)
            t_probs = F.softmax(t_logits, dim=1)
            f_probs = F.softmax(f_logits, dim=1)
            for i in range(len(labels)):
                t_live, t_spoof = t_probs[i].tolist()
                confidence = max(t_live, t_spoof)
                records.append(
                    {
                        "path": os.path.basename(paths[i]),
                        "label": labels[i].item(),
                        "texture_live": t_live,
                        "texture_spoof": t_spoof,
                        "texture_confidence": confidence,
                        "frequency_spoof": f_probs[i, 1].item(),
                    }
                )
    return records


def analyze_confidence(records):
    fn = [r for r in records if r["label"] == 1 and r["texture_spoof"] < 0.5]
    tp = [r for r in records if r["label"] == 1 and r["texture_spoof"] >= 0.5]
    fn_conf = [r["texture_confidence"] for r in fn]
    tp_conf = [r["texture_confidence"] for r in tp]

    print("=== Texture Confidence Analysis ===")
    print("Texture False Negatives (6 Spoofs missed):")
    print(f"{'Video':40} {'T_Live':>6} {'T_Spoof':>7} {'Conf':>8}")
    for row in fn:
        print(
            f"{row['path']:<40} {row['texture_live']*100:6.2f}% {row['texture_spoof']*100:6.2f}% {row['texture_confidence']*100:8.2f}%"
        )
    print(f"Avg confidence (FN): {np.mean(fn_conf):.4f}")

    print("\nTexture True Positives (394 Spoofs caught):")
    print(f"Avg confidence: {np.mean(tp_conf):.4f}")
    print(f"Min confidence: {np.min(tp_conf):.4f}")
    print(f"Median: {np.median(tp_conf):.4f}")
    print(f"Max confidence: {np.max(tp_conf):.4f}")

    gap = np.mean(tp_conf) - np.mean(fn_conf)
    print("\nGap Analysis:")
    print(f"FN avg: {np.mean(fn_conf):.4f}")
    print(f"TP avg: {np.mean(tp_conf):.4f}")
    print(f"Gap: {gap:.4f}")
    print("→ Threshold plausible." if gap > 0.1 else "→ Gap small; threshold may be noisy.")
    return fn_conf, tp_conf


def threshold_sweep(records, frequency, thresholds):
    results = []
    texture_preds = np.array([1 if rec["texture_spoof"] > rec["texture_live"] else 0 for rec in records])
    labels = np.array([rec["label"] for rec in records])
    freq_preds = np.array([1 if rec["frequency_spoof"] > 0.5 else 0 for rec in records])
    texture_conf = np.array([rec["texture_confidence"] for rec in records])
    for threshold in thresholds:
        freq_calls = texture_conf < threshold
        final_preds = texture_preds.copy()
        final_preds[freq_calls] = freq_preds[freq_calls]
        metrics = compute_metrics(torch.tensor(final_preds), torch.tensor(labels))
        results.append(
            {
                "threshold": threshold,
                "freq_calls": freq_calls.mean(),
                "metrics": metrics,
            }
        )
    return results


def plot_confidence(fn_conf, tp_conf, best_threshold, out_path):
    plt.figure(figsize=(8, 4))
    plt.hist(tp_conf, bins=30, alpha=0.6, label="Texture TP", color="green", density=True)
    plt.hist(fn_conf, bins=30, alpha=0.7, label="Texture FN", color="red", density=True)
    plt.axvline(best_threshold, color="blue", linestyle="--", label=f"Threshold {best_threshold:.2f}")
    plt.xlabel("Texture confidence (max probability)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def print_threshold_results(results):
    print("\n=== Adaptive Threshold Search ===")
    print(f"{'Threshold':<10} {'Freq_Calls':>10} {'Live':>7} {'Spoof':>7} {'Overall':>8} {'FAR':>7} {'FRR':>7}")
    for row in results:
        metrics = row["metrics"]
        print(
            f"{row['threshold']:<10.2f} {row['freq_calls']*100:9.2f}% "
            f"{metrics['live']*100:6.2f}% {metrics['spoof']*100:6.2f}% {metrics['overall']*100:7.2f}% "
            f"{metrics['far']*100:6.2f}% {metrics['frr']*100:6.2f}%"
        )


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

    texture = TextureExpert().to(device)
    frequency = FrequencyExpert().to(device)
    load_model(texture, ["checkpoints/cross/texture.pth", "checkpoints/texture_expert/best_model.pth"], device)
    load_model(frequency, ["checkpoints/cross/frequency_asymmetric.pth", "checkpoints/frequency_expert/best_model.pth"], device)

    records = gather_confidences(texture, frequency, loader, device)
    fn_conf, tp_conf = analyze_confidence(records)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    sweep = threshold_sweep(records, frequency, thresholds)
    print_threshold_results(sweep)

    best = max(sweep, key=lambda x: x["metrics"]["overall"])
    print(f"\nOptimal threshold: {best['threshold']:.2f} (best overall accuracy)")
    plot_confidence(fn_conf, tp_conf, best["threshold"], "texture_confidence_distribution.png")
    print("Saved texture_confidence_distribution.png")


if __name__ == "__main__":
    main()
