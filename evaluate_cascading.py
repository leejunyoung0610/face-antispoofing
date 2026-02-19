import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms.functional import to_pil_image

from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert
from utils.cascading import cascading_predict


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cascading strategy.")
    parser.add_argument("--data_root", default="data", help="Workspace data root.")
    parser.add_argument("--freq_crop", type=int, choices=[0, 1], default=1, help="Enable face cropping for frequency path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for val split and crop dump.")
    parser.add_argument(
        "--dump_crops",
        action="store_true",
        help="Save a few raw/freq crops under output/debug_crops/ for manual inspection.",
    )
    parser.add_argument(
        "--max_dump",
        type=int,
        default=20,
        help="Max number of random samples to dump when --dump_crops is set.",
    )
    parser.add_argument(
        "--dump_output",
        default="output/debug_crops",
        help="Directory to place crop debug images.",
    )
    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_to_val(dataset, val_ratio=0.2, seed=42):
    total = len(dataset)
    val_len = max(1, int(total * val_ratio))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    val_indices = indices[:val_len]
    return Subset(dataset, val_indices)


def collect_logits(texture, frequency, loader, device):
    texture.eval()
    frequency.eval()
    tex_logits = []
    freq_logits = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frame"].to(device)
            freq_raw = batch["freq_raw"].to(device)
            labels.append(batch["label"])
            tex_logits.append(texture(frames))
            freq_logits.append(frequency(freq_raw))
    return (
        torch.cat(tex_logits, dim=0).cpu(),
        torch.cat(freq_logits, dim=0).cpu(),
        torch.cat(labels, dim=0).cpu(),
    )


def compute_metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    total = len(labels)
    acc = (tp + tn) / total if total else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    hter = (far + frr) / 2
    return {"accuracy": acc, "far": far, "frr": frr, "hter": hter, "fp": fp, "fn": fn}


def grid_search_thresholds(
    tex_logits, freq_logits, labels, t_live_range, t_spoof_range, f_spoof_range
):
    best = None
    for t_live in t_live_range:
        for t_spoof in t_spoof_range:
            for f_spoof in f_spoof_range:
                preds = cascading_predict(
                    tex_logits, freq_logits, t_live=t_live, t_spoof=t_spoof, f_spoof=f_spoof
                )
                metrics = compute_metrics(preds, labels)
                better_far = best is None or metrics["far"] < best["metrics"]["far"] - 1e-6
                far_tie = (
                    best is not None
                    and abs(metrics["far"] - best["metrics"]["far"]) < 1e-6
                )
                better_frr = far_tie and metrics["frr"] < best["metrics"]["frr"] - 1e-6
                frr_tie = far_tie and abs(metrics["frr"] - best["metrics"]["frr"]) < 1e-6
                better_hter = frr_tie and metrics["hter"] < best["metrics"]["hter"] - 1e-6
                if better_far or better_frr or better_hter:
                    best = {
                        "t_live": t_live,
                        "t_spoof": t_spoof,
                        "f_spoof": f_spoof,
                        "metrics": metrics,
                    }
    """Select thresholds by FAR -> FRR -> HTER."""
    return best


def evaluate_on_loader(texture, frequency, loader, device, thresholds=None):
    texture.eval()
    frequency.eval()
    all_metrics = []
    cascade_preds = []
    texture_preds = []
    frequency_preds = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frame"].to(device)
            freq_raw = batch["freq_raw"].to(device)
            tex_out = texture(frames)
            freq_out = frequency(freq_raw)
            labels_buff = batch["label"].cpu()
            labels.append(labels_buff)
            texture_preds.append(tex_out.argmax(1).cpu())
            frequency_preds.append(freq_out.argmax(1).cpu())
            if thresholds:
                cascade = cascading_predict(
                    tex_out,
                    freq_out,
                    t_live=thresholds["t_live"],
                    t_spoof=thresholds["t_spoof"],
                    f_spoof=thresholds["f_spoof"],
                )
                cascade_preds.append(cascade.cpu())
    texture_preds = torch.cat(texture_preds)
    frequency_preds = torch.cat(frequency_preds)
    labels = torch.cat(labels)
    metrics = {
        "texture": compute_metrics(texture_preds, labels),
        "frequency": compute_metrics(frequency_preds, labels),
    }
    if thresholds:
        metrics["cascading"] = compute_metrics(torch.cat(cascade_preds), labels)
    return metrics


def log_crop_statistics(dataset, label):
    total = len(dataset)
    if total == 0:
        return
    success = 0
    for item in dataset:
        if item.get("freq_crop_success"):
            success += 1
    fallback = total - success
    ratio = success / total
    print(
        f"[{label}] Face crop success: {success}/{total} ({ratio*100:.1f}%), fallback: {fallback}"
    )


def dump_random_crops(dataset, out_dir, count=20, seed=42):
    os.makedirs(out_dir, exist_ok=True)
    frame_dir = os.path.join(out_dir, "crops")
    os.makedirs(frame_dir, exist_ok=True)
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(count, len(dataset)))
    for idx in indices:
        sample = dataset[idx]
        raw = sample["raw"]
        freq_raw = sample["freq_raw"]
        success = sample.get("freq_crop_success", False)
        raw_img = to_pil_image(raw)
        freq_img = to_pil_image(freq_raw)
        suffix = "success" if success else "fallback"
        raw_img.save(os.path.join(frame_dir, f"{idx}_{suffix}_raw.png"))
        freq_img.save(os.path.join(frame_dir, f"{idx}_{suffix}_freq.png"))


def load_model_checkpoint(model, possible_paths, device):
    for path in possible_paths:
        if os.path.exists(path):
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
            return path
    return None


def main():
    args = parse_args()
    device = get_device()
    data_root = os.path.join(args.data_root, "replay-attack", "datasets", "fas_pure_data", "Idiap-replayattack")
    freq_crop_flag = bool(args.freq_crop)
    train_dataset = ReplayAttackDataset(
        data_root,
        split="train",
        transform=raw_transforms(),
        raw_transform=raw_transforms(),
        freq_crop=freq_crop_flag,
    )
    log_crop_statistics(train_dataset, "Replay train (entire)")
    val_subset = split_to_val(train_dataset, val_ratio=0.2, seed=args.seed)
    log_crop_statistics(val_subset, "Replay train subset (freq_crop stats)")
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=0)

    if args.dump_crops:
        dump_random_crops(val_subset, args.dump_output, count=args.max_dump, seed=args.seed)

    texture = TextureExpert().to(device)
    frequency = FrequencyExpert().to(device)
    texture_path = load_model_checkpoint(
        texture,
        [
            "checkpoints/cross/texture.pth",
            "checkpoints/texture_expert/best_model.pth",
        ],
        device,
    )
    frequency_path = load_model_checkpoint(
        frequency,
        [
            "checkpoints/cross/frequency_asymmetric.pth",
            "checkpoints/frequency_expert/best_model.pth",
        ],
        device,
    )
    print(f"Texture checkpoint: {texture_path}")
    print(f"Frequency checkpoint: {frequency_path}")

    val_tex_logits, val_freq_logits, val_labels = collect_logits(texture, frequency, val_loader, device)

    t_live_range = np.round(np.arange(0.9, 0.981, 0.01), 3)
    t_spoof_range = np.round(np.arange(0.9, 0.981, 0.01), 3)
    f_spoof_range = np.round(np.arange(0.95, 0.996, 0.005), 3)
    best = grid_search_thresholds(val_tex_logits, val_freq_logits, val_labels, t_live_range, t_spoof_range, f_spoof_range)
    print("\nBest thresholds (validated on Replay train split):")
    print(f"  t_live={best['t_live']}, t_spoof={best['t_spoof']}, f_spoof={best['f_spoof']}")
    print(f"  FAR={best['metrics']['far']*100:.2f}%, FRR={best['metrics']['frr']*100:.2f}%, HTER={best['metrics']['hter']*100:.2f}%")
    print("  Selection priority: FAR → FRR → HTER (break ties in this order).")

    test_dataset = ReplayAttackDataset(
        data_root,
        split="test",
        transform=raw_transforms(),
        raw_transform=raw_transforms(),
        freq_crop=freq_crop_flag,
    )
    log_crop_statistics(test_dataset, "Replay test (freq_crop stats)")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    metrics = evaluate_on_loader(texture, frequency, test_loader, device, thresholds=best)

    print("\n=== Final Test Metrics (Replay test) ===")
    for name, values in metrics.items():
        print(f"{name.title():<12} Acc={values['accuracy']*100:6.2f}% FAR={values['far']*100:6.2f}% FRR={values['frr']*100:6.2f}% HTER={values['hter']*100:6.2f}% FP={values['fp']} FN={values['fn']}")


if __name__ == "__main__":
    main()
