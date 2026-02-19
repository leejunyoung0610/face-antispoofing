import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve

from evaluate_cross_dataset import ReplayAttackDataset
from models.baseline import BaselineModel
from models.final_system import FinalSystem
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


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = fpr[idx]
    return eer * 100


def metrics_from_preds(preds: torch.Tensor, labels: torch.Tensor):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
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
    hter = (far + frr) / 2
    return live_acc, spoof_acc, overall, hter


def eval_baseline(model, loader, device):
    all_preds = []
    all_labels = []
    all_logits = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["frame"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_preds.append(logits.argmax(dim=1))
            all_labels.append(y)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    eer = compute_eer(labels.cpu().numpy(), probs)
    live_acc, spoof_acc, overall, hter = metrics_from_preds(preds, labels)
    return live_acc, spoof_acc, overall, hter, eer


def eval_raw_classifier(model, loader, device):
    all_preds = []
    all_labels = []
    all_logits = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["raw"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            all_logits.append(logits.detach().cpu())
            all_preds.append(logits.argmax(dim=1))
            all_labels.append(y)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    eer = compute_eer(labels.cpu().numpy(), probs)
    live_acc, spoof_acc, overall, hter = metrics_from_preds(preds, labels)
    return live_acc, spoof_acc, overall, hter, eer


def eval_final_system(system: FinalSystem, loader, device):
    all_preds = []
    all_labels = []
    all_logits = []
    system.eval()
    with torch.inference_mode():
        for batch in loader:
            x = batch["raw"].to(device)
            y = batch["label"].to(device)
            combined, _ = system(x)
            all_logits.append(combined.detach().cpu())
            all_preds.append(combined.argmax(dim=1))
            all_labels.append(y)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    eer = compute_eer(labels.cpu().numpy(), probs)
    live_acc, spoof_acc, overall, hter = metrics_from_preds(preds, labels)
    return live_acc, spoof_acc, overall, hter, eer


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline/Experts/2-Expert final system.")
    parser.add_argument(
        "--data_root",
        default="data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
        help="Replay-Attack root.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    ds = ReplayAttackDataset(args.data_root, split="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    baseline = BaselineModel().to(device)
    baseline_ckpt = (
        "checkpoints/cross/baseline_asymmetric.pth"
        if os.path.exists("checkpoints/cross/baseline_asymmetric.pth")
        else "checkpoints/cross/baseline.pth"
        if os.path.exists("checkpoints/cross/baseline.pth")
        else "checkpoints/best_model.pth"
    )
    baseline.load_state_dict(torch.load(baseline_ckpt, map_location=device))

    texture = TextureExpert().to(device)
    texture_ckpt = (
        "checkpoints/cross/texture.pth"
        if os.path.exists("checkpoints/cross/texture.pth")
        else "checkpoints/texture_expert/best_model.pth"
    )
    texture.load_state_dict(torch.load(texture_ckpt, map_location=device))

    frequency = FrequencyExpert().to(device)
    frequency_ckpt = (
        "checkpoints/cross/frequency_asymmetric.pth"
        if os.path.exists("checkpoints/cross/frequency_asymmetric.pth")
        else "checkpoints/cross/frequency.pth"
        if os.path.exists("checkpoints/cross/frequency.pth")
        else "checkpoints/frequency_expert/best_model.pth"
    )
    frequency.load_state_dict(torch.load(frequency_ckpt, map_location=device))

    system = FinalSystem().to(device)
    # 2-expert 시스템은 내부 expert weights를 현재 로드한 것으로 맞춤
    system.texture.load_state_dict(texture.state_dict())
    system.frequency.load_state_dict(frequency.state_dict())

    print("=== Final System Evaluation ===")
    print(f"{'':20} {'Live':>8} {'Spoof':>8} {'Overall':>8} {'HTER':>8} {'EER':>8}")

    live, spoof, overall, hter, eer = eval_baseline(baseline, loader, device)
    print(f"{'Baseline (Ref):':20} {live*100:7.2f}% {spoof*100:7.2f}% {overall*100:7.2f}% {hter*100:7.2f}% {eer:7.2f}%")

    live, spoof, overall, hter, eer = eval_raw_classifier(texture, loader, device)
    print(f"{'Texture:':20} {live*100:7.2f}% {spoof*100:7.2f}% {overall*100:7.2f}% {hter*100:7.2f}% {eer:7.2f}%")

    live, spoof, overall, hter, eer = eval_raw_classifier(frequency, loader, device)
    print(f"{'Frequency:':20} {live*100:7.2f}% {spoof*100:7.2f}% {overall*100:7.2f}% {hter*100:7.2f}% {eer:7.2f}%")

    live, spoof, overall, hter, eer = eval_final_system(system, loader, device)
    print(f"{'2-Expert System:':20} {live*100:7.2f}% {spoof*100:7.2f}% {overall*100:7.2f}% {hter*100:7.2f}% {eer:7.2f}%")


if __name__ == "__main__":
    main()

