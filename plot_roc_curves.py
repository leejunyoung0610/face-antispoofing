import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader

from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.adaptive_system import AdaptiveSystem
from models.alarm_sensor_system import AlarmSensorSystem
from models.baseline import BaselineModel
from models.frequency_expert import FrequencyExpert
from models.frequency_first_system import FrequencyFirstSystem
from models.texture_expert import TextureExpert
from utils.cascading import cascading_predict


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_state(model, path, device, expected_prefix=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sd = torch.load(path, map_location=device)
    keys = list(sd.keys())[:10]
    print(f"Loading {os.path.basename(path)} keys preview: {keys}")
    prefix = keys[0].split('.')[0]
    if expected_prefix and expected_prefix not in keys[0]:
        raise AssertionError(f"Expected '{expected_prefix}' in checkpoint keys but got '{keys[0]}'")
    model.load_state_dict(sd)


def cascade_logits(t_logits, f_logits, device):
    p_tex = F.softmax(t_logits, dim=1)
    p_freq = F.softmax(f_logits, dim=1)
    final = torch.zeros_like(t_logits)
    for idx in range(t_logits.size(0)):
        t_spoof = p_tex[idx, 1]
        t_live = p_tex[idx, 0]
        if t_spoof > 0.95:
            final[idx] = torch.tensor([-100.0, 100.0], device=device)
        elif t_live > 0.95:
            final[idx] = torch.tensor([100.0, -100.0], device=device)
        else:
            if p_freq[idx, 1] > 0.95:
                final[idx] = torch.tensor([-100.0, 100.0], device=device)
            else:
                final[idx] = t_logits[idx]
    return final


def capture_scores(loader, device, texture_model, frequency_model, baseline_model, adaptive_system, frequency_first, alarm_system):
    labels = []
    scores = {
        "Texture": [],
        "Frequency": [],
        "Baseline": [],
        "Cascading": [],
        "Adaptive": [],
        "Frequency-First": [],
        "Alarm": [],
    }
    texture_model.eval()
    frequency_model.eval()
    baseline_model.eval()
    adaptive_system.eval()
    frequency_first.eval()
    alarm_system.eval()

    with torch.inference_mode():
        for batch in loader:
            raw = batch["raw"].to(device)
            frame = batch["frame"].to(device)
            label = batch["label"].to(device)
            labels.append(label.cpu())

            t_logits = texture_model(raw)
            f_logits = frequency_model(raw)
            b_logits = baseline_model(frame)

            scores["Texture"].extend(F.softmax(t_logits, dim=1)[:, 1].cpu().numpy())
            scores["Frequency"].extend(F.softmax(f_logits, dim=1)[:, 1].cpu().numpy())
            scores["Baseline"].extend(F.softmax(b_logits, dim=1)[:, 1].cpu().numpy())

            cascade_out = cascade_logits(t_logits, f_logits, device)
            scores["Cascading"].extend(F.softmax(cascade_out, dim=1)[:, 1].cpu().numpy())

            adapt_out, _ = adaptive_system(raw)
            scores["Adaptive"].extend(F.softmax(adapt_out, dim=1)[:, 1].cpu().numpy())

            freq_first_out = frequency_first(raw)
            if isinstance(freq_first_out, tuple):
                freq_first_out = freq_first_out[0]
            scores["Frequency-First"].extend(F.softmax(freq_first_out, dim=1)[:, 1].cpu().numpy())

            alarm_out = alarm_system(raw)
            if isinstance(alarm_out, tuple):
                alarm_out = alarm_out[0]
            scores["Alarm"].extend(F.softmax(alarm_out, dim=1)[:, 1].cpu().numpy())

    labels = torch.cat(labels).cpu().numpy()
    return labels, scores


def compute_far_tar(scores, labels):
    thresholds = np.unique(scores)
    far = []
    tar = []
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tar.append(tp / (tp + fn) if (tp + fn) else 0.0)
        far.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return np.array(far), np.array(tar)


def compute_eer(scores, labels):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return ((fpr[idx] + fnr[idx]) / 2) * 100


def main():
    device = get_device()
    texture_model = TextureExpert().to(device)
    frequency_model = FrequencyExpert().to(device)
    baseline_model = BaselineModel().to(device)
    load_state(texture_model, "checkpoints/cross/texture.pth", device, expected_prefix="backbone.conv_stem")
    load_state(frequency_model, "checkpoints/cross/frequency_asymmetric.pth", device, expected_prefix="backbone.conv_stem")
    load_state(baseline_model, "checkpoints/cross/baseline.pth", device, expected_prefix="backbone.conv1")

    adaptive_system = AdaptiveSystem(confidence_threshold=0.8).to(device)
    load_state(adaptive_system.texture, "checkpoints/cross/texture.pth", device, expected_prefix="backbone.conv_stem")
    load_state(adaptive_system.frequency, "checkpoints/cross/frequency_asymmetric.pth", device, expected_prefix="backbone.conv_stem")

    frequency_first = FrequencyFirstSystem().to(device)
    load_state(frequency_first.texture, "checkpoints/cross/texture.pth", device, expected_prefix="backbone.conv_stem")
    load_state(frequency_first.frequency, "checkpoints/cross/frequency_asymmetric.pth", device, expected_prefix="backbone.conv_stem")

    alarm_system = AlarmSensorSystem(alarm_threshold=0.99).to(device)
    load_state(alarm_system.texture, "checkpoints/cross/texture.pth", device, expected_prefix="backbone.conv_stem")
    load_state(alarm_system.frequency, "checkpoints/cross/frequency_asymmetric.pth", device, expected_prefix="backbone.conv_stem")

    loader = DataLoader(
        ReplayAttackDataset(
            "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
            split="test",
            transform=raw_transforms(),
            raw_transform=raw_transforms(),
            freq_crop=False,
        ),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    labels, scores = capture_scores(
        loader,
        device,
        texture_model,
        frequency_model,
        baseline_model,
        adaptive_system,
        frequency_first,
        alarm_system,
    )

    roc_data = {}
    for name, sc in scores.items():
        far, tar = compute_far_tar(np.array(sc), labels)
        roc_data[name] = {
            "far": far,
            "tar": tar,
            "auc": auc(far, tar),
            "eer": compute_eer(np.array(sc), labels),
        }

    plt.figure(figsize=(12, 8))
    for name in ["Baseline", "Texture", "Frequency"]:
        data = roc_data[name]
        plt.plot(data["far"], data["tar"], label=f"{name} (AUC={data['auc']:.4f})", linewidth=2)
        frr = 1 - data["tar"]
        eer_idx = np.nanargmin(np.abs(data["far"] - frr))
        plt.plot(data["far"][eer_idx], data["tar"][eer_idx], "o", markersize=6)
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.5000)", linewidth=1)
    plt.xlabel("FAR")
    plt.ylabel("TAR")
    plt.title("ROC Curves: Face Anti-Spoofing Systems Comparison")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.3)
    plt.ylim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig("roc_curves.png", dpi=300, bbox_inches="tight")

    print("=== ROC Analysis Summary ===")
    for name in ["Baseline", "Texture", "Frequency"]:
        data = roc_data[name]
        print(
            f"{name:<15} AUC={data['auc']:.4f} EER={data['eer']:.2f}% "
            f"Best(FAR={data['far'][np.nanargmin(np.abs(data['far'] - (1 - data['tar'])))]*100:.2f} "
            f"TAR={data['tar'][np.nanargmin(np.abs(data['far'] - (1 - data['tar'])))]*100:.2f})"
        )

    def tuned_metrics(name):
        sc = scores[name]
        far, tar = compute_far_tar(np.array(sc), labels)
        frr = 1 - tar
        idx = np.nanargmin(np.abs(far - frr))
        hter = (far[idx] + frr[idx]) / 2
        return far[idx] * 100, frr[idx] * 100, hter * 100

    print("\n=== Cascading Family Threshold Metrics ===")
    for name in ["Cascading", "Adaptive", "Frequency-First", "Alarm"]:
        far_val, frr_val, hter_val = tuned_metrics(name)
        print(
            f"{name:<15} FAR={far_val:5.2f}% FRR={frr_val:5.2f}% HTER={hter_val:5.2f}%"
        )


if __name__ == "__main__":
    main()
