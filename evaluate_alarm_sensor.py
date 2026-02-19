import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.alarm_sensor_system import AlarmSensorSystem
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
    raise FileNotFoundError("Missing checkpoint: " + ", ".join(checkpoints))


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
    hter = (far + frr) / 2
    return {
        "live": live,
        "spoof": spoof,
        "overall": overall,
        "far": far,
        "frr": frr,
        "fp": fp,
        "fn": fn,
    }


def evaluate_threshold(threshold, alarm_system, texture, frequency, loader, device):
    alarm_system.alarm_threshold = threshold
    alarm_system.eval()
    texture.eval()
    frequency.eval()
    all_preds = []
    all_labels = []
    alarm_count = 0
    correct_alarms = 0
    with torch.inference_mode():
        for batch in loader:
            frame = batch["frame"].to(device)
            freq_raw = batch["freq_raw"].to(device)
            labels = batch["label"].to(device)
            tex_out = texture(frame)
            freq_out = frequency(freq_raw)
            f_probs = F.softmax(freq_out, dim=1)
            alarms = f_probs[:, 1] > threshold
            alarm_count += alarms.sum().item()
            correct_alarms += ((alarms) & (labels == 1)).sum().item()
            preds = alarm_system(frame).argmax(1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    alarm_precision = correct_alarms / alarm_count if alarm_count else 0.0
    alarm_rate = alarm_count / len(all_labels)
    return metrics, alarm_rate, alarm_precision


def evaluate_texture(texture, loader, device):
    texture.eval()
    preds = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            frame = batch["frame"].to(device)
            labels.append(batch["label"])
            preds.append(texture(frame).argmax(1).cpu())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    return compute_metrics(preds, labels)


def main():
    device = get_device()
    data_root = os.path.join("data", "replay-attack", "datasets", "fas_pure_data", "Idiap-replayattack")
    dataset = ReplayAttackDataset(
        data_root, split="test", transform=raw_transforms(), raw_transform=raw_transforms(), freq_crop=True
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    texture = TextureExpert().to(device)
    frequency = FrequencyExpert().to(device)
    load_model(
        texture,
        [
            "checkpoints/cross/texture.pth",
            "checkpoints/texture_expert/best_model.pth",
        ],
        device,
    )
    load_model(
        frequency,
        [
            "checkpoints/cross/frequency_asymmetric.pth",
            "checkpoints/frequency_expert/best_model.pth",
        ],
        device,
    )

    alarm_system = AlarmSensorSystem().to(device)
    alarm_system.texture.load_state_dict(texture.state_dict())
    alarm_system.frequency.load_state_dict(frequency.state_dict())

    print("=== Alarm Sensor System (Frequency as Trigger) ===")
    print(f"{'Threshold':<10} {'Live':>7} {'Spoof':>7} {'Overall':>8} {'FAR':>7} {'FRR':>7} {'Notes':>8}")
    thresholds = [0.8, 0.85, 0.9, 0.95, 0.99]
    best_threshold = None
    best_balance = 1.0
    alarm_stats = {}
    for thr in thresholds:
        metrics, rate, precision = evaluate_threshold(thr, alarm_system, texture, frequency, loader, device)
        note = ""
        if thr == 0.9:
            note = "Recommended"
        print(
            f"{thr:<10.2f} {metrics['live']*100:6.2f}% {metrics['spoof']*100:6.2f}% {metrics['overall']*100:7.2f}% "
            f"{metrics['far']*100:6.2f}% {metrics['frr']*100:6.2f}% {note:>8}"
        )
        alarm_stats[thr] = (rate, precision)
        balance = abs(metrics["far"] - metrics["frr"])
        if balance < best_balance:
            best_balance = balance
            best_threshold = thr

    base_metrics = evaluate_texture(texture, loader, device)
    print(
        f"{'Texture':<10} {base_metrics['live']*100:6.2f}% {base_metrics['spoof']*100:6.2f}% {base_metrics['overall']*100:7.2f}% "
        f"{base_metrics['far']*100:6.2f}% {base_metrics['frr']*100:6.2f}% {'Baseline':>8}"
    )

    best_rate, best_precision = alarm_stats[best_threshold]
    print("\nAnalysis:")
    print(f"- Best threshold: {best_threshold:.2f} balances FAR/FRR.")
    print(f"- Frequency alarms triggered: {best_rate*100:.2f}% of samples.")
    print(f"- Alarm precision (correct spoof): {best_precision*100:.2f}%")


if __name__ == "__main__":
    main()
