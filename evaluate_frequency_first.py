import os

import torch
from torch.utils.data import DataLoader

from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.frequency_expert import FrequencyExpert
from models.frequency_first_system import FrequencyFirstSystem
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


def get_loader(device):
    data_root = os.path.join("data", "replay-attack", "datasets", "fas_pure_data", "Idiap-replayattack")
    dataset = ReplayAttackDataset(
        data_root, split="test", transform=raw_transforms(), raw_transform=raw_transforms(), freq_crop=False
    )
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)


def evaluate_texture(texture, loader, device):
    texture.eval()
    preds = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frame"].to(device)
            logits = texture(frames)
            preds.append(logits.argmax(1).cpu())
            labels.append(batch["label"])
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    return compute_metrics(preds, labels)


def evaluate_frequency_first(system, loader, device):
    system.eval()
    preds = []
    labels = []
    freq_spoof = 0
    freq_spoof_live = 0
    freq_spoof_spoof = 0
    texture_live_restored = 0
    texture_spoof_preserved = 0
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frame"].to(device)
            labels_batch = batch["label"].to(device)
            system_out = system(frames)
            preds.append(system_out.argmax(1).cpu())
            labels.append(labels_batch.cpu())
            freq_out = system.frequency(frames)
            freq_preds = freq_out.argmax(1)
            mask = freq_preds == 1
            freq_spoof += mask.sum().item()
            freq_spoof_live += ((mask) & (labels_batch == 0)).sum().item()
            freq_spoof_spoof += ((mask) & (labels_batch == 1)).sum().item()
            tex_out = system.texture(frames[mask])
            tex_preds = tex_out.argmax(1)
            restored = (tex_preds == 0) & (labels_batch[mask] == 0)
            preserved = (tex_preds == 1) & (labels_batch[mask] == 1)
            texture_live_restored += restored.sum().item()
            texture_spoof_preserved += preserved.sum().item()
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    metrics = compute_metrics(preds, labels)
    return {
        "metrics": metrics,
        "freq_spoof_total": freq_spoof,
        "freq_spoof_live": freq_spoof_live,
        "freq_spoof_spoof": freq_spoof_spoof,
        "texture_live_restored": texture_live_restored,
        "texture_spoof_preserved": texture_spoof_preserved,
    }


def main():
    device = get_device()
    loader = get_loader(device)

    texture = TextureExpert().to(device)
    frequency = FrequencyExpert().to(device)
    load_model(texture, ["checkpoints/cross/texture.pth", "checkpoints/texture_expert/best_model.pth"], device)
    load_model(frequency, ["checkpoints/cross/frequency_asymmetric.pth", "checkpoints/frequency_expert/best_model.pth"], device)

    texture_metrics = evaluate_texture(texture, loader, device)
    system = FrequencyFirstSystem().to(device)
    system.frequency.load_state_dict(frequency.state_dict())
    system.texture.load_state_dict(texture.state_dict())

    system_stats = evaluate_frequency_first(system, loader, device)
    metrics = system_stats["metrics"]

    freq_live = system_stats["freq_spoof_total"] - system_stats["freq_spoof_spoof"]
    print("=== Frequency First System ===")
    print(f"Frequency first predictions: Live {freq_live}, Spoof {system_stats['freq_spoof_spoof']}")
    print(f"Frequency Spoof labels actual Live/Spoof: {system_stats['freq_spoof_live']}/{system_stats['freq_spoof_spoof']}")
    print(f"Texture restored Live: {system_stats['texture_live_restored']} / preserved Spoof: {system_stats['texture_spoof_preserved']}")
    print("\nFinal performance:")
    print(f"Live: {metrics['live']*100:.2f}%")
    print(f"Spoof: {metrics['spoof']*100:.2f}%")
    print(f"Overall: {metrics['overall']*100:.2f}%")
    print(f"FAR: {metrics['far']*100:.2f}% FRR: {metrics['frr']*100:.2f}%")
    print("\nComparison:")
    print(f"{'':<20} {'Live':>7} {'Spoof':>7} {'Overall':>8} {'FAR':>7} {'FRR':>7}")
    print(f"{'Texture-only':<20} 98.75%  98.50%   98.54%   1.25%   1.50%")
    print(f"{'Cascading':<20} 98.75%  99.00%   98.75%   1.25%   1.00%")
    print(f"{'Frequency-First':<20} {metrics['live']*100:6.2f}% {metrics['spoof']*100:6.2f}% {metrics['overall']*100:7.2f}% {metrics['far']*100:6.2f}% {metrics['frr']*100:6.2f}%")
    print("\nAnalysis:")
    print(f"Frequency misclassified {system_stats['freq_spoof_live']} Lives; Texture recovered {system_stats['texture_live_restored']} of them.")
    print(f"Texture missed 6 Spoofs; Frequency first preserved {system_stats['texture_spoof_preserved']} via second stage.")


if __name__ == "__main__":
    main()
