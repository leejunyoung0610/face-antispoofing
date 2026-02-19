import os

import torch
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
    return {"live": live, "spoof": spoof, "overall": overall}


def main():
    device = get_device()
    data_root = os.path.join("data", "replay-attack", "datasets", "fas_pure_data", "Idiap-replayattack")
    dataset = ReplayAttackDataset(data_root, split="test", transform=raw_transforms(), raw_transform=raw_transforms(), freq_crop=False)
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

    final_preds = []
    labels_tensor = []
    freq_spoof_labels = []
    texture_recover_live = 0
    texture_keep_spoof = 0

    with torch.inference_mode():
        for batch in loader:
            raw = batch["raw"].to(device)
            frame = batch["frame"].to(device)
            labels = batch["label"].to(device)
            freq_logits = frequency(raw)
            freq_preds = freq_logits.argmax(1)
            tex_logits = texture(frame)
            tex_preds = tex_logits.argmax(1)
            # record freq spoof cases
            mask = freq_preds == 1
            freq_spoof_labels.extend(labels[mask].tolist())
            # track texture live/spoof counts among freq==1
            texture_recover_live += ((mask) & (tex_preds == 0) & (labels == 0)).sum().item()
            texture_keep_spoof += ((mask) & (tex_preds == 1) & (labels == 1)).sum().item()
            # final decision: freq live -> live, else texture result
            combined = freq_preds.clone()
            combined[mask] = tex_preds[mask]
            final_preds.append(combined.cpu())
            labels_tensor.append(labels.cpu())

    final_preds = torch.cat(final_preds)
    labels_tensor = torch.cat(labels_tensor)
    freq_spoof_labels = torch.tensor(freq_spoof_labels)
    metrics = compute_metrics(final_preds, labels_tensor)
    freq_spoof_total = freq_spoof_labels.numel()
    freq_spoof_live = (freq_spoof_labels == 0).sum().item()
    freq_spoof_spoof = (freq_spoof_labels == 1).sum().item()

    print("=== Frequency-First Verification ===")
    print(f"Frequency predicted Spoof count: {freq_spoof_total}")
    print(f"- Actual Spoof among them: {freq_spoof_spoof}")
    print(f"- Actual Live among them: {freq_spoof_live}")
    print(f"Texture reevaluation restored {texture_recover_live} Lives and kept {texture_keep_spoof} Spoofs.")
    print("\nFinal performance:")
    print(f"Live: {metrics['live']*100:6.2f}%")
    print(f"Spoof: {metrics['spoof']*100:6.2f}%")
    print(f"Overall: {metrics['overall']*100:6.2f}%")
    print("\nvs Texture-only baseline:")
    print("Live: 98.75%")
    print("Spoof: 98.50%")
    print("Overall: 98.54%")


if __name__ == "__main__":
    main()
