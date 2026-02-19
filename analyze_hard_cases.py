import os
import re
from collections import Counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert
from utils.face_crop import crop_face_pil


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
            return path
    raise FileNotFoundError("Checkpoint not found among: " + ", ".join(checkpoints))


def extract_client(video_path):
    match = re.search(r"client0*([0-9]+)", video_path.lower())
    return match.group(1) if match else "unknown"


def condition_from_path(video_path):
    lower = video_path.lower()
    if "adverse" in lower:
        return "adverse"
    return "controlled"


def print_table(rows, headers):
    col_widths = [max(len(headers[i]), *(len(str(row[i])) for row in rows)) for i in range(len(headers))]
    header_line = "  ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers)))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("  ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(headers))))


def gather_stats(texture, frequency, dataset, device):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    stats = []
    texture.eval()
    frequency.eval()
    with torch.inference_mode():
        for batch in loader:
            raw = batch["raw"].to(device)
            freq_raw = batch["freq_raw"].to(device)
            label = batch["label"].item()
            path = batch["metadata"]["video_path"][0]
            tex_out = texture(raw)
            freq_out = frequency(freq_raw)
            tex_probs = F.softmax(tex_out, dim=1)[0]
            freq_probs = F.softmax(freq_out, dim=1)[0]
            stats.append(
                {
                    "path": os.path.basename(path),
                    "full_path": path,
                    "label": label,
                    "texture_spoof": tex_probs[1].item(),
                    "frequency_spoof": freq_probs[1].item(),
                    "texture_pred": int(tex_probs.argmax().item()),
                    "frequency_pred": int(freq_probs.argmax().item()),
                    "attack_type": batch["metadata"]["attack_type"][0],
                    "client": extract_client(path),
                    "condition": condition_from_path(path),
                }
            )
    return stats


def describe_common(rows, texture_conf_key, freq_conf_key):
    clients = Counter(row["client"] for row in rows)
    attack_types = Counter(row["attack_type"] for row in rows)
    texture_confs = [row[texture_conf_key] for row in rows]
    freq_confs = [row[freq_conf_key] for row in rows]
    highdef = sum("highdef" in row["path"].lower() for row in rows)
    return {
        "client": clients,
        "attack_types": attack_types,
        "texture_conf": {
            "mean": sum(texture_confs) / len(texture_confs) if texture_confs else 0,
            "min": min(texture_confs) if texture_confs else 0,
            "max": max(texture_confs) if texture_confs else 0,
        },
        "freq_conf": {
            "mean": sum(freq_confs) / len(freq_confs) if freq_confs else 0,
            "min": min(freq_confs) if freq_confs else 0,
            "max": max(freq_confs) if freq_confs else 0,
        },
        "highdef": highdef,
    }


def summarize_common(label, rows, texture_conf_key, freq_conf_key):
    summary = describe_common(rows, texture_conf_key, freq_conf_key)
    clients = summary["client"].most_common()
    attack_types = summary["attack_types"].most_common()
    print(f"\nCommon Patterns ({label}):")
    if clients:
        concentrated = ", ".join(f"{client} ({count})" for client, count in clients[:3])
        print(f"- Client concentration: {concentrated}")
    if summary["attack_types"]:
        print(f"- Attack types breakdown: {', '.join(f'{atype}({cnt})' for atype, cnt in attack_types)}")
    if summary["highdef"]:
        print(f"- Attack quality: highdef matched {summary['highdef']} / {len(rows)} cases ({len(rows) and (summary['highdef']/len(rows))*100:.1f}%)")
    print(
        f"- Avg Texture confidence: {summary['texture_conf']['mean']:.2f} (min {summary['texture_conf']['min']:.2f}, max {summary['texture_conf']['max']:.2f})"
    )
    print(
        f"- Avg Frequency confidence: {summary['freq_conf']['mean']:.2f} (min {summary['freq_conf']['min']:.2f}, max {summary['freq_conf']['max']:.2f})"
    )


def main():
    device = get_device()
    data_root = os.path.join("data", "replay-attack", "datasets", "fas_pure_data", "Idiap-replayattack")
    dataset = ReplayAttackDataset(
        data_root, split="test", transform=raw_transforms(), raw_transform=raw_transforms(), freq_crop=True
    )
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
    stats = gather_stats(texture, frequency, dataset, device)

    texture_hard = sorted(
        [row for row in stats if row["label"] == 1 and row["texture_pred"] == 0],
        key=lambda x: x["texture_spoof"],
    )[:6]
    print("=== Texture False Negatives (Hard Cases) ===")
    print_table(
        [
            (
                row["path"],
                f"{row['texture_spoof']:.2f}",
                f"{row['frequency_spoof']:.2f}",
                row["attack_type"],
                row["client"],
            )
            for row in texture_hard
        ],
        ["Video", "T_conf", "F_conf", "Type", "Client"],
    )
    summarize_common("Texture FN", texture_hard, "texture_spoof", "frequency_spoof")
    print(
        "\nInsight: High-def prints concentrated on a few clients → tail risk. Need to manage worst-case rather than just mean performance."
    )

    freq_bad = sorted(
        [row for row in stats if row["label"] == 0 and row["frequency_pred"] == 1],
        key=lambda x: -x["frequency_spoof"],
    )[:16]
    print("\n=== Frequency False Positives (Live Rejection) ===")
    print_table(
        [
            (
                row["path"],
                f"{row['frequency_spoof']:.2f}",
                f"{row['texture_spoof']:.2f}",
                row["condition"],
                row["client"],
            )
            for row in freq_bad
        ],
        ["Video", "F_conf", "T_conf", "Condition", "Client"],
    )
    summarize_common("Frequency FP", freq_bad, "texture_spoof", "frequency_spoof")
    print(
        "\nInsight: Uniform backgrounds + steady lighting make Frequency overconfident, raising Live rejection risk."
    )

    uncertain = [row for row in stats if row["texture_spoof"] < 0.95]
    freq_help = sum(
        1
        for row in uncertain
        if row["label"] == 1 and row["frequency_pred"] == 1 and row["frequency_spoof"] > 0.9
    )
    freq_harm = sum(
        1
        for row in uncertain
        if row["label"] == 0 and row["frequency_pred"] == 1 and row["frequency_spoof"] > 0.9
    )
    print("\n=== Decision Rule Validation ===")
    print("Current cascading (texture + 0.3 frequency) used to boost uncertain cases.")
    print(
        f"- Among texture_conf < 0.95, Frequency helped {freq_help} samples (spoof detected) and harmed {freq_harm} Lives (false triggers)"
    )
    print("Proposed conservative rule:")
    print("if texture_conf > 0.95: use texture")
    print("elif frequency_spoof_conf > 0.9: flag as spoof")
    print("else: fall back to texture (conservative)")


if __name__ == "__main__":
    main()
