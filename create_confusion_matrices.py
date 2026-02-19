import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from PIL import Image

from models.baseline import BaselineModel
from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert
from models.gating import GatingModule
from analyze_attack_types import MultiExpertWrapper


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_frame(path, device):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    preds = []
    if total == 0:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, min(10, total), dtype=int)
    outputs = []
    with torch.inference_mode():
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            tensor = transform(frame).unsqueeze(0).to(device)
            outputs.append(tensor)
    cap.release()
    if not outputs:
        return None
    return torch.cat(outputs, dim=0)


def evaluate_model(name, model, dataset, device):
    model.to(device).eval()
    y_true = []
    y_pred = []
    details = []
    for video_path, label in dataset:
        frames = load_frame(video_path, device)
        if frames is None:
            continue
        with torch.inference_mode():
            if getattr(model, "multi_expert", False):
                out = model(frames, frames)
            else:
                out = model(frames)
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.softmax(out, dim=1)
            mean_confidence, _ = probs.max(dim=1)
            avg_conf = mean_confidence.mean().item()
            pred = out.argmax(1)
            majority = torch.mode(pred.cpu(), dim=0).values.item()
        y_true.append(label)
        y_pred.append(majority)
        details.append((video_path, avg_conf, label, majority))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    totals = cm.sum(axis=1)
    live_prec = cm[0, 0] / cm[:, 0].sum() if cm[:, 0].sum() else 0
    live_rec = cm[0, 0] / totals[0] if totals[0] else 0
    spoof_prec = cm[1, 1] / cm[:, 1].sum() if cm[:, 1].sum() else 0
    spoof_rec = cm[1, 1] / totals[1] if totals[1] else 0
    far = cm[0, 1] / totals[0] if totals[0] else 0
    frr = cm[1, 0] / totals[1] if totals[1] else 0
    return {
        "name": name,
        "cm": cm,
        "metrics": {
            "precision_live": live_prec,
            "recall_live": live_rec,
            "precision_spoof": spoof_prec,
            "recall_spoof": spoof_rec,
            "far": far,
            "frr": frr,
        },
        "details": details,
    }


def print_result(res):
    print(f"=== {res['name']} ===")
    print("Confusion Matrix:")
    print("               Predicted Live  Predicted Spoof")
    print(f"Actual Live         {res['cm'][0,0]}               {res['cm'][0,1]}")
    print(f"Actual Spoof        {res['cm'][1,0]}               {res['cm'][1,1]}")
    print("\nMetrics:")
    for k, v in res["metrics"].items():
        print(f"- {k.replace('_',' ').title()}: {v*100:.2f}%")


def print_failures(res, typ):
    print(f"\n=== {typ} ===")
    failures = [
        (path, conf, act, pred)
        for path, conf, act, pred in res["details"]
        if (act == 0 and pred == 1 and typ == "False Positives (Live → Spoof)")
        or (act == 1 and pred == 0 and typ == "False Negatives (Spoof → Live)")
    ]
    failures.sort(key=lambda x: x[1], reverse=True)
    for entry in failures[:10]:
        print(f"- Video: {entry[0]}, Confidence: {entry[1]:.2f}, Actual: {entry[2]}, Predicted: {entry[3]}")


def plot_confusion(confusion_matrices):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, res in zip(axes, confusion_matrices):
        cm = res["cm"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(res["name"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    plt.savefig("confusion_matrices.png")


def prepare_dataset():
    msu_dir = "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack/test"
    live = [
        os.path.join(msu_dir, "real", f)
        for f in os.listdir(os.path.join(msu_dir, "real"))
        if f.lower().endswith((".mov", ".mp4"))
    ]
    attack = []
    attack_root = os.path.join(msu_dir, "attack")
    for sub in os.listdir(attack_root):
        subpath = os.path.join(attack_root, sub)
        if os.path.isdir(subpath):
            attack += [
                os.path.join(subpath, f)
                for f in os.listdir(subpath)
                if f.lower().endswith((".mov", ".mp4"))
            ]
    dataset = []
    for path in live:
        dataset.append((path, 0))
    for path in attack:
        dataset.append((path, 1))
    return dataset


def main():
    device = get_device()
    print(f"Using device: {device}")

    baseline = BaselineModel().to(device)
    baseline.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    texture = TextureExpert().to(device)
    texture.load_state_dict(torch.load("checkpoints/cross/texture.pth", map_location=device))
    frequency = FrequencyExpert().to(device)
    frequency.load_state_dict(
        torch.load("checkpoints/cross/frequency_asymmetric.pth", map_location=device)
    )

    gating = GatingModule(num_experts=3)
    gating_path = "checkpoints/multi_expert/best_gating.pth"
    if os.path.exists(gating_path):
        gating.load_state_dict(torch.load(gating_path, map_location=device))
    else:
        print(f"[WARN] {gating_path} not found, gating weights randomly initialized.")
    gating.to(device)
    multi_expert = MultiExpertWrapper(baseline, texture, frequency, gating).to(device)

    dataset = prepare_dataset()

    models = [
        ("Baseline (Reference)", baseline),
        ("Texture CNN", texture),
        ("Frequency CNN", frequency),
        ("2-Expert Static", multi_expert),
    ]

    results = []
    for name, model in models:
        res = evaluate_model(name, model, dataset, device)
        print_result(res)
        print_failures(res, "False Positives (Live → Spoof)")
        print_failures(res, "False Negatives (Spoof → Live)")
        results.append(res)

    plot_confusion(results)


if __name__ == "__main__":
    main()

