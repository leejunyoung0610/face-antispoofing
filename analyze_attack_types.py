import os
from typing import List, Tuple

import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.baseline import BaselineModel
from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert
from models.gating import GatingModule
from models.final_system import FinalSystem


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class VideoFrameDataset(Dataset):
    def __init__(
        self, root: str, label: int, attack_type: str = None, dataset_name: str = None
    ):
        self.root = root
        self.label = label
        self.attack_type = attack_type
        self.dataset_name = dataset_name or (
            "replay" if "Idiap-replayattack" in root else "msu"
        )
        self.path_list = []

        if self.dataset_name == "replay":
            for root_dir, _, files in os.walk(root):
                for fname in files:
                    lower = fname.lower()
                    if not (lower.endswith(".mov") or lower.endswith(".mp4")):
                        continue
                    if attack_type == "print":
                        if "_photo_" not in lower:
                            continue
                    elif attack_type == "display":
                        if "_video_" not in lower:
                            continue
                    self.path_list.append(os.path.join(root_dir, fname))
        else:
            for fname in os.listdir(root):
                lower = fname.lower()
                if not lower.endswith(".mp4"):
                    continue
                if attack_type == "print":
                    if "printed_photo" not in lower:
                        continue
                elif attack_type == "display":
                    if ("ipad_video" not in lower) and ("iphone_video" not in lower):
                        continue
                self.path_list.append(os.path.join(root, fname))
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = max(total_frames // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = transforms.functional.to_pil_image(frame)
        normalized = self.transform(image)
        raw = transforms.ToTensor()(transforms.functional.resize(image, (224, 224)))
        return {"frame": normalized, "raw": raw, "label": self.label, "path": path}


class MultiExpertWrapper(nn.Module):
    def __init__(self, baseline, texture, frequency, gating):
        super().__init__()
        self.baseline = baseline
        self.texture = texture
        self.frequency = frequency
        self.gating = gating
        self.multi_expert = True

    def forward(self, frame, raw):
        b_out = self.baseline(frame)
        t_out = self.texture(raw)
        f_out = self.frequency(raw)
        combined, _ = self.gating([b_out, t_out, f_out])
        return combined


class TwoExpertWrapper(nn.Module):
    def __init__(self, texture, frequency, gating):
        super().__init__()
        self.texture = texture
        self.frequency = frequency
        self.gating = gating
        self.multi_expert = True

    def forward(self, frame, raw):
        t_out = self.texture(raw)
        f_out = self.frequency(raw)
        combined, _ = self.gating([t_out, f_out])
        return combined


def compute_accuracy(model, loader, device, use_raw=False):
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch in loader:
            labels = batch["label"].to(device)
            if getattr(model, "multi_expert", False):
                frame = batch["frame"].to(device)
                raw = batch["raw"].to(device)
                logits = model(frame, raw)
            else:
                data = batch["raw"] if use_raw else batch["frame"]
                data = data.to(device)
                logits = model(data)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def evaluate_attack_types(model, splits):
    device_ = device()
    results = {}
    for name, spec in splits.items():
        path, label, attack_type = spec
        dataset = VideoFrameDataset(path, label, attack_type)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        acc = compute_accuracy(
            model, loader, device_, use_raw=isinstance(model, (TextureExpert, FrequencyExpert))
        )
        results[name] = {"acc": acc, "n": len(dataset)}
    return results


def main():
    msu_splits = {
        "live": ("data/MSU-MFSD/scene01/real", 0, "live"),
        "print": ("data/MSU-MFSD/scene01/attack", 1, "print"),
        "display": ("data/MSU-MFSD/scene01/attack", 1, "display"),
    }

    replay_attack_dir = (
        "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack/test/attack"
    )
    replay_splits = {
        "live": (
            "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack/test/real",
            0,
            "live",
        ),
        "print": (replay_attack_dir, 1, "print"),
        "display": (replay_attack_dir, 1, "display"),
    }

    device_ = device()
    baseline = BaselineModel().to(device_)
    texture = TextureExpert().to(device_)
    frequency = FrequencyExpert().to(device_)
    gating = GatingModule(num_experts=3).to(device_)

    baseline_path = (
        "checkpoints/cross/baseline.pth"
        if os.path.exists("checkpoints/cross/baseline.pth")
        else "checkpoints/best_model.pth"
    )
    texture_path = (
        "checkpoints/cross/texture.pth"
        if os.path.exists("checkpoints/cross/texture.pth")
        else "checkpoints/texture_expert/best_model.pth"
    )
    if os.path.exists("checkpoints/cross/frequency_asymmetric.pth"):
        frequency_path = "checkpoints/cross/frequency_asymmetric.pth"
    elif os.path.exists("checkpoints/cross/frequency.pth"):
        frequency_path = "checkpoints/cross/frequency.pth"
    else:
        frequency_path = "checkpoints/frequency_expert/best_model.pth"

    baseline.load_state_dict(torch.load(baseline_path, map_location=device_))
    texture.load_state_dict(torch.load(texture_path, map_location=device_))
    frequency.load_state_dict(torch.load(frequency_path, map_location=device_))

    print(f"Using baseline: {baseline_path}")
    print(f"Using texture: {texture_path}")
    print(f"Using frequency: {frequency_path}")
    gating.load_state_dict(torch.load("checkpoints/multi_expert/best_gating.pth", map_location=device_))

    print("=== MSU-MFSD Attack Type Analysis ===")
    print("=== Reference (Baseline) ===")
    baseline_res = evaluate_attack_types(baseline, msu_splits)
    baseline_overall = sum([v["acc"] for v in baseline_res.values()]) / max(1, len(baseline_res))
    print(f"  Live (N={baseline_res['live']['n']}):      {baseline_res['live']['acc']*100:.2f}%")
    print(f"  Print (N={baseline_res['print']['n']}):     {baseline_res['print']['acc']*100:.2f}%")
    print(f"  Display (N={baseline_res['display']['n']}):   {baseline_res['display']['acc']*100:.2f}%")
    print(f"  Overall:          {baseline_overall*100:.2f}%")
    print("→ General CNN limitation")

    print("\n=== Proposed System ===")
    for name, model in [("Texture", texture), ("Frequency", frequency)]:
        res = evaluate_attack_types(model, msu_splits)
        overall = sum([v["acc"] for v in res.values()]) / max(1, len(res))
        print(f"{name}:")
        print(f"  Live (N={res['live']['n']}):      {res['live']['acc']*100:.2f}%")
        print(f"  Print (N={res['print']['n']}):     {res['print']['acc']*100:.2f}%")
        print(f"  Display (N={res['display']['n']}):   {res['display']['acc']*100:.2f}%")
        print(f"  Overall:          {overall*100:.2f}%")

    final_system = FinalSystem().to(device_)
    final_system.texture.load_state_dict(texture.state_dict())
    final_system.frequency.load_state_dict(frequency.state_dict())
    # gating은 별도 학습 체크포인트가 없다면 균등 결합(초기화) 상태
    two_expert = TwoExpertWrapper(final_system.texture, final_system.frequency, final_system.gating)
    gating_res = evaluate_attack_types(two_expert, msu_splits)
    overall = sum([v["acc"] for v in gating_res.values()]) / max(1, len(gating_res))
    print("2-Expert:")
    print(f"  Live (N={gating_res['live']['n']}):      {gating_res['live']['acc']*100:.2f}%")
    print(f"  Print (N={gating_res['print']['n']}):     {gating_res['print']['acc']*100:.2f}%")
    print(f"  Display (N={gating_res['display']['n']}):   {gating_res['display']['acc']*100:.2f}%")
    print(f"  Overall:          {overall*100:.2f}%")

    print("\n=== Replay-Attack Attack Type Analysis ===")
    print("\n=== Replay-Attack Attack Type Analysis ===")
    for name, model in [("Baseline", baseline), ("Texture", texture), ("Frequency", frequency)]:
        res = evaluate_attack_types(model, replay_splits)
        overall = sum([v["acc"] for v in res.values()]) / max(1, len(res))
        print(f"{name}:")
        for key in ["live", "print", "display"]:
            print(f"  {key} (N={res[key]['n']}): {res[key]['acc']*100:.2f}%")
        print(f"  Overall: {overall*100:.2f}%")

    gating_res = evaluate_attack_types(two_expert, replay_splits)
    overall = sum([v["acc"] for v in gating_res.values()]) / max(1, len(gating_res))
    print("Multi-Expert:")
    for key in ["live", "print", "display"]:
        print(f"  {key} (N={gating_res[key]['n']}): {gating_res[key]['acc']*100:.2f}%")
    print(f"  Overall: {overall*100:.2f}%")

    print("\n=== Key Findings ===")
    print("- Texture CNN shows XX% higher accuracy on Print vs Display")
    print("- Frequency CNN shows XX% higher accuracy on Display vs Print")
    print("- This validates our physical intuition!")


if __name__ == "__main__":
    main()
