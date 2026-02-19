import argparse
import os
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import os
import torch
from PIL import Image
from torch import nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from utils.face_crop import crop_face_pil
from tqdm import tqdm

from models.baseline import BaselineModel
from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert
from models.gating import GatingModule


def default_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def raw_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


class ReplayAttackDataset(Dataset):
    """Replay-Attack 이미지 셋을 영상처럼 읽는 dataset."""

    def __init__(
        self,
        root: str,
        split: str = "test",
        transform: Optional[Callable] = None,
        raw_transform: Optional[Callable] = None,
        freq_crop: bool = False,
        seed: int = 42,
    ):
        self.root = root
        self.split = split
        self.transform = transform or default_transforms()
        self.raw_transform = raw_transform or raw_transforms()
        self.freq_transform = raw_transforms()
        self.freq_crop = freq_crop
        self.seed = seed

        base_split = "train" if split in {"train", "dev"} else split
        split_dir = os.path.join(root, base_split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"{split_dir} not found.")

        self.samples: List[Tuple[str, int]] = []
        for label_name, label in [("real", 0), ("attack", 1)]:
            label_dir = os.path.join(split_dir, label_name)
            if not os.path.exists(label_dir):
                continue

            if label_name == "real":
                for fname in os.listdir(label_dir):
                    if fname.lower().endswith(".mov"):
                        video_path = os.path.join(label_dir, fname)
                        self.samples.append({"video_path": video_path, "label": label})
            else:
                for subdir in os.listdir(label_dir):
                    subdir_path = os.path.join(label_dir, subdir)
                    if os.path.isdir(subdir_path):
                        for fname in os.listdir(subdir_path):
                            if fname.lower().endswith(".mov"):
                                video_path = os.path.join(subdir_path, fname)
                                self.samples.append({"video_path": video_path, "label": label})

        if not self.samples:
            raise RuntimeError(f"No samples found under {split_dir}.")

        # train split을 train/dev로 고정 분할 (누수 방지)
        if split in {"train", "dev"}:
            import random

            rng = random.Random(self.seed)
            idxs = list(range(len(self.samples)))
            rng.shuffle(idxs)
            n_train = int(len(idxs) * 0.8)
            train_idxs = idxs[:n_train]
            dev_idxs = idxs[n_train:]
            chosen = train_idxs if split == "train" else dev_idxs
            self.samples = [self.samples[i] for i in chosen]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        label = sample["label"]

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = max(total_frames // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        normalized = self.transform(image)
        raw = TF.to_tensor(TF.resize(image, (224, 224)))
        if self.freq_crop:
            freq_result = crop_face_pil(image, return_status=True)
            freq_source, freq_success = freq_result
        else:
            freq_source, freq_success = image, False
        freq_raw = self.freq_transform(freq_source)
        attack_type = "print" if "print" in video_path.lower() or "photo" in video_path.lower() else "display"
        metadata = {"video_path": video_path, "attack_type": attack_type}
        return {
            "frame": normalized,
            "raw": raw,
            "freq_raw": freq_raw,
            "freq_crop_success": freq_success,
            "label": label,
            "metadata": metadata,
        }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_raw: bool) -> Tuple[float, float]:
    model.eval()
    correct = total = tp = tn = fp = fn = 0
    with torch.inference_mode():
        for batch in loader:
            data = batch["raw"] if use_raw else batch["frame"]
            data = data.to(device)
            labels = batch["label"].to(device)
            logits = model(data)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    acc = correct / total if total else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    hter = (far + frr) / 2
    return acc, hter


def evaluate_gating(baseline, texture, frequency, gating, loader, device):
    gating.eval()
    correct = total = tp = tn = fp = fn = 0
    with torch.inference_mode():
        for batch in loader:
            frames = batch["frame"].to(device)
            raw = batch["raw"].to(device)
            labels = batch["label"].to(device)
            b_out = baseline(frames)
            t_out = texture(raw)
            f_out = frequency(raw)
            combined, _ = gating([b_out, t_out, f_out])
            preds = combined.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()
    acc = correct / total if total else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    hter = (far + frr) / 2
    return acc, hter


def evaluate_with_tta(model, loader, device, use_raw=False, n_aug=3):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for batch in tqdm(loader):
            x = batch["raw"] if use_raw else batch["frame"]
            x = x.to(device)
            labels = batch["label"].to(device)
            outputs = []
            outputs.append(model(x))
            outputs.append(model(torch.flip(x, dims=[3])))
            x_bright = torch.clamp(x * 1.2, 0, 1)
            outputs.append(model(x_bright))
            avg_output = torch.stack(outputs).mean(dim=0)
            preds = avg_output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    correct = (all_preds == all_labels).sum().item()
    total = len(all_preds)
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()
    acc = correct / total if total else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    hter = (far + frr) / 2
    return acc, hter


def main():
    parser = argparse.ArgumentParser(description="Evaluate MSU-trained experts on Replay-Attack.")
    parser.add_argument(
        "--data_root",
        default="data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
        help="Replay attack root.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--use_tta", action="store_true", help="Use Test-Time Augmentation"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dataset = ReplayAttackDataset(args.data_root, split="test")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    baseline = BaselineModel().to(device)
    texture = TextureExpert().to(device)
    frequency = FrequencyExpert().to(device)

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
    frequency_path = (
        "checkpoints/cross/frequency.pth"
        if os.path.exists("checkpoints/cross/frequency.pth")
        else "checkpoints/frequency_expert/best_model.pth"
    )

    baseline.load_state_dict(torch.load(baseline_path, map_location=device))
    texture.load_state_dict(torch.load(texture_path, map_location=device))
    frequency.load_state_dict(torch.load(frequency_path, map_location=device))

    print(f"Using: {baseline_path}")
    print(f"Using: {texture_path}")
    print(f"Using: {frequency_path}")
    gating = GatingModule(num_experts=3).to(device)
    gating.load_state_dict(torch.load("checkpoints/multi_expert/best_gating.pth", map_location=device))

    models = [
        ("Baseline", baseline),
        ("Texture", texture),
        ("Frequency", frequency),
    ]

    print("=== Cross-dataset (MSU→Replay) ===")
    evaluate_fn = evaluate_with_tta if args.use_tta else evaluate

    acc, hter = evaluate_fn(baseline, loader, device, use_raw=False)
    print(f"Baseline:     Acc={acc:.4f} HTER={hter:.4f}")
    acc, hter = evaluate_fn(texture, loader, device, use_raw=True)
    print(f"Texture:      Acc={acc:.4f} HTER={hter:.4f}")
    acc, hter = evaluate_fn(frequency, loader, device, use_raw=True)
    print(f"Frequency:    Acc={acc:.4f} HTER={hter:.4f}")
    acc, hter = evaluate_gating(baseline, texture, frequency, gating, loader, device)
    print(f"Multi-Expert: Acc={acc:.4f} HTER={hter:.4f}")


if __name__ == "__main__":
    main()
