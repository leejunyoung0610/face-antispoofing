import os
from typing import Callable, List, Optional, Tuple

import re

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.face_crop import crop_face_pil


SUPPORTED_DATASETS = {"MSU-MFSD", "OULU-NPU", "Replay-Attack"}
LABEL_MAP = {"Live": 0, "Spoof": 1}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov"}
ATTACK_TYPE_KEYWORDS = ("ipad_video", "iphone_video", "printed_photo")


def default_transforms():
    """Normalized transform for CNN"""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def raw_transforms():
    """Raw transform (0-1) for LBP/FFT"""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )


def train_augmentation_transforms():
    """Augmentation for training"""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class FaceAntiSpoofDataset(Dataset):
    """Generic dataset for Live / Spoof binary classification."""

    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        transform: Optional[Callable] = None,
    ):
        self.dataset_name = dataset_name
        self.root = root
        self.split = split
        if split == "train":
            self.norm_transform = transform or train_augmentation_transforms()
        else:
            self.norm_transform = transform or default_transforms()
        self.raw_transform = raw_transforms()

        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(f"{dataset_name} is not supported.")

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of 'train', 'val', or 'test'.")

        if not os.path.isdir(self._dataset_path()):
            raise FileNotFoundError(f"{self._dataset_path()} does not exist.")

        self.samples = self._prepare_split(split_ratios, seed)[split]

    def _dataset_path(self) -> str:
        return os.path.join(self.root, self.dataset_name)

    def _gather_samples(self) -> List[Tuple[str, int]]:
        samples: List[Tuple[str, int]] = []
        for label_name, idx in LABEL_MAP.items():
            label_dir = os.path.join(self._dataset_path(), label_name)
            if not os.path.isdir(label_dir):
                continue
            for root_dir, _, files in os.walk(label_dir):
                for file in sorted(files):
                    _, ext = os.path.splitext(file)
                    if ext.lower() in IMAGE_EXTENSIONS:
                        samples.append((os.path.join(root_dir, file), idx))
        if not samples:
            raise RuntimeError(f"No data discovered inside {self._dataset_path()}.")
        return samples

    def _prepare_split(
        self, split_ratios: Tuple[float, float, float], seed: int
    ) -> dict:
        samples = self._gather_samples()
        train_ratio, val_ratio, test_ratio = split_ratios
        if not torch.isclose(torch.tensor(train_ratio + val_ratio + test_ratio), torch.tensor(1.0)):
            raise ValueError("split_ratios must sum to 1.0.")

        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(samples), generator=rng).tolist()

        train_end = int(len(indices) * train_ratio)
        val_end = train_end + int(len(indices) * val_ratio)

        ordered_samples = [samples[i] for i in indices]
        return {
            "train": ordered_samples[:train_end],
            "val": ordered_samples[train_end:val_end],
            "test": ordered_samples[val_end:],
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def _subject_list_from_file(path: str) -> set[int]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    return {int(line) for line in lines}


def _infer_attack_type(filename: str, label: int) -> str:
    lowercase = filename.lower()
    for attack_type in ATTACK_TYPE_KEYWORDS:
        if attack_type in lowercase:
            return attack_type
    return "live" if label == 0 else "unknown"


def _sample_frame_indices(total_frames: int, target_count: int) -> List[int]:
    if total_frames <= 0 or target_count <= 0:
        return []
    if total_frames <= target_count:
        return list(range(total_frames))
    step = total_frames / target_count
    return [min(int(i * step), total_frames - 1) for i in range(target_count)]


class MSUMFSDDataset(Dataset):
    """MSU-MFSD에서 제공하는 Live/Spoof 비디오를 프레임 단위로 반환하는 Dataset."""

    def __init__(self, root: str = "data", split: str = "train", transform: Optional[Callable] = None, frames_per_video: int = 10, freq_crop: bool = False):
        self.root = root
        self.dataset_root = os.path.join(root, "MSU-MFSD")
        self.split = split
        self.frames_per_video = frames_per_video
        self.freq_crop = freq_crop

        if split == "train":
            self.norm_transform = transform or train_augmentation_transforms()
        else:
            self.norm_transform = transform or default_transforms()

        self.raw_transform = raw_transforms()
        self.freq_transform = raw_transforms()

        if split not in {"train", "test"}:
            raise ValueError("MSU-MFSD split must be either 'train' or 'test'.")

        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"{self.dataset_root} does not exist.")

        list_path = os.path.join(self.dataset_root, f"{split}_sub_list.txt")
        self.subject_ids = _subject_list_from_file(list_path)
        self.samples = self._build_sample_index()

        if not self.samples:
            raise RuntimeError(f"No samples found for split '{split}'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, dict]:
        sample = self.samples[index]
        frame = self._read_frame(sample["video_path"], sample["frame_idx"])
        raw_tensor = self.raw_transform(frame)
        norm_tensor = self.norm_transform(frame) if self.norm_transform is not None else raw_tensor
        if self.freq_crop:
            freq_result = crop_face_pil(frame, return_status=True)
            freq_source, freq_success = freq_result
        else:
            freq_source, freq_success = frame, False
        freq_raw_tensor = self.freq_transform(freq_source)

        label = sample["label"]
        metadata = {
            "video_path": sample["video_path"],
            "attack_type": sample["attack_type"],
            "frame_idx": sample["frame_idx"],
        }
        return {
            "frame": norm_tensor,
            "raw": raw_tensor,
            "freq_raw": freq_raw_tensor,
            "freq_crop_success": freq_success,
            "label": label,
            "metadata": metadata,
        }

    def _build_sample_index(self) -> List[dict]:
        videos = self._discover_videos()
        indexed: List[dict] = []
        for video in videos:
            frame_indices = _sample_frame_indices(
                video["total_frames"], self.frames_per_video
            )
            if not frame_indices:
                continue
            for frame_idx in frame_indices:
                indexed.append(
                    {
                        "video_path": video["path"],
                        "label": video["label"],
                        "attack_type": video["attack_type"],
                        "frame_idx": frame_idx,
                    }
                )
        return indexed

    def _discover_videos(self) -> List[dict]:
        videos: List[dict] = []
        for scene_name in sorted(os.listdir(self.dataset_root)):
            scene_path = os.path.join(self.dataset_root, scene_name)
            if not os.path.isdir(scene_path) or not scene_name.startswith("scene"):
                continue

            for label_dir, label in (("real", 0), ("attack", 1)):
                dir_path = os.path.join(scene_path, label_dir)
                if not os.path.isdir(dir_path):
                    continue

                for filename in sorted(os.listdir(dir_path)):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in VIDEO_EXTENSIONS:
                        continue
                    subject_id = self._parse_subject_id(filename)
                    if subject_id not in self.subject_ids:
                        continue

                    video_path = os.path.join(dir_path, filename)
                    attack_type = _infer_attack_type(filename, label)
                    total_frames = self._count_frames(video_path)
                    if total_frames <= 0:
                        continue

                    videos.append(
                        {
                            "path": video_path,
                            "label": label,
                            "attack_type": attack_type,
                            "total_frames": total_frames,
                        }
                    )
        return videos

    def _parse_subject_id(self, filename: str) -> Optional[int]:
        match = re.search(r"client0*([0-9]+)", filename.lower())
        if not match:
            return None
        return int(match.group(1))

    def _count_frames(self, path: str) -> int:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(total, 0)

    def _read_frame(self, path: str, frame_idx: int) -> Image.Image:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            raise RuntimeError(f"Unable to read frame {frame_idx} from {path}")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)


if __name__ == "__main__":
    dataset = MSUMFSDDataset(root="data", split="train", frames_per_video=5)
    print(f"Total samples: {len(dataset)}")
    sample = dataset[0]
    print(
        f"Sample shapes: raw={tuple(sample['raw'].shape)}, frame={tuple(sample['frame'].shape)}, "
        f"label={sample['label']}, meta={sample['metadata']}"
    )
