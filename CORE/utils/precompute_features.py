import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.baseline import BaselineModel
from models.texture2_expert import Texture2Expert
from models.texture_expert import TextureExpert
from utils.dataset import MSUMFSDDataset


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def precompute():
    device = get_device()
    print(f"Using device for caching: {device}")

    baseline = BaselineModel().to(device)
    texture = TextureExpert().to(device)
    texture2 = Texture2Expert().to(device)

    baseline.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    texture.load_state_dict(
        torch.load("checkpoints/texture_expert/best_model.pth", map_location=device)
    )
    texture2.load_state_dict(
        torch.load("checkpoints/texture2_expert/best_model.pth", map_location=device)
    )

    baseline.eval()
    texture.eval()
    texture2.eval()

    cache = {"train": [], "test": []}

    for split in ["train", "test"]:
        dataset = MSUMFSDDataset("data", split=split)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
        for batch in tqdm(loader, desc=f"Caching {split}"):
            frames = batch["frame"].to(device)
            raw = batch["raw"].to(device)
            labels = batch["label"]
            with torch.inference_mode():
                b_logits = baseline(frames)
                t_logits = texture(raw)
                f_logits = texture2(raw)
            for j in range(len(labels)):
                cache[split].append(
                    {
                        "baseline": b_logits[j].cpu(),
                        "texture": t_logits[j].cpu(),
                        "texture2": f_logits[j].cpu(),
                        "label": int(labels[j]),
                    }
                )
        print(f"Cached {len(cache[split])} samples for {split}")

    os.makedirs("data", exist_ok=True)
    torch.save(cache, "data/feature_cache.pt")
    print("Feature cache saved to data/feature_cache.pt")


if __name__ == "__main__":
    precompute()
