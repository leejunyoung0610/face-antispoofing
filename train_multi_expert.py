import argparse
import csv
import os
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.gating import GatingModule


class CachedFeatureDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-expert gating module.")
    parser.add_argument("--data_root", default="data", help="Root directory with MSU-MFSD.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_dir", default="checkpoints/multi_expert")
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def load_feature_cache(root: str):
    path = os.path.join(root, "feature_cache.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run utils/precompute_features.py first.")
    return torch.load(path)


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    cache = load_feature_cache(args.data_root)
    train_ds = CachedFeatureDataset(cache["train"])
    val_ds = CachedFeatureDataset(cache["test"])

    print(f"Train cached samples: {len(train_ds)}")
    print(f"Val cached samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    gating = GatingModule(num_experts=3).to(device)
    optimizer = optim.Adam(gating.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "training_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "val_acc_baseline",
                "val_acc_texture",
                "val_acc_frequency",
            ]
        )

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        gating.train()
        train_loss = 0.0
        train_acc = 0.0
        steps = 0
        epoch_weights = []
        prog = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in prog:
            optimizer.zero_grad()
            b_out = batch["baseline"].to(device)
            t_out = batch["texture"].to(device)
            f_out = batch["frequency"].to(device)
            labels = batch["label"].to(device)
            combined, weights = gating([b_out, t_out, f_out])
            if epoch == 1 and steps == 0:
                for name, logit in zip(["baseline", "texture", "freq"], [b_out, t_out, f_out]):
                    print(f"[DEBUG] {name} logits mean={logit.mean():.3f} std={logit.std():.3f}")
            epoch_weights.append(weights.detach().cpu())
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            loss = criterion(combined, labels) - 0.1 * entropy
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += accuracy(combined, labels)
            steps += 1
            prog.set_postfix(loss=train_loss / steps, acc=train_acc / steps)
        train_loss /= steps
        train_acc /= steps

        gating.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_acc_baseline = 0.0
        val_acc_texture = 0.0
        val_acc_frequency = 0.0
        val_steps = 0
        with torch.inference_mode():
            for batch in val_loader:
                b_out = batch["baseline"].to(device)
                t_out = batch["texture"].to(device)
                f_out = batch["frequency"].to(device)
                labels = batch["label"].to(device)
                combined, _ = gating([b_out, t_out, f_out])
                loss = criterion(combined, labels)
                val_loss += loss.item()
                val_acc += accuracy(combined, labels)
                val_acc_baseline += accuracy(b_out, labels)
                val_acc_texture += accuracy(t_out, labels)
                val_acc_frequency += accuracy(f_out, labels)
                val_steps += 1
        val_loss /= val_steps
        val_acc /= val_steps
        val_acc_baseline /= val_steps
        val_acc_texture /= val_steps
        val_acc_frequency /= val_steps

        if epoch_weights:
            mean_w = torch.cat(epoch_weights, dim=0).mean(dim=0)
            print(
                f"[DEBUG] Gating weights: baseline={mean_w[0]:.3f} "
                f"texture={mean_w[1]:.3f} freq={mean_w[2]:.3f}"
            )

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"(baseline={val_acc_baseline:.4f} texture={val_acc_texture:.4f} "
            f"frequency={val_acc_frequency:.4f})"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    epoch,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    val_acc_baseline,
                    val_acc_texture,
                    val_acc_frequency,
                ]
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                gating.state_dict(), os.path.join(args.save_dir, "best_gating.pth")
            )
        torch.save(gating.state_dict(), os.path.join(args.save_dir, "last_gating.pth"))


if __name__ == "__main__":
    main()
