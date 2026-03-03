import argparse
import csv
import os
from typing import Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.texture2_expert import Texture2Expert
from utils.dataset import MSUMFSDDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train texture2 expert anti-spoofing model.")
    parser.add_argument(
        "--data_root",
        default="data",
        help="Root directory that contains the MSU-MFSD folder.",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--save_dir", default="checkpoints/texture2_expert", help="Directory to save checkpoints."
    )
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    preds = torch.argmax(output, dim=1)
    return (preds == target).float().mean().item()


def evaluate(model: nn.Module, criterion, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_total = 0.0
    acc_total = 0.0
    steps = 0
    with torch.inference_mode():
        for batch in loader:
            frames = batch["freq_raw"].to(device)
            labels = batch["label"].to(device)
            logits = model(frames)
            loss = criterion(logits, labels)
            loss_total += loss.item()
            acc_total += accuracy(logits, labels)
            steps += 1
    return loss_total / steps, acc_total / steps


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "training_log.csv")

    train_dataset = MSUMFSDDataset(root=args.data_root, split="train")
    val_dataset = MSUMFSDDataset(root=args.data_root, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model = Texture2Expert().to(device)
    labels = [entry["label"] for entry in train_dataset.samples]
    n_live = labels.count(0)
    n_spoof = labels.count(1)
    total = len(labels)
    weight = torch.tensor(
        [total / max(n_live, 1), total / max(n_spoof, 1)], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val_acc = 0.0

    with open(log_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_steps = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in progress:
            frames = batch["freq_raw"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logits, labels)
            train_steps += 1
            progress.set_postfix(loss=train_loss / train_steps, acc=train_acc / train_steps)
        train_loss /= train_steps
        train_acc /= train_steps

        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        scheduler.step(val_loss)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))


if __name__ == "__main__":
    main()
