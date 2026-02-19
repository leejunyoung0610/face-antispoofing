import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from evaluate_cross_dataset import ReplayAttackDataset
from models.baseline import BaselineModel
from models.frequency_expert import FrequencyExpert
from models.texture_expert import TextureExpert


def fine_tune(model, loader, device, epochs=5, lr=1e-5, save_path="checkpoint.pth"):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch in loader:
            data = batch["raw"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total if total else 0.0
        avg_loss = running_loss / total if total else 0.0
        print(f"[{model.__class__.__name__}] Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Saved {model.__class__.__name__} to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MSU models on Replay-Attack.")
    parser.add_argument(
        "--data_root",
        default="data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    dataset = ReplayAttackDataset(args.data_root, split="train")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    baseline = BaselineModel()
    baseline.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    texture = TextureExpert()
    texture.load_state_dict(torch.load("checkpoints/texture_expert/best_model.pth", map_location=device))
    frequency = FrequencyExpert()
    frequency.load_state_dict(torch.load("checkpoints/frequency_expert/best_model.pth", map_location=device))

    fine_tune(baseline, loader, device, epochs=5, lr=1e-5, save_path="checkpoints/cross/baseline.pth")
    fine_tune(texture, loader, device, epochs=5, lr=1e-5, save_path="checkpoints/cross/texture.pth")
    fine_tune(frequency, loader, device, epochs=5, lr=1e-5, save_path="checkpoints/cross/frequency.pth")


if __name__ == "__main__":
    os.makedirs("checkpoints/cross", exist_ok=True)
    main()
