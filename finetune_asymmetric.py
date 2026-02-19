import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_cross_dataset import ReplayAttackDataset
from models.asymmetric_loss import AsymmetricLoss
from models.frequency_expert import FrequencyExpert


def get_device() -> torch.device:
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def evaluate_per_class(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    correct_live = total_live = 0
    correct_spoof = total_spoof = 0
    with torch.inference_mode():
        for batch in loader:
            x = batch["raw"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            live_mask = y == 0
            spoof_mask = y == 1
            if live_mask.any():
                correct_live += (preds[live_mask] == 0).sum().item()
                total_live += live_mask.sum().item()
            if spoof_mask.any():
                correct_spoof += (preds[spoof_mask] == 1).sum().item()
                total_spoof += spoof_mask.sum().item()
    live_acc = correct_live / total_live if total_live else 0.0
    spoof_acc = correct_spoof / total_spoof if total_spoof else 0.0
    overall = (correct_live + correct_spoof) / (total_live + total_spoof) if (total_live + total_spoof) else 0.0
    return live_acc, spoof_acc, overall


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FrequencyExpert with asymmetric loss.")
    parser.add_argument(
        "--data_root",
        default="data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
        help="Replay-Attack root.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--fn_weight", type=float, default=5.0)
    parser.add_argument("--fp_weight", type=float, default=1.0)
    parser.add_argument(
        "--save_path",
        default="checkpoints/cross/frequency_asymmetric.pth",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    train_ds = ReplayAttackDataset(args.data_root, split="train")
    test_ds = ReplayAttackDataset(args.data_root, split="test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FrequencyExpert().to(device)
    base_ckpt = (
        "checkpoints/cross/frequency.pth"
        if os.path.exists("checkpoints/cross/frequency.pth")
        else "checkpoints/frequency_expert/best_model.pth"
    )
    print(f"Loading base weights: {base_ckpt}")
    model.load_state_dict(torch.load(base_ckpt, map_location=device))

    criterion = AsymmetricLoss(fn_weight=args.fn_weight, fp_weight=args.fp_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_overall = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in prog:
            x = batch["raw"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            prog.set_postfix(loss=running / max(1, prog.n))

        live_acc, spoof_acc, overall = evaluate_per_class(model, test_loader, device)
        print(
            f"[Epoch {epoch}] Live Acc={live_acc:.4f} Spoof Acc={spoof_acc:.4f} Overall={overall:.4f}"
        )
        if overall > best_overall:
            best_overall = overall
            torch.save(model.state_dict(), args.save_path)

    print(f"Saved best model to: {args.save_path}")
    live_acc, spoof_acc, overall = evaluate_per_class(model, test_loader, device)
    print("=== Final (Replay-Attack test) ===")
    print(f"Live  정확도: {live_acc:.4f}")
    print(f"Spoof 정확도: {spoof_acc:.4f}")
    print(f"전체 정확도: {overall:.4f}")


if __name__ == "__main__":
    main()

