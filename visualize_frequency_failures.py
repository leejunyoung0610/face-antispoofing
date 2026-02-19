import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from evaluate_cross_dataset import ReplayAttackDataset, raw_transforms
from models.frequency_expert import FrequencyExpert


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_frequency(device):
    frequency = FrequencyExpert().to(device)
    ckpt = (
        "checkpoints/cross/frequency_asymmetric.pth"
        if os.path.exists("checkpoints/cross/frequency_asymmetric.pth")
        else "checkpoints/frequency_expert/best_model.pth"
    )
    frequency.load_state_dict(torch.load(ckpt, map_location=device))
    frequency.eval()
    return frequency


def preprocess(raw):
    gray = 0.299 * raw[:, 0] + 0.587 * raw[:, 1] + 0.114 * raw[:, 2]
    fft = torch.fft.fft2(gray)
    mag = torch.log1p(torch.abs(torch.fft.fftshift(fft)))
    mag = (mag - mag.amin(dim=[1, 2], keepdim=True)) / (
        mag.amax(dim=[1, 2], keepdim=True) - mag.amin(dim=[1, 2], keepdim=True) + 1e-8
    )
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=raw.device
    ).view(1, 1, 3, 3)
    edge = F.conv2d(gray.unsqueeze(1), kernel, padding=1).squeeze(1)
    edge = torch.abs(edge)
    edge = (edge - edge.amin(dim=[1, 2], keepdim=True)) / (
        edge.amax(dim=[1, 2], keepdim=True) - edge.amin(dim=[1, 2], keepdim=True) + 1e-8
    )
    return gray, mag, edge


def tensor_to_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def normalized_colormap(channel, cmap="viridis"):
    arr = channel.squeeze().cpu().numpy()
    arr = np.nan_to_num(arr)
    norm = (arr - arr.min()) / ((arr.max() - arr.min()) + 1e-8)
    cmap_img = plt.get_cmap(cmap)(norm)
    return (cmap_img[..., :3] * 255).astype(np.uint8)


def find_cases(model, dataset, device, target="fp", limit=5):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    cases = []
    with torch.inference_mode():
        for batch in loader:
            raw = batch["raw"].to(device)
            label = batch["label"].item()
            logits = model(raw)
            pred = logits.argmax(1).item()
            if target == "fp" and label == 0 and pred == 1:
                cases.append((raw[0], label, pred, batch["raw"].squeeze(0).cpu()))
            if target == "tn" and label == 0 and pred == 0:
                cases.append((raw[0], label, pred, batch["raw"].squeeze(0).cpu()))
            if len(cases) == limit:
                break
    return cases


def sample_images(sample):
    raw_layer, _, _, raw_cpu = sample
    gray, mag, edge = preprocess(raw_layer.unsqueeze(0))
    orig = tensor_to_image(raw_cpu.unsqueeze(0))
    gray_img = np.stack((gray.squeeze().cpu().numpy(),) * 3, axis=-1)
    gray_img = (gray_img * 255).astype(np.uint8)
    fft_img = normalized_colormap(mag, cmap="viridis")
    lap_img = normalized_colormap(edge, cmap="viridis")
    return orig, gray_img, fft_img, lap_img


def main():
    device = get_device()
    frequency = load_frequency(device)
    dataset = ReplayAttackDataset(
        "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack",
        split="test",
        transform=raw_transforms(),
        raw_transform=raw_transforms(),
    )
    failure_cases = find_cases(frequency, dataset, device, target="fp", limit=5)
    success_cases = find_cases(frequency, dataset, device, target="tn", limit=5)

    fig = plt.figure(figsize=(20, 20))
    fig.suptitle("Frequency Expert: Live Failure vs Successful Live", fontsize=16)
    all_cases = failure_cases + success_cases
    labels = ["Live FP"] * len(failure_cases) + ["Live OK"] * len(success_cases)
    for idx, sample in enumerate(all_cases):
        orig, gray_img, fft_img, lap_img = sample_images(sample)
        for sub_idx, image in enumerate([orig, gray_img, fft_img, lap_img]):
            ax = fig.add_subplot(10, 4, idx * 4 + sub_idx + 1)
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if sub_idx == 0:
                ax.set_ylabel(labels[idx], rotation=0, labelpad=40)
            titles = ["RGB", "Grayscale", "FFT Mag", "Laplacian"]
            ax.set_title(titles[sub_idx], fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("frequency_failure_comparison.png")


if __name__ == "__main__":
    main()
