import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from utils.dataset import ReplayAttackDataset, raw_transforms
from models.texture_expert import TextureExpert
from models.frequency_expert import FrequencyExpert


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model, preferred, fallback, device):
    path = preferred if os.path.exists(preferred) else fallback
    model.load_state_dict(torch.load(path, map_location=device))
    return path


def compute_metrics(preds: np.ndarray, labels: np.ndarray):
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    live = tn / (tn + fp) if (tn + fp) else 0.0
    spoof = tp / (tp + fn) if (tp + fn) else 0.0
    overall = (tp + tn) / max(tp + tn + fp + fn, 1)
    far = fp / (fp + tn) if (fp + tn) else 0.0
    frr = fn / (fn + tp) if (fn + tp) else 0.0
    hter = (far + frr) / 2
    return {"overall": overall, "live": live, "spoof": spoof, "far": far, "frr": frr, "hter": hter, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def adaptive_predict_logits(texture, frequency, x, threshold: float):
    t_logits = texture(x)
    t_probs = F.softmax(t_logits, dim=1)
    t_conf = t_probs.max(dim=1).values
    use_freq = t_conf < threshold
    out = t_logits.clone()
    if use_freq.any():
        f_logits = frequency(x[use_freq])
        out[use_freq] = 0.7 * t_logits[use_freq] + 0.3 * f_logits
    return out, use_freq


def evaluate_frame_level(texture, frequency, dataset, device, threshold: float):
    preds = []
    labels = []
    freq_calls = 0
    with torch.inference_mode():
        for i in range(len(dataset)):
            item = dataset[i]
            x = item["raw"].unsqueeze(0).to(device)
            y = int(item["label"])
            out, used = adaptive_predict_logits(texture, frequency, x, threshold)
            pred = int(out.argmax(dim=1).item())
            preds.append(pred)
            labels.append(y)
            freq_calls += int(used.item())
    preds = np.array(preds)
    labels = np.array(labels)
    m = compute_metrics(preds, labels)
    m["freq_call"] = freq_calls / max(len(labels), 1)
    return m


def _sample_frame_indices(total_frames: int, target_count: int = 10):
    if total_frames <= 0:
        return []
    if total_frames <= target_count:
        return list(range(total_frames))
    step = total_frames / target_count
    return [min(int(i * step), total_frames - 1) for i in range(target_count)]


def read_video_frames(video_path: str, indices):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def evaluate_video_level(texture, frequency, dataset, device, threshold: float):
    preds = []
    labels = []
    freq_calls = 0
    total_frames = 0
    transform = raw_transforms()

    with torch.inference_mode():
        for sample in dataset.samples:
            vpath = sample["video_path"]
            label = int(sample["label"])
            import cv2

            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            idxs = _sample_frame_indices(total, 10)
            pil_frames = read_video_frames(vpath, idxs)
            if not pil_frames:
                continue
            xs = torch.stack([transform(im) for im in pil_frames], dim=0).to(device)
            out, used = adaptive_predict_logits(texture, frequency, xs, threshold)
            frame_preds = out.argmax(dim=1).detach().cpu().numpy().tolist()
            video_pred = Counter(frame_preds).most_common(1)[0][0]
            preds.append(video_pred)
            labels.append(label)
            freq_calls += int(used.sum().item())
            total_frames += len(pil_frames)

    preds = np.array(preds)
    labels = np.array(labels)
    m = compute_metrics(preds, labels)
    m["freq_call"] = freq_calls / max(total_frames, 1)
    m["n_videos"] = int(len(labels))
    return m


def main():
    device = get_device()
    seed = 42
    data_root = os.path.join("data", "replay-attack", "datasets", "fas_pure_data", "Idiap-replayattack")
    test_ds = ReplayAttackDataset(
        data_root,
        split="test",
        transform=raw_transforms(),
        raw_transform=raw_transforms(),
        freq_crop=False,
        seed=seed,
    )

    opt_path = Path("output/optimal_threshold.json")
    if not opt_path.exists():
        raise FileNotFoundError("output/optimal_threshold.json not found. Run find_optimal_threshold.py first.")
    opt = json.loads(opt_path.read_text(encoding="utf-8"))
    threshold = float(opt["threshold"])

    texture = TextureExpert().to(device)
    frequency = FrequencyExpert().to(device)
    tex_path = load_checkpoint(texture, "checkpoints/cross/texture.pth", "checkpoints/texture_expert/best_model.pth", device)
    freq_path = load_checkpoint(frequency, "checkpoints/cross/frequency_asymmetric.pth", "checkpoints/frequency_expert/best_model.pth", device)

    texture.eval()
    frequency.eval()

    frame_m = evaluate_frame_level(texture, frequency, test_ds, device, threshold)
    video_m = evaluate_video_level(texture, frequency, test_ds, device, threshold)

    print("=== Final Test Evaluation (1회, 누수 없음) ===")
    print(f"Optimal Threshold (from DEV): {threshold:.2f}")
    print(f"Using texture: {tex_path}")
    print(f"Using frequency: {freq_path}")

    print("\nFrame-level:")
    print(f"Overall: {frame_m['overall']*100:.2f}%")
    print(f"Live: {frame_m['live']*100:.2f}%")
    print(f"Spoof: {frame_m['spoof']*100:.2f}%")
    print(f"FAR: {frame_m['far']*100:.2f}%")
    print(f"FRR: {frame_m['frr']*100:.2f}%")
    print(f"HTER: {frame_m['hter']*100:.2f}%")
    print(f"Frequency Calls: {frame_m['freq_call']*100:.2f}%")

    print("\nVideo-level:")
    print(f"Overall: {video_m['overall']*100:.2f}%")
    print(f"Live: {video_m['live']*100:.2f}%")
    print(f"Spoof: {video_m['spoof']*100:.2f}%")
    print(f"FAR: {video_m['far']*100:.2f}%")
    print(f"FRR: {video_m['frr']*100:.2f}%")
    print(f"HTER: {video_m['hter']*100:.2f}%")
    print(f"Frequency Calls (per-frame): {video_m['freq_call']*100:.2f}%")
    print(f"N videos: {video_m['n_videos']}")

    out = {
        "seed": seed,
        "threshold_from_dev": threshold,
        "frame_level": frame_m,
        "video_level": video_m,
        "checkpoints": {"texture": tex_path, "frequency": freq_path},
    }
    Path("adaptive_final_results.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nSaved adaptive_final_results.json")


if __name__ == "__main__":
    main()

