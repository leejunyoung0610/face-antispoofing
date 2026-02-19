import glob
from collections import Counter

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from models.texture_expert import TextureExpert


def get_device() -> torch.device:
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels):
        self.videos = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.videos[idx], self.labels[idx]


def evaluate_video(model, video_paths, labels, device):
    """
    각 비디오에서 10 프레임 추출 → 예측 → 다수결
    """
    model.eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    frame_correct = 0
    frame_total = 0
    video_correct = 0

    with torch.inference_mode():
        for vpath, label in zip(video_paths, labels):
            cap = cv2.VideoCapture(vpath)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                continue

            preds = []
            step = max(total // 10, 1)
            for i in range(min(10, total)):
                frame_idx = min(i * step, total - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame_t = transform(frame).unsqueeze(0).to(device)

                out = model(frame_t)
                if isinstance(out, tuple):
                    out = out[0]

                pred = int(out.argmax(1).item())
                preds.append(pred)
                frame_correct += int(pred == label)
                frame_total += 1

            cap.release()

            if preds:
                video_pred = Counter(preds).most_common(1)[0][0]
                video_correct += int(video_pred == label)

    frame_acc = frame_correct / frame_total if frame_total > 0 else 0.0
    video_acc = video_correct / len(labels) if len(labels) > 0 else 0.0
    return frame_acc, video_acc


def main():
    device = get_device()
    print(f"Using device: {device}")

    texture = TextureExpert().to(device)

    print("=== MSU-MFSD (Original Checkpoint) ===")
    texture.load_state_dict(
        torch.load("checkpoints/texture_expert/best_model.pth", map_location=device)
    )

    msu_real = glob.glob("data/MSU-MFSD/scene01/real/*.mp4")
    msu_print = [
        f for f in glob.glob("data/MSU-MFSD/scene01/attack/*.mp4") if "printed_photo" in f
    ]
    msu_display = [
        f
        for f in glob.glob("data/MSU-MFSD/scene01/attack/*.mp4")
        if ("ipad_video" in f) or ("iphone_video" in f)
    ]

    live_f, live_v = evaluate_video(texture, msu_real, [0] * len(msu_real), device)
    print(f"Live:    frame={live_f:.2%} video={live_v:.2%}")

    print_f, print_v = evaluate_video(texture, msu_print, [1] * len(msu_print), device)
    print(f"Print:   frame={print_f:.2%} video={print_v:.2%}")

    disp_f, disp_v = evaluate_video(
        texture, msu_display, [1] * len(msu_display), device
    )
    print(f"Display: frame={disp_f:.2%} video={disp_v:.2%}")

    print("\n=== Replay-Attack (Fine-tuned) ===")
    texture.load_state_dict(torch.load("checkpoints/cross/texture.pth", map_location=device))

    replay_real = glob.glob(
        "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack/test/real/*.mov"
    )
    replay_attack = glob.glob(
        "data/replay-attack/datasets/fas_pure_data/Idiap-replayattack/test/attack/*/*.mov"
    )

    live_f, live_v = evaluate_video(texture, replay_real, [0] * len(replay_real), device)
    print(f"Live:   frame={live_f:.2%} video={live_v:.2%}")

    attack_f, attack_v = evaluate_video(
        texture, replay_attack, [1] * len(replay_attack), device
    )
    print(f"Attack: frame={attack_f:.2%} video={attack_v:.2%}")


if __name__ == "__main__":
    main()
