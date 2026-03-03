# cross_dataset_msu_latefusion_eval.py
import os
import re
import csv
import pickle
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.feature import local_binary_pattern

from models.texture_expert import TextureExpert


# -----------------------------
# Config
# -----------------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MSU_ROOT = "data/msu-mfsd"
TEST_SUB_LIST = os.path.join(MSU_ROOT, "test_sub_list.txt")

TEXTURE_CKPT = "checkpoints/cross/texture.pth"   # Replay에서 학습/파인튜닝된 Texture 모델이라고 가정
LBP_RF_PKL = "lbp_rf_model.pkl"

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
ATTACK_TYPES = ["printed_photo", "ipad_video", "iphone_video", "unknown"]


# -----------------------------
# Helpers: IDs / parsing
# -----------------------------
def load_subject_ids(path: str) -> Optional[set]:
    """Return set[int] of subject ids, or None if file missing."""
    if not os.path.exists(path):
        return None
    ids = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # '01' -> 1
            try:
                ids.add(int(s))
            except ValueError:
                # 혹시 'client01' 같은 게 섞여있을 때
                m = re.search(r"(\d+)", s)
                if m:
                    ids.add(int(m.group(1)))
    return ids


def parse_subject_id_from_filename(filename: str) -> Optional[int]:
    """
    MSU 파일명 예:
      attack_client033_laptop_SD_iphone_video_scene01.mov
      real_client001_... (보통 real도 clientXXX 포함)
    """
    m = re.search(r"client0*([0-9]+)", filename.lower())
    if not m:
        return None
    return int(m.group(1))


def infer_label_from_path(path: str) -> int:
    """
    MSU 폴더 구조:
      .../sceneXX/real/*.mp4|mov -> Live(0)
      .../sceneXX/attack/*.mp4|mov -> Spoof(1)
    """
    lower = path.lower()
    if "/real/" in lower or "\\real\\" in lower:
        return 0
    if "/attack/" in lower or "\\attack\\" in lower:
        return 1
    # fallback: 파일명에 real/attack이 들어가는 경우
    if "attack_" in os.path.basename(lower):
        return 1
    return 0


def discover_msu_videos(msu_root: str, subject_ids: Optional[set]) -> List[Dict]:
    videos = []
    for scene_name in sorted(os.listdir(msu_root)):
        if not scene_name.startswith("scene"):
            continue
        scene_path = os.path.join(msu_root, scene_name)
        if not os.path.isdir(scene_path):
            continue

        for subdir in ("real", "attack"):
            d = os.path.join(scene_path, subdir)
            if not os.path.isdir(d):
                continue

            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith(VIDEO_EXTS):
                    continue

                sid = parse_subject_id_from_filename(fn)
                if subject_ids is not None:
                    # subject list 기반 필터
                    if sid is None or sid not in subject_ids:
                        continue

                videos.append(
                    {
                        "video_path": os.path.join(d, fn),
                        "label": 0 if subdir == "real" else 1,
                        "scene": scene_name,
                        "subject_id": sid,
                    }
                )
    return videos


# -----------------------------
# Helpers: frame extraction
# -----------------------------
def read_middle_frame(video_path: str) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # 일부 코덱에서 프레임카운트가 0일 수 있어 fallback로 첫 프레임 시도
        idx = 0
    else:
        idx = total // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()

    if not ok or frame is None:
        # fallback: 앞쪽 몇 프레임 중 하나라도 읽히면 사용
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                break

    cap.release()
    if not ok or frame is None:
        return None

    # BGR -> RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def read_frames_at_fractions(video_path: str, fractions: List[float]) -> List[np.ndarray]:
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return frames

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, 1)

    for frac in fractions:
        idx = min(max(int(total * frac), 0), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames


def infer_attack_type(video_path: str) -> str:
    lower = os.path.basename(video_path).lower()
    if "printed_photo" in lower:
        return "printed_photo"
    if "ipad_video" in lower:
        return "ipad_video"
    if "iphone_video" in lower:
        return "iphone_video"
    return "unknown"


def late_fusion_for_frame(frame: np.ndarray, tfm, texture, lbp_rf) -> Tuple[float, float, float]:
    pil = Image.fromarray(frame)
    raw = tfm(pil)
    x = raw.unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        out = texture(x)
        t_prob = F.softmax(out, dim=1)[0, 1].item()

    rgb_uint8 = (raw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    lbp_feat = extract_multiscale_lbp(rgb_uint8)
    l_prob = float(lbp_rf.predict_proba([lbp_feat])[0][1])

    fused = (t_prob + l_prob) / 2.0
    return fused, t_prob, l_prob


# -----------------------------
# LBP features
# -----------------------------
def extract_multiscale_lbp(rgb_uint8: np.ndarray) -> np.ndarray:
    """
    Multi-scale LBP:
      (R=1,P=8,bins=59), (R=2,P=16,bins=243), (R=3,P=24,bins=299)
    """
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    feats = []
    for R, P, bins in [(1, 8, 59), (2, 16, 243), (3, 24, 299)]:
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        feats.extend(hist.tolist())
    return np.asarray(feats, dtype=np.float32)


# -----------------------------
# Main eval
# -----------------------------
def main():
    print("=== MSU-MFSD Cross-dataset (Replay->MSU) Zero-shot 평가 ===\n")
    print(f"Device: {DEVICE}")
    print(f"MSU root: {MSU_ROOT}")
    print(f"Test subject list: {TEST_SUB_LIST}")

    # 1) load subject ids
    subject_ids = load_subject_ids(TEST_SUB_LIST)
    if subject_ids is None:
        print("⚠️ test_sub_list.txt 없음 → subject filter 없이 전체 스캔합니다.\n")
    else:
        print(f"✅ subject_ids loaded: n={len(subject_ids)} sample={sorted(list(subject_ids))[:10]}\n")

    # 2) discover videos
    videos = discover_msu_videos(MSU_ROOT, subject_ids)
    n_live = sum(1 for v in videos if v["label"] == 0)
    n_spoof = sum(1 for v in videos if v["label"] == 1)

    print(f"Discovered MSU videos: {len(videos)}")
    print(f"  Live:  {n_live}")
    print(f"  Spoof: {n_spoof}\n")

    if len(videos) == 0:
        print("❌ 0 videos indexed. (subject filter / path / filename parsing을 다시 확인)")
        return

    # 3) load models
    print("Loading Texture CNN checkpoint...")
    texture = TextureExpert().to(DEVICE)

    sd = torch.load(TEXTURE_CKPT, map_location="cpu")
    # 혹시 {'state_dict': ...} 형태면 처리
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    texture.load_state_dict(sd)
    texture.to(DEVICE)
    texture.eval()

    print("Loading LBP RandomForest...")
    with open(LBP_RF_PKL, "rb") as f:
        lbp_rf = pickle.load(f)

    print("✅ models loaded.\n")

    # 4) transforms (raw path: [0,1] only)
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    # 5) evaluation loop (video-level = 1 frame per video)
    # Confusion matrix (positive = Spoof=1)
    TP = TN = FP = FN = 0
    TP3 = TN3 = FP3 = FN3 = 0

    # per-class
    live_total = spoof_total = 0
    live_correct = spoof_correct = 0
    live_total3 = spoof_total3 = 0
    live_correct3 = spoof_correct3 = 0

    attack_stats = {
        atype: {"total": 0, "correct": 0, "TP": 0, "FN": 0}
        for atype in ATTACK_TYPES
    }

    failures = []  # store up to some
    for i, v in enumerate(videos, start=1):
        vp = v["video_path"]
        y = v["label"]

        frame = read_middle_frame(vp)
        if frame is None:
            # unreadable video -> skip (or treat as failure)
            continue

        pil = Image.fromarray(frame)
        raw = tfm(pil)  # (3,224,224) in [0,1]

        # Texture prob (spoof=1)
        x = raw.unsqueeze(0).to(DEVICE)
        with torch.inference_mode():
            out = texture(x)
            t_prob = F.softmax(out, dim=1)[0, 1].item()

        # LBP RF prob (spoof=1)
        rgb_uint8 = (raw.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        lbp_feat = extract_multiscale_lbp(rgb_uint8)
        l_prob = float(lbp_rf.predict_proba([lbp_feat])[0][1])

        fused = (t_prob + l_prob) / 2.0
        pred = 1 if fused >= 0.5 else 0
        attack_type = infer_attack_type(vp)
        if y == 1:
            stats = attack_stats.setdefault(attack_type, {
                "total": 0, "correct": 0, "TP": 0, "FN": 0
            })
            stats["total"] += 1
            if pred == 1:
                stats["correct"] += 1
                stats["TP"] += 1
            else:
                stats["FN"] += 1

        # update confusion (positive=1 spoof)
        if y == 1 and pred == 1:
            TP += 1
        elif y == 0 and pred == 0:
            TN += 1
        elif y == 0 and pred == 1:
            FP += 1  # Live -> Spoof (False Reject)
            failures.append((vp, y, pred, t_prob, l_prob, fused))
        elif y == 1 and pred == 0:
            FN += 1  # Spoof -> Live (False Accept)
            failures.append((vp, y, pred, t_prob, l_prob, fused))

        # class stats
        if y == 0:
            live_total += 1
            if pred == y:
                live_correct += 1
        else:
            spoof_total += 1
            if pred == y:
                spoof_correct += 1

        frames_three = read_frames_at_fractions(vp, [0.25, 0.5, 0.75])
        three_scores: List[float] = []
        for frame_3 in frames_three:
            fused_3, _, _ = late_fusion_for_frame(frame_3, tfm, texture, lbp_rf)
            three_scores.append(fused_3)

        if three_scores:
            avg_score = sum(three_scores) / len(three_scores)
            pred3 = 1 if avg_score >= 0.5 else 0
            if y == 1 and pred3 == 1:
                TP3 += 1
            elif y == 0 and pred3 == 0:
                TN3 += 1
            elif y == 0 and pred3 == 1:
                FP3 += 1
            elif y == 1 and pred3 == 0:
                FN3 += 1

            if y == 0:
                live_total3 += 1
                if pred3 == y:
                    live_correct3 += 1
            else:
                spoof_total3 += 1
                if pred3 == y:
                    spoof_correct3 += 1

        if i % 50 == 0:
            print(f"  {i}/{len(videos)}...")

    total = TP + TN + FP + FN
    if total == 0:
        print("\n❌ No evaluatable samples (all videos unreadable?).")
        return

    acc = (TP + TN) / total
    live_acc = live_correct / max(live_total, 1)
    spoof_acc = spoof_correct / max(spoof_total, 1)

    # Metrics for PAD:
    # FAR: Spoof -> Live = FN / Spoof_total
    # FRR: Live -> Spoof = FP / Live_total
    far = FN / max(spoof_total, 1)
    frr = FP / max(live_total, 1)
    hter = (far + frr) / 2

    print("\n" + "=" * 60)
    print("MSU-MFSD Zero-shot (Replay-trained) Results (Late Fusion)")
    print("=" * 60)
    print(f"Total:    {total} (Live {live_total}, Spoof {spoof_total})")
    print(f"Overall:  {acc*100:.2f}%  ({TP+TN}/{total})")
    print(f"Live acc: {live_acc*100:.2f}% ({live_correct}/{live_total})")
    print(f"Spoof acc:{spoof_acc*100:.2f}% ({spoof_correct}/{spoof_total})")
    print(f"\nFAR (Spoof→Live): {far*100:.2f}%  (FN={FN})")
    print(f"FRR (Live→Spoof): {frr*100:.2f}%  (FP={FP})")
    print(f"HTER:             {hter*100:.2f}%")
    print("\nConfusion (positive=Spoof=1):")
    print(f"  TN={TN}  FP={FP}")
    print(f"  FN={FN}  TP={TP}")
    print("=" * 60)

    total3 = TP3 + TN3 + FP3 + FN3
    if total3 > 0:
        acc3 = (TP3 + TN3) / total3
        far3 = FN3 / max(spoof_total3, 1)
        frr3 = FP3 / max(live_total3, 1)
        hter3 = (far3 + frr3) / 2
    else:
        acc3 = far3 = frr3 = hter3 = 0.0

    print("\n3-Frame Late Fusion (25%,50%,75%) summary:")
    print(f"  Total evald: {total3}")
    print(f"  Overall:  {acc3*100:.2f}%")
    print(f"  FAR:      {far3*100:.2f}%")
    print(f"  FRR:      {frr3*100:.2f}%")
    print(f"  HTER:     {hter3*100:.2f}%")

    print("\nComparison: single middle frame vs 3-frame average")
    print(f"{'Metric':<15}{'Middle frame':>18}{'3-frame avg':>18}")
    print(f"{'Accuracy':<15}{acc*100:>18.2f}{acc3*100:>18.2f}")
    print(f"{'FAR (%)':<15}{far*100:>18.2f}{far3*100:>18.2f}")
    print(f"{'FRR (%)':<15}{frr*100:>18.2f}{frr3*100:>18.2f}")
    print(f"{'HTER (%)':<15}{hter*100:>18.2f}{hter3*100:>18.2f}")

    print("\nAttack-type breakdown (printed_photo/ipad_video/iphone_video/unknown)")
    print(f"{'Type':<16}{'Count':>8}{'Acc (%)':>12}{'FAR (%)':>12}{'FRR (%)':>12}{'TP':>6}{'FN':>6}")
    print("-" * 70)
    for atype in ATTACK_TYPES:
        stats = attack_stats.get(atype, {})
        total_type = stats.get("total", 0)
        correct = stats.get("correct", 0)
        fn = stats.get("FN", 0)
        tp = stats.get("TP", 0)
        acc_type = correct / max(total_type, 1)
        fr = fn / max(total_type, 1)
        print(f"{atype:<16}{total_type:>8}{acc_type*100:>12.2f}{0.0:>12.2f}{fr*100:>12.2f}{tp:>6}{fn:>6}")

    attack_csv = "msu_attack_type_results.csv"
    with open(attack_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack_type", "count", "accuracy", "FAR", "FRR", "TP", "FN"])
        for atype in ATTACK_TYPES:
            stats = attack_stats.get(atype, {})
            total_type = stats.get("total", 0)
            correct = stats.get("correct", 0)
            fn = stats.get("FN", 0)
            tp = stats.get("TP", 0)
            acc_type = correct / max(total_type, 1)
            fr = fn / max(total_type, 1)
            w.writerow([
                atype,
                total_type,
                f"{acc_type:.6f}",
                f"{0.0:.6f}",
                f"{fr:.6f}",
                tp,
                fn,
            ])

    print(f"\n✅ Attack-type breakdown saved: {attack_csv}")

    # Save CSV
    out_csv = "msu_cross_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "total", "live_total", "spoof_total",
                    "acc", "live_acc", "spoof_acc", "FAR", "FRR", "HTER",
                    "TN", "FP", "FN", "TP"])
        w.writerow(["MSU-MFSD", total, live_total, spoof_total,
                    f"{acc:.6f}", f"{live_acc:.6f}", f"{spoof_acc:.6f}",
                    f"{far:.6f}", f"{frr:.6f}", f"{hter:.6f}",
                    TN, FP, FN, TP])

    # Save failures
    fail_csv = "msu_cross_failures.csv"
    with open(fail_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_path", "true_label", "pred_label", "texture_spoof_prob", "lbp_spoof_prob", "fused_spoof_prob"])
        for row in failures[:200]:
            w.writerow(row)

    print(f"\n✅ Saved: {out_csv}, {fail_csv}")
    print("Note: MSU가 scene01만 있거나 subset이면, 보고서에 'subset MSU'로 명시하세요.")


if __name__ == "__main__":
    main()