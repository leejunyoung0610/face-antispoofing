import torch
import torch.nn.functional as F
from models.texture_expert import TextureExpert
import pickle
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
import os
from torchvision import transforms

device = torch.device('mps')

def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    for R, P, bins in [(1, 8, 59), (2, 16, 243), (3, 24, 299)]:
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        features.extend(hist)
    return np.array(features)

def read_video_frame(video_path, frame_idx=30):
    """비디오에서 프레임 추출"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None

print("="*70)
print("MSU증")
print("="*70)

# 모델 로드
print("\n모델 로드...")
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval()

with open('lbp_rf_model.pkl', 'rb') as f:
    lbp_rf = pickle.load(f)

print("✅ 완료\n")

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# MSU 데이터 수집
msu_path = 'data/msu-mfsd/scene01'

real_files = [os.path.join(msu_path, 'real', f) 
              for f in os.listdir(os.path.join(msu_path, 'real'))
              if f.endswith(('.mov', '.mp4', '.avi'))]

attack_files = [os.path.join(msu_path, 'attack', f)
                for f in os.listdir(os.path.join(msu_path, 'attack'))
                if f.endswith(('.mov', '.mp4', '.avi'))]

print(f"Real videos: {len(real_files)}")
print(f"Attack videos: {len(attack_files)}")
print(f"Total: {len(real_files) + len(attack_files)}\n")

# 평가
print("평가 중...\n")

all_preds = []
all_labels = []
failed_files = []
for i, video_path in enumerate(real_files):
    frame = read_video_frame(video_path)
    
    if frame is None:
        failed_files.append(('real', video_path))
        continue
    
    # Texture
    x = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        t_prob = F.softmax(texture(x), dim=1)[0][1].item()
    
    # LBP
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_resized = cv2.resize(frame_bgr, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    l_prob = lbp_rf.predict_proba([extract_lbp(frame_rgb)])[0][1]
    
    # Late Fusion
    avg = (t_prob + l_prob) / 2
    pred = 1 if avg > 0.5 else 0
    
    all_preds.append(pred)
    all_labels.append(0)  # Real = 0
    
    if (i + 1) % 20 == 0:
        print(f"  Real: {i+1}/{len(real_files)}")

# Attack (Spoof)
for i, video_path in enumerate(attack_files):
    frame = read_video_frame(video_path)
    
    if frame is None:
        failed_files.append(('attack', video_path))
        continue
    
    # Texture
    x = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        t_prob = F.softmax(texture(x), dim=1)[0][1].item()
    
    # LBP
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_resized = cv2.resize(frame_bgr, (224, 224))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    l_prob = lbp_rf.predict_proba([extract_lbp(frame_rgb)])[0][1]
    
    # Late Fusion
    avg = (t_prob + l_prob) / 2
    pred = 1 if avg > 0.5 else 0
    
    all_preds.append(pred)
    all_labels.append(1)  # Attack = 1
    
    if (i + 1) % 50 == 0:
        print(f"  Attack: {i+1}/{len(attack_files)}")

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# 결과
correct = (all_preds == all_labels).sum()
total = len(all_labels)

live_mask = all_labels == 0
spoof_mask = all_labels == 1

live_correct = ((all_preds == 0) & live_mask).sum()
live_total = live_mask.sum()

spoof_correct = ((all_preds == 1) & spoof_mask).sum()
spoof_total = spoof_mask.sum()

acc = correct / total * 100
live_acc = live_correct / live_total * 100 if live_total > 0 else 0
spoof_acc = spoof_correct / spoof_total * 100 if spoof_total > 0 else 0
far = (live_total - live_correct) / live_total * 100 if live_total > 0 else 0
frr = (spoof_total - spoof_correct) / spoof_total * 100 if spoof_total > 0 else 0
hter = (far + frr) / 2

print("\n" + "="*70)
print("MSU Cross-dataset 성능 (Late Fusion)")
print("="*70)
print(f"Total:   {total} samples")
print(f"  Real:  {live_total}")
print(f"  Attack: {spoof_total}")
print(f"  Failed: {len(failed_files)}")
print()
print(f"Overall: {acc:.2f}% ({correct}/{total})")
print(f"Real:    {live_acc:.2f}% ({live_correct}/{live_total})")
print(f"Attack:  {spoof_acc:.2f}% ({spoof_correct}/{spoof_total})")
print()
print(f"FAR:     {far:.2f}%")
print(f"FRR:     {frr:.2f}%")
print(f"HTER:    {hter:.2f}%")
print("="*70)

if acc == 100.0:
    print("\n⚠️  100% = 환상 가능성!")
    print("   → 데이터 누수 확인 필요")
elif acc > 95:
    print("\n📈 고성능 (95-100%)")
    print("   -> 일반화 우수")
elif acc > 80:
    print(f"\n📊 괜찮은 성능 (80-95%)")
    print("   → Cross-dataset 성능 저하")
else:
    print(f"\n❌ 낮은 성능 (80% 미만)")
    print("   → 일반화 실패")
    print("   → Domain shift 심각")

print("\n" + "="*70)
print("비교")
print("="*70)
print(f"Replay-Attack (In-domain):  98.96%")
print(f"MSU-MFSD (Cross-dataset):   {acc:.2f}%")
print(f"성능 차이:                   {98.96 - acc:+.2f}%p")
print("="*70)

# CSV 저장
import csv
with open('msu_cross_verified.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Dataset', 'Total', 'Real', 'Attack', 'Overall(%)', 
                     'Real(%)', 'Attack(%)', 'FAR(%)', 'FRR(%)', 'HTER(%)'])
    writer.writerow(['MSU-MFSD', total, live_total, spoof_total, 
                     f'{acc:.2f}', f'{live_acc:.2f}', f'{spoof_acc:.2f}',
                     f'{far:.2f}', f'{frr:.2f}', f'{hter:.2f}'])

print(f"\n✅ 저장: msu_cross_verified.csv")

if failed_files:
    print("\n실패한 파일 목록:")
    for label, path in failed_files:
        print(f"  [{label}] {os.path.basename(path)}")
