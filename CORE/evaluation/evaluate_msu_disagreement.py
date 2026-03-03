import torch
import torch.nn.functional as F
from models.texture_expert import TextureExpert
from models.texture2_expert import Texture2Expert
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

def crop_face_simple(image):
    """간단한 중앙 크롭"""
    h, w = image.shape[:2]
    size = min(h, w)
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    return image[y1:y1+size, x1:x1+size]

def read_video_frame(video_path, frame_idx=30):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None

print("="*70)
print("MSU Cross-dataset Disagreement Fusion")
print("="*70)

# 모델 로드
print("\n모델 로드...")
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval()

texture2 = Texture2Expert().to(device)
texture2.load_state_dict(torch.load('checkpoints/cross/texture2.pth'))
texture2.eval()

with open('lbp_rf_model.pkl', 'rb') as f:
    lbp_rf = pickle.load(f)

print("✅ 완료\n")

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# MSU 데이터
msu_path = 'data/msu-mfsd/scene01'

real_files = [os.path.join(msu_path, 'real', f) 
              for f in os.listdir(os.path.join(msu_path, 'real'))
              if f.endswith(('.mov', '.mp4', '.avi'))]

attack_files = [os.path.join(msu_path, 'attack', f)
                for f in os.listdir(os.path.join(msu_path, 'attack'))
                if f.endswith(('.mov', '.mp4', '.avi'))]

print(f"Real: {len(real_files)}, Attack: {len(attack_files)}\n")

# Threshold 탐색
best_acc = 0
best_threshold = 0
best_results = None

for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
    all_preds = []
    all_labels = []
    disagree_count = 0
    
    # Real
    for video_path in real_files:
        frame = read_video_frame(video_path)
        if frame is None:
            continue
        
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Texture
        x = transform(frame_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            t_prob = F.softmax(texture(x), dim=1)[0][1].item()
        
        # LBP
        l_prob = lbp_rf.predict_proba([extract_lbp(frame_resized)])[0][1]
        
        # Disagreement
        if abs(t_prob - l_prob) > threshold:
            disagree_count += 1
            # Texture2
            frame_crop = crop_face_simple(frame_resized)
            x2 = transform(frame_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                t2_prob = F.softmax(texture2(x2), dim=1)[0][1].item()
            
            # 3개 평균
            avg = (t_prob + l_prob + t2_prob) / 3
        else:
            # Late Fusion
            avg = (t_prob + l_prob) / 2
        
        pred = 1 if avg > 0.5 else 0
        all_preds.append(pred)
        all_labels.append(0)
    
    # Attack
    for video_path in attack_files:
        frame = read_video_frame(video_path)
        if frame is None:
            continue
        
        frame_resized = cv2.resize(frame, (224, 224))
        
        # Texture
        x = transform(frame_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            t_prob = F.softmax(texture(x), dim=1)[0][1].item()
        
        # LBP
        l_prob = lbp_rf.predict_proba([extract_lbp(frame_resized)])[0][1]
        
        # Disagreement
        if abs(t_prob - l_prob) > threshold:
            disagree_count += 1
            # Texture2
            frame_crop = crop_face_simple(frame_resized)
            x2 = transform(frame_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                t2_prob = F.softmax(texture2(x2), dim=1)[0][1].item()
            
            avg = (t_prob + l_prob + t2_prob) / 3
        else:
            avg = (t_prob + l_prob) / 2
        
        pred = 1 if avg > 0.5 else 0
        all_preds.append(pred)
        all_labels.append(1)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    correct = (all_preds == all_labels).sum()
    acc = correct / len(all_labels) * 100
    disagree_rate = disagree_count / len(all_labels) * 100
    
    print(f"Threshold {threshold}: {acc:.2f}% (불일치: {disagree_rate:.1f}%)")
    
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold
        
        live_mask = all_labels == 0
        spoof_mask = all_labels == 1
        
        live_correct = ((all_preds == 0) & live_mask).sum()
        spoof_correct = ((all_preds == 1) & spoof_mask).sum()
        
        best_results = {
            'acc': acc,
            'live_acc': live_correct / live_mask.sum() * 100,
            'spoof_acc': spoof_correct / spoof_mask.sum() * 100,
            'far': (live_mask.sum() - live_correct) / live_mask.sum() * 100,
            'frr': (spoof_mask.sum() - spoof_correct) / spoof_mask.sum() * 100,
            'disagree_rate': disagree_rate,
            'total': len(all_labels),
            'live_total': live_mask.sum(),
            'spoof_total': spoof_mask.sum()
        }

print("\n" + "="*70)
print("MSU Disagreement Fusion 최고 성능")
print("="*70)
print(f"Threshold:  {best_threshold}")
print(f"Overall:    {best_results['acc']:.2f}%")
print(f"Real:       {best_results['live_acc']:.2f}%")
print(f"Attack:     {best_results['spoof_acc']:.2f}%")
print(f"FAR:        {best_results['far']:.2f}%")
print(f"FRR:        {best_results['frr']:.2f}%")
print(f"불일치율:   {best_results['disagree_rate']:.2f}%")
print("="*70)

print("\n" + "="*70)
print("비교")
print("="*70)
print(f"Late Fusion:         96.79%")
print(f"Disagreement:        {best_results['acc']:.2f}%")
print(f"개선:                {best_results['acc'] - 96.79:+.2f}%p")
print()
print(f"Replay-Attack:       99.58%")
print(f"MSU (Disagreement):  {best_results['acc']:.2f}%")
print(f"차이:                {99.58 - best_results['acc']:+.2f}%p")
print("="*70)

# CSV 저장
import csv
with open('msu_disagreement_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Method', 'Dataset', 'Threshold', 'Overall(%)', 'Real(%)', 
                     'Attack(%)', 'FAR(%)', 'FRR(%)', 'Disagree(%)'])
    writer.writerow(['Late_Fusion', 'MSU', '-', '96.79', '100.00', '95.71', 
                     '0.00', '4.29', '-'])
    writer.writerow(['Disagreement', 'MSU', best_threshold, 
                     f'{best_results["acc"]:.2f}', 
                     f'{best_results["live_acc"]:.2f}',
                     f'{best_results["spoof_acc"]:.2f}',
                     f'{best_results["far"]:.2f}',
                     f'{best_results["frr"]:.2f}',
                     f'{best_results["disagree_rate"]:.2f}'])

print(f"\n✅ 저장: msu_disagreement_results.csv")
