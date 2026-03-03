import torch
import torch.nn.functional as F
from utils.dataset import ReplayAttackDataset
from models.texture_expert import TextureExpert
from models.texture2_expert import Texture2Expert
import pickle
import numpy as np
from skimage.feature import local_binary_pattern
import cv2

device = torch.device('mps')

def extract_multiscale_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    for R, P, bins in [(1, 8, 59), (2, 16, 243), (3, 24, 299)]:
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        features.extend(hist)
    return np.array(features)

print("=== Disagreement 기반 전략 테스트 ===\n")

# 모델 로드
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval()

texture2 = Texture2Expert().to(device)
texture2.load_state_dict(torch.load('checkpoints/cross/texture2.pth'))
texture2.eval()

with open('lbp_rf_model.pkl', 'rb') as f:
    lbp_rf = pickle.load(f)

test_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'test_full'
)

print(f"전체 샘플: {len(test_ds)}\n")

# 여러 threshold 테스트
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.61]

for threshold in thresholds:
    print(f"Threshold: {threshold:.1f}")
    
    correct = 0
    total = 0
    live_correct = 0
    live_total = 0
    spoof_correct = 0
    spoof_total = 0
    live_fp = 0
    spoof_fn = 0
    
    texture2_used = 0
    
    for i in range(len(test_ds)):
        item = test_ds[i]
        label = item['label']
        
        # Texture
        x = item['raw'].unsqueeze(0).to(device)
        with torch.no_grad():
            t_out = texture(x)
            t_prob = F.softmax(t_out, dim=1)[0][1].item()
        
        # LBP
        image = item['raw'].permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        lbp_feat = extract_multiscale_lbp(image)
        l_prob = lbp_rf.predict_proba([lbp_feat])[0][1]
        
        # Disagreement
        disagreement = abs(t_prob - l_prob)
        
        # 전략
        if disagreement > threshold:
            # 불일치 큼 → Texture2 추가
            texture2_used += 1
            with torch.no_grad():
                t2_out = texture2(x)
                t2_prob = F.softmax(t2_out, dim=1)[0][1].item()
            fused = (t_prob + l_prob + t2_prob) / 3
        else:
            # 일치 → 2-way
            fused = (t_prob + l_prob) / 2
        
        pred = 1 if fused > 0.5 else 0
        
        total += 1
        if pred == label:
            correct += 1
        
        if label == 0:
            live_total += 1
            if pred == label:
                live_correct += 1
            else:
                live_fp += 1
        else:
            spoof_total += 1
            if pred == label:
                spoof_correct += 1
            else:
                spoof_fn += 1
    
    acc = correct / total * 100 if total > 0 else 0
    live_acc = live_correct / live_total * 100 if live_total > 0 else 0
    spoof_acc = spoof_correct / spoof_total * 100 if spoof_total > 0 else 0
    far = live_fp / live_total * 100 if live_total > 0 else 0
    frr = spoof_fn / spoof_total * 100 if spoof_total > 0 else 0
    hter = (far + frr) / 2
    usage = texture2_used / total * 100
    
    print(f"  Overall: {acc:.2f}% ({correct}/{total})")
    print(f"  Live:    {live_acc:.2f}% ({live_correct}/{live_total})")
    print(f"  Spoof:   {spoof_acc:.2f}% ({spoof_correct}/{spoof_total})")
    print(f"  FAR:     {far:.2f}%")
    print(f"  FRR:     {frr:.2f}%")
    print(f"  HTER:    {hter:.2f}%")
    print(f"  Texture2: {usage:.2f}% ({texture2_used}/{total})")
    print()

print("="*60)
print("Late Fusion (기존) 비교:")
print("  Overall: 98.96%")
print("  HTER:    1.12%")
print("  Texture2: 0%")
print("="*60)
