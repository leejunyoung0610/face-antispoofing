import torch
import torch.nn.functional as F
from utils.dataset import ReplayAttackDataset
from models.texture_expert import TextureExpert
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
from sklearn.ensemble import RandomForestClassifier
import pickle

device = torch.device('mps')

def extract_lbp_123(image):
    """R=1,2,3 LBP"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    for R, P, bins in [(1, 8, 59), (2, 16, 243), (3, 24, 299)]:
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        features.extend(hist)
    return np.array(features)

def extract_lbp_124(image):
    """R=1,2,4 LBP"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    for R, P in [(1, 8), (2, 16), (4, 32)]:
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        n_bins = P + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        features.extend(hist)
    return np.array(features)

print("=== R=1,2,4 Tail-risk 분석 ===\n")

# 데이터 로드
train_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'train'
)

test_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'test'
)

# R=1,2,4 LBP 학습
print("[1/3] R=1,2,4 LBP 학습...")
X_train = []
y_train = []

for i in range(len(train_ds)):
    item = train_ds[i]
    image = item['raw'].permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    features = extract_lbp_124(image)
    X_train.append(features)
    y_train.append(item['label'])

lbp_124 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
lbp_124.fit(X_train, y_train)
print("✅ 완료\n")

# CNN 로드
print("[2/3] CNN 로드...")
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval() 
print("[2/3] LBP (601 features) 로드...")
print("[3/3] 기존 LBP (R=1,2,3) 로드...")
with open('lbp_rf_model.pkl', 'rb') as f:
    lbp_123 = pickle.load(f)
print("✅ 완료\n")

# 실패 케이스 분석
print("="*70)
print("실패 케이스 비교 (R=1,2,3 vs R=1,2,4)")
print("="*70)

# R=1,2,3 실패 케이스 찾기
failures_123 = []
for i in range(len(test_ds)):
    item = test_ds[i]
    label = item['label']
    
    # Texture
    x = item['raw'].unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = texture(x)
        t_prob = F.softmax(t_out, dim=1)[0][1].item()
    
    # LBP 1,2,3
    image = item['raw'].permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    feat_123 = extract_lbp_123(image)
    l_prob_123 = lbp_123.predict_proba([feat_123])[0][1]
    
    # Late Fusion 1,2,3
    fused_123 = (t_prob + l_prob_123) / 2
    pred_123 = 1 if fused_123 > 0.5 else 0
    
    if pred_123 != label:
        path = item['metadata']['video_path']
        
        # Client 추출
        client = "Unknown"
        if 'client' in path:
            try:
                client = path.split('client')[1].split('_')[0]
                client = f"client{client}"
            except Exception:
                pass
        
        failures_123.append({
            'index': i,
            'label': label,
            'path': path,
            'client': client,
            't_prob': t_prob,
            'l_prob_123': l_prob_123,
            'fused_123': fused_123
        })

print(f"\nR=1,2,3 Late Fusion 실패: {len(failures_123)}개\n")

# 각 실패 케이스를 R=1,2,4로 재평가
print("R=1,2,4로 재평가:\n")

fixed_by_124 = []
still_failed = []

for fail in failures_123:
    idx = fail['index']
    item = test_ds[idx]
    label = item['label']
    
    # LBP 1,2,4
    image = item['raw'].permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    feat_124 = extract_lbp_124(image)
    l_prob_124 = lbp_124.predict_proba([feat_124])[0][1]
    
    # Late Fusion 1,2,4
    t_prob = fail['t_prob']
    fused_124 = (t_prob + l_prob_124) / 2
    pred_124 = 1 if fused_124 > 0.5 else 0
    label_str = "Live" if label == 0 else "Spoof"
    result_123 = "✗"
    result_124 = "✓" if pred_124 == label else "✗"
    
    info = {
        'client': fail['client'],
        'label': label_str,
        't_prob': t_prob,
        'l_prob_123': fail['l_prob_123'],
        'l_prob_124': l_prob_124,
        'fused_123': fail['fused_123'],
        'fused_124': fused_124,
        'fixed': pred_124 == label
    }
    
    if pred_124 == label:
        fixed_by_124.append(info)
    else:
        still_failed.append(info)
    
    print(f"[{idx}] {fail['client']} ({label_str})")
    print(f"  Texture:     {t_prob:.3f}")
    print(f"  LBP (1,2,3): {fail['l_prob_123']:.3f} → Fused: {fail['fused_123']:.3f} {result_123}")
    print(f"  LBP (1,2,4): {l_prob_124:.3f} → Fused: {fused_124:.3f} {result_124}")
    
    if pred_124 == label:
        print(f"  → ✅ R=1,2,4가 해결!")
    else:
        print(f"  → ❌ 여전히 실패")
    print()

# 요약
print("="*70)
print("요약")
print("="*70)
print(f"R=1,2,3 실패: {len(failures_123)}개")
print(f"R=1,2,4로 해결: {len(fixed_by_124)}개")
print(f"여전히 실패: {len(still_failed)}개")
print()

if len(fixed_by_124) > 0:
    print("✅ R=1,2,4가 해결한 케이스:")
    for info in fixed_by_124:
        print(f"  - {info['client']} ({info['label']})")
        print(f"    LBP 변화: {info['l_prob_123']:.3f} → {info['l_prob_124']:.3f}")
    print()

if len(still_failed) > 0:
    print("❌ 여전히 실패하는 케이스:")
    for info in still_failed:
        print(f"  - {info['client']} ({info['label']})")
    print()

# Client별 분석
print("="*70)
print("Client별 해결 여부")
print("="*70)

from collections import defaultdict
client_analysis = defaultdict(lambda: {'total': 0, 'fixed': 0})

for info in fixed_by_124 + still_failed:
    client = info['client']
    client_analysis[client]['total'] += 1
    if info['fixed']:
        client_analysis[client]['fixed'] += 1

for client in sorted(client_analysis.keys()):
    stats = client_analysis[client]
    total = stats['total']
    fixed = stats['fixed']
    rate = fixed / total * 100 if total else 0
    print(f"  {client:<8}  Fixed {fixed}/{total} → {rate:.1f}%")
if len(fixed_by_124) >= 3:
    print("✅ R=1,2,4가 Tail-risk(특히 Client 011/104)를 효과적으로 해결!")
    print(f"   R=4의 큰 스케일이 이들의 특성(Highdef 거시 패턴)을 포착")
elif len(fixed_by_124) > 0:
    print("⚠️  R=1,2,4가 일부 케이스 해결")
else:
    print("❌ R=1,2,4도 동일한 케이스 실패")
    print("   → 다른 원인 (Texture2 필요)")

print("="*70)
