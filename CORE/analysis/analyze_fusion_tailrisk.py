import torch
import torch.nn.functional as F
from utils.dataset import ReplayAttackDataset
from models.texture_expert import TextureExpert
import pickle
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

device = torch.device('mps')

def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    for R, P, bins in [(1, 8, 59), (2, 16, 243), (3, 24, 299)]:
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        features.extend(hist)
    return np.array(features)

def image_to_fft(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    magnitude_log = np.log(magnitude + 1)
    magnitude_log = (magnitude_log - magnitude_log.min()) / (magnitude_log.max() - magnitude_log.min() + 1e-8)
    fft_image = np.stack([magnitude_log] * 3, axis=0)
    
    return fft_image

print("="*70)
print("Tail-risk 비교 분석")
print("="*70)

# 모델 로드
print("\n모델 로드...")
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval()

# Frequency 모델
from torchvision.models import efficientnet_b0
import torch.nn as nn

class TrueFrequencyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(pretrained=False)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 2)
    
    def forward(self, x):
        return self.backbone(x)

frequency = TrueFrequencyExpert().to(device)
frequency.load_state_dict(torch.load('true_frequency_expert_best.pth'))
frequency.eval()

with open('lbp_rf_model.pkl', 'rb') as f:
    lbp_rf = pickle.load(f)
test_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'test_full'
)

print(f"Test: {len(test_ds)}\n")

# 확률 계산
print("확률 계산 중...")
texture_probs = []
frequency_probs = []
lbp_probs = []
labels = []
clients = []
videos = []

for i in range(len(test_ds)):
    item = test_ds[i]
    
    # Texture
    x = item['raw'].unsqueeze(0).to(device)
    with torch.no_grad():
        t_prob = F.softmax(texture(x), dim=1)[0][1].item()
    
    # Frequency
    image_np = item['raw'].permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    fft_img = image_to_fft(image_np)
    x_fft = torch.FloatTensor(fft_img).unsqueeze(0).to(device)
    with torch.no_grad():
        f_prob = F.softmax(frequency(x_fft), dim=1)[0][1].item()
    
    # LBP
    l_prob = lbp_rf.predict_proba([extract_lbp(image_np)])[0][1]
    
    texture_probs.append(t_prob)
    frequency_probs.append(f_prob)
    lbp_probs.append(l_prob)
    labels.append(item['label'])
    path = str(test_ds.samples[i])
    parts = path.split('/')
    for part in parts:
        if 'client' in part.lower():
            clients.append(part)
            break
    else:
        clients.append('unknown')
    
    videos.append(path.split('/')[-1] if '/' in path else 'unknown')
    
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(test_ds)}")

texture_probs = np.array(texture_probs)
frequency_probs = np.array(frequency_probs)
lbp_probs = np.array(lbp_probs)
labels = np.array(labels)

print("✅ 완료\n")

# Fusion 방법들
methods = {}

# 1. T+F Weighted 0.7
preds = ((0.7*texture_probs + 0.3*frequency_probs) > 0.5).astype(int)
methods['T+F_Weight_0.7'] = preds

# 2. T+F Weighted 0.8
preds = ((0.8*texture_probs + 0.2*frequency_probs) > 0.5).astype(int)
methods['T+F_Weight_0.8'] = preds

# 3. T+F Cascading
preds = []
for i in range(len(texture_probs)):
    if texture_probs[i] > 0.7 or texture_probs[i] < 0.3:
        pred = 1 if texture_probs[i] > 0.5 else 0
    else:
        avg = (texture_probs[i] + frequency_probs[i]) / 2
        pred = 1 if avg > 0.5 else 0
    preds.append(pred)
methods['T+F_Cascading'] = np.array(preds)

# 4. T+L Late Fusion (기존)
preds = ((texture_probs + lbp_probs) / 2 > 0.5).astype(int)
methods['T+L_Late_Fusion'] = preds

# 실패 케이스 분석
print("="*70)
print("실패 케이스 분석")
print("="*70)

failures = {}
for name, preds in methods.items():
    fail_idx = np.where(preds != labels)[0]
    failures[name] = {
        'indices': fail_idx,
        'count': len(fail_idx),
        'clients': [clients[i] for i in fail_idx],
        'videos': [videos[i] for i in fail_idx],
        'labels': [labels[i] for i in fail_idx]
    }
    
    acc = (preds == labels).mean() * 100
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.2f}%")
    print(f"  Failures: {len(fail_idx)}")

# Tail-risk 동일성 체크
print("\n" + "="*70)
print("Tail-risk 동일성 분석")
print("="*70)

fail_sets = {name: set(info['indices']) for name, info in failures.items()}

# 모든  조합 비교
from itertools import combinations

print("\n교집합 분석:")
for m1, m2 in combinations(methods.keys(), 2):
    intersection = fail_sets[m1] & fail_sets[m2]
    union = fail_sets[m1] | fail_sets[m2]
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0
    
    print(f"\n{m1} ∩ {m2}:")
    print(f"  공통 실패: {len(intersection)}개")
    print(f"  Jaccard: {jaccard:.2%}")
    
    if jaccard > 0.8:
        print(f"  → ✅ 거의 동일!")
    elif jaccard > 0.5:
        print(f"  → ⚠️  유사함")
    else:
        print(f"  → ❌ 다름")

# 가장 많이 실패하는 샘플
print("\n" + "="*70)
print("빈도별 실패 샘플")
print("="*70)

all_failures = []
for name, info in failures.items():
    all_failures.extend(info['indices'])

failure_counts = Counter(all_failures)
most_common = failure_counts.most_common(10)

print("\n가장 자주 실패하는 샘플 (Top 10):")
for idx, count in most_common:
    print(f"\nIndex {idx}: {count}/4 방법에서 실패")
    print(f"  Client: {clients[idx]}")
    print(f"  Video: {videos[idx]}")
    print(f"  Label: {'Spoof' if labels[idx] == 1 else 'Live'}")
    print(f"  실패 방법: {[name for name, info in failures.items() if idx in info['indices']]}")

# Client별 집중도
print("\n" + "="*70)
print("Client별 실패 분포")
print("="*70)

for name, info in failures.items():
    client_counts = Counter(info['clients'])
    print(f"\n{name}:")
    for client, count in client_counts.most_common(5):
        print(f"  {client}: {count}개 ({count/len(info['indices'])*100:.1f}%)")

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 왼쪽: Venn-like 비교
method_names = list(methods.keys())
failure_counts_by_method = [len(failures[m]['indices']) for m in method_names]

bars = ax1.barh(method_names, failure_counts_by_method, 
                color=['#e74c3c', '#3498db', '#f39c12', '#2ecc71'])
ax1.set_xlabel('실패 케이스 수', fontsize=12, fontweight='bold')
ax1.set_title('Fusion 실패 케이스 수', alpha=0.3)

for bar, val in zip(bars, failure_counts_by_method):
    ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
             f'{val}개', va='center', fontsize=10, fontweight='bold')

# 오른쪽: Client 분포 (T+L Late Fusion)
late_fusion_clients = Counter(failures['T+L_Late_Fusion']['clients'])
top_clients = late_fusion_clients.most_common(10)

if top_clients:
    client_names = [c[0] for c in top_clients]
    client_vals = [c[1] for c in top_clients]
    
    bars2 = ax2.barh(client_names, client_vals, color='#2ecc71')
    ax2.set_xlabel('실패 케이스 수', fontsize=12, fontweight='bold')
    ax2.set_title('T+L Late Fusion Client별 실패 분포', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars2, client_vals):
        ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                 f'{val}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('fusion_tailrisk_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ 저장: fusion_tailrisk_comparison.png")

# CSV 저장
df_failures = []
for name, info in failures.items():
    for i, idx in enumerate(info['indices']):
        df_failures.append({
            'Method': name,
            'Index': idx,
            'Client': info['clients'][i],
            'Video': info['videos'][i],
            'Label': 'Spoof' if info['labels'][i] == 1 else 'Live'
        })

df = pd.DataFrame(df_failures)
df.to_csv('fusion_tailrisk_analysis.csv', index=False)
print(f"✅ 저장: fusion_tailrisk_analysis.csv")

# 요약
print("\n" + "="*70)
print("핵심 발견")
print("="*70)

# T+F 방법들끼리 비교
tf_methods = ['T+F_Weight_0.7', 'T+F_Weight_0.8', 'T+F_Cascading']
tf_intersection = fail_sets[tf_methods[0]]
for m in tf_methods[1:]:
    tf_intersection = tf_intersection & fail_sets[m]

print(f"\nT+F 방법 3개 모두 실패: {len(tf_intersection)}개")

# T+L과 비교
tf_all = fail_sets['T+F_Weight_0.7'] | fail_sets['T+F_Weight_0.8'] | fail_sets['T+F_Cascading']
tl_only = fail_sets['T+L_Late_Fusion'] - tf_all
tf_only = tf_all - fail_sets['T+L_Late_Fusion']
common = fail_sets['T+L_Late_Fusion'] & tf_all

print(f"\nT+L만 실패: {len(tl_only)}개")
print(f"T+F만 실패: {len(tf_only)}개")
print(f"공통 실패: {len(common)}개")

if len(tl_only) == 0 and len(tf_only) > 0:
    print("\n→ T+L이 T+F보다 우수!")
elif len(tf_only) == 0 and len(tl_only) > 0:
    print("\n→ T+F가 T+L보다 우수!")
else:
    print("\n→ 상보적 관계")

print("="*70)
