import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
from utils.dataset import ReplayAttackDataset
from models.texture_expert import TextureExpert
import pickle
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device('mps')

# Frequency 모델
class TrueFrequencyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(pretrained=False)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, 2)
    
    def forward(self, x):
        return self.backbone(x)

# FFT 변환
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

# LBP 추출
def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    for R, P, bins in [(1, 8, 59), (2, 16, 243), (3, 24, 299)]:
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        features.extend(hist)
    return np.array(features)

print("="*70)
print("완전한 Fusion 테스트")
print("="*70)

# 모델 로드
print("\n모델 로드 중...")
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval()

frequency = TrueFrequencyExpert().to(device)
frequency.load_state_dict(torch.load('true_frequency_expert_best.pth'))
frequency.eval()

with open('lbp_rf_model.pkl', 'rb') as f:
    lbp_rf = pickle.load(f)

print("✅ 완료\n")

# 데이터
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
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(test_ds)}")

texture_probs = np.array(texture_probs)
frequency_probs = np.array(frequency_probs)
lbp_probs = np.array(lbp_probs)
labels = np.array(labels)

print("✅ 완료\n")

# Fusion 테스트
results = {}

print("="*70)
print("단독 성능")
print("="*70)

t_acc = ((texture_probs > 0.5).astype(int) == labels).mean() * 100
f_acc = ((frequency_probs > 0.5).astype(int) == labels).mean() * 100
l_acc = ((lbp_probs > 0.5).astype(int) == labels).mean() * 100

results['Texture (단독)'] = t_acc
results['Frequency (단독)'] = f_acc
results['LBP (단독)'] = l_acc

print(f"Texture:   {t_acc:.2f}%")
print(f"Frequency: {f_acc:.2f}%")
print(f"LBP:       {l_acc:.2f}%")

print("\n" + "="*70)
print("Texture + Frequency Fusion")
print("="*70)

# 1. Two-Expert (Simple Average)
preds = ((texture_probs + frequency_probs) / 2 > 0.5).astype(int)
acc = (preds == labels).mean() * 100
results['T+F: Two-Expert'] = acc
print(f"1. Two-Expert (Avg):      {acc:.2f}%")

# 2. Weighted (Texture 우선 70%)
preds = ((0.7*texture_probs + 0.3*frequency_probs) > 0.5).astype(int)
acc = (preds == labels).mean() * 100
results['T+F: Weighted(0.7)'] = acc
print(f"2. Weighted (0.7:0.3):    {acc:.2f}%")

# 3. Weighted (Texture 우선 80%)
preds = ((0.8*texture_probs + 0.2*frequency_probs) > 0.5).astype(int)
acc = (preds == labels).mean() * 100
results['T+F: Weighted(0.8)'] = acc
print(f"3. Weighted (0.8:0.2):    {acc:.2f}%")

# 4. Cascading (Texture 확신도 먼저)
correct = 0
cascade_used = 0
for i in range(len(texture_probs)):
    if texture_probs[i] > 0.7 or texture_probs[i] < 0.3:
        pred = 1 if texture_probs[i] > 0.5 else 0
    else:
        cascade_used += 1
        avg = (texture_probs[i] + frequency_probs[i]) / 2
        pred = 1 if avg > 0.5 else 0
    if pred == labels[i]:
        correct += 1
acc = correct / len(labels) * 100
cascade_rate = cascade_used / len(labels) * 100
results['T+F: Cascading'] = acc
print(f"4. Cascading:             {acc:.2f}% (Stage2: {cascade_rate:.1f}%)")

# 5 (높은 확신도 선택)
correct = 0
for i in range(len(texture_probs)):
    t_conf = abs(texture_probs[i] - 0.5)
    f_conf = abs(frequency_probs[i] - 0.5)
    
    if t_conf > f_conf:
        pred = 1 if texture_probs[i] > 0.5 else 0
    else:
        pred = 1 if frequency_probs[i] > 0.5 else 0
    
    if pred == labels[i]:
        correct += 1
acc = correct / len(labels) * 100
results['T+F: Confidence'] = acc
print(f"5. Confidence-Based:      {acc:.2f}%")

# 6. Max Confidence
max_probs = np.maximum(texture_probs, frequency_probs)
preds = (max_probs > 0.5).astype(int)
acc = (preds == labels).mean() * 100
results['T+F: Max'] = acc
print(f"6. Max Confidence:        {acc:.2f}%")

# 7. Product Rule
product = texture_probs * frequency_probs
preds = (product > 0.5).astype(int)
acc = (preds == labels).mean() * 100
results['T+F: Product'] = acc
print(f"7. Product Rule:          {acc:.2f}%")

print("\n" + "="*70)
print("Texture + LBP Fusion (비교)")
print("="*70)

# 8. T+L Two-Expert
preds = ((texture_probs + lbp_probs) / 2 > 0.5).astype(int)
acc = (preds == labels).mean() * 100
results['T+L: Two-Expert'] = acc
print(f"8. Two-Expert (Avg):      {acc:.2f}%")

# 9. T+L Cascading
correct = 0
for i in range(len(texture_probs)):
    if texture_probs[i] > 0.7 or texture_probs[i] < 0.3:
        pred = 1 if texture_probs[i] > 0.5 else 0
    else:
        avg = (texture_probs[i] + lbp_probs[i]) / 2
        pred = 1 if avg > 0.5 else 0
    if pred == labels[i]:
        correct += 1
acc = correct / len(labels) * 100
results['T+L: Cascading'] = acc
print(f"9. Cascading:             {acc:.2f}%")

# 10. T+L Confidence
correct = 0
for i in range(len(texture_probs)):
    t_conf = abs(texture_probs[i] - 0.5)
    l_conf = abs(lbp_probs[i] - 0.5)
    
    if t_conf > l_conf:
        pred = 1 if texture_probs[i] > 0.5 else 0
    else:
        pred = 1 if lbp_probs[i] > 0.5 else 0
    
    if pred == labels[i]:
        correct += 1
acc = correct / len(labels) * 100
results['T+L: Confidence'] = acc
print(f"10. Confidence-Based:     {acc:.2f}%")

print("\n" + "="*70)
print("결론")
print("="*70)

best_tf = max([results[k] for k in results if 'T+F' in k])
best_tl = max([results[k] for k in results if 'T+L' in k])

print(f"Texture 단독:         {t_acc:.2f}%")
print(f"T+F 최고:             {best_tf:.2f}%")
print(f"T+L 최고:             {best_tl:.2f}%")
print("-"*70)

if best_tf < t_acc:
    print("❌ Frequency 결합 → 성능 저하!")
    print("   → '약한 모델과 결합 = 독' 증명")
else:
    print("⚠️  Frequency 결합 → 유사")

if best_tl > t_acc:
    print("✅ LBP 결합 → 성능 향상!")
else:
    print("⚠️  LBP 결합 → 유사")

print("="*70)

# 시각화 1: 전체 비교
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# 왼쪽: 전체
all_methods = list(results.keys())
all_values = list(results.values())

colors = []
for m in all_methods:
    if '단독' in m:
        if 'Texture' in m:
            colors.append('#2ecc71')
        elif 'Frequency' in m:
            colors.append('#e74c3c')
        else:
            colors.append('#95a5a6')
    elif 'T+F' in m:
        colors.append('#f39c12')
    else:
        colors.append('#3498db')

bars = ax1.barh(range(len(all_methods)), all_values, color=colors)
ax1.set_yticks(range(len(all_methods)))
ax1.set_yticklabels(all_methods, fontsize=9)
ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Complete Fusion Architecture Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim(85, 100)
ax1.axvline(t_acc, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Texture ({t_acc:.2f}%)')
ax1.grid(axis='x', alpha=0.3)
ax1.legend()

for bar, val in zip(bars, all_values):
    ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}%', va='center', fontsize=8, fontweight='bold')

# 오른쪽: Fusion 비교
fusion_methods = {k: v for k, v in results.items() if 'T+F' in k or 'T+L' in k}
fusion_names = list(fusion_methods.keys())
fusion_values = list(fusion_methods.values())
fusion_colors = ['#3498db' if 'T+F' in k else '#f39c12' for k in fusion_names]

bars2 = ax2.barh(range(len(fusion_names)), fusion_values, color=fusion_colors)
ax2.set_yticks(range(len(fusion_names)))
ax2.set_yticklabels(fusion_names, fontsize=9)
ax2.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Fusion Methods Only', fontsize=14, fontweight='bold')
ax2.set_xlim(85, 100)
ax2.axvline(t_acc, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Texture ({t_acc:.2f}%)')
ax2.grid(axis='x', alpha=0.3)
ax2.legend()

for bar, val in zip(bars2, fusion_values):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}%', va='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('complete_fusion_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ 저장: complete_fusion_comparison.png")

# 시각화 2: 개선폭
fig, ax = plt.subplots(figsize=(14, 8))

improvements = {k: v - t_acc for k, v in fusion_methods.items()}
imp_names = list(improvements.keys())
imp_values = list(improvements.values())
imp_colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in imp_values]

bars = ax.barh(range(len(imp_names)), imp_valurs)
ax.set_yticks(range(len(imp_names)))
ax.set_yticklabels(imp_names, fontsize=10)
ax.set_xlabel('Improvement vs Texture CNN (%p)', fontsize=12, fontweight='bold')
ax.set_title('Fusion Performance Gain/Loss', fontsize=14, fontweight='bold')
ax.axvline(0, color='black', linestyle='-', linewidth=1.5)
ax.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, imp_values):
    x_pos = val + 0.1 if val > 0 else val - 0.1
    ha = 'left' if val > 0 else 'right'
    sign = '+' if val >= 0 else ''
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, 
            f'{sign}{val:.2f}%p', va='center', ha=ha, 
            fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('fusion_improvements.png', dpi=150, bbox_inches='tight')
print(f"✅ 저장: fusion_improvements.png")

# CSV 저장
df = pd.DataFrame(list(results.items()), columns=['Method', 'Accuracy(%)'])
df['vs_Texture(%p)'] = df['Accuracy(%)'] - t_acc
df['Type'] = df['Method'].apply(
    lambda x: 'Baseline' if '단독' in x else ('T+F' if 'T+F' in x else 'T+L')
)
df.to_csv('complete_fusion_results.csv', index=False)
print(f"✅ 저장: complete_fusion_results.csv")

print("\n🎉 완료!")
