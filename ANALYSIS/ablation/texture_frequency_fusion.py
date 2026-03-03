import torch
import torch.nn.functional as F
from utils.dataset import ReplayAttackDataset
from models.texture_expert import TextureExpert
import pickle
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
import matplotlib.pyplot as plt

device = torch.device('mps')

def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    for R, P, bins in [(1, 8, 59), (2, 16, 243), (3, 24, 299)]:
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        features.extend(hist)
    return np.array(features)

print("=== Texture + Frequency Fusion 테스트 ===\n")

# 모델 로드
print("모델 로드...")
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval()

# Frequency (구조 동일하니 TextureExpert 사용)
frequency = TextureExpert().to(device)
frequency.load_state_dict(torch.load('checkpoints/frequency_expert/best_model.pth'))
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
    x = item['raw'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Texture
        t_out = texture(x)
        t_prob = F.softmax(t_out, dim=1)[0][1].item()
        
        # Frequency
        f_out = frequency(x)
        f_prob = F.softmax(f_out, dim=1)[0][1].item()
    
    # LBP
    image = item['raw'].permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    l_prob = lbp_rf.predict_proba([extract_lbp(image)])[0][1]
    
    texture_probs.append(t_prob)
frequency_probs.append(f_prob)
labels.append(item['label'])
    
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(test_ds)}")

texture_probs = np.array(texture_probs)
frequency_probs = np.array(frequency_probs)
lbp_probs = np.array(lbp_probs)
labels = np.array(labels)

print("✅ 완료\n")

# 단독 성능 확인
t_acc = ((texture_probs > 0.5).astype(int) == labels).mean() * 100
f_acc = ((frequency_probs > 0.5).astype(int) == labels).mean() * 100
l_acc = ((lbp_probs > 0.5).astype(int) == labels).mean() * 100

print("="*60)
print("단독 성능")
print("="*60)
print(f"Texture:   {t_acc:.2f}%")
print(f"Frequency: {f_acc:.2f}%")
print(f"LBP:       {l_acc:.2f}%")
print("="*60)

# Fusion 테스트
results = {
    'Texture (단독)': t_acc,
    'Frequency (단독)': f_acc,
    'LBP (단독)': l_acc,
}

print("\n" + "="*60)
print("Texture + Frequency Fusion")
print("="*60)

# 1. Simple Average
preds = ((texture_probs + frequency_probs) / 2 > 0.5).astype(int)
acc = (preds == labels).mean() * 100
results['T+F: Simple Avg'] = acc
print(f"Simple Average:      {acc:.2f}%")
preds = ((0.7*texture_probs + 0.3*frequency_probs) > 0.5).astype(int)
acc = (preds == labels).mean() * 100
results['T+F: Weighted(0.7)'] = acc
print(f"Weighted (0.7:0.3):   {acc:.2f}%")

# 3. Cascading
correct = 0
for i in range(len(texture_probs)):
    if texture_probs[i] > 0.7 or texture_probs[i] < 0.3:
        pred = 1 if texture_probs[i] > 0.5 else 0
    else:
        avg = (texture_probs[i] + frequency_probs[i]) / 2
        pred = 1 if avg > 0.5 else 0
    if pred == labels[i]:
        correct += 1
acc = correct / len(labels) * 100
results['T+F: Cascading'] = acc
print(f"Cascading:            {acc:.2f}%")

print("\n" + "="*60)
print("Texture + LBP Fusion (비교)")
print("="*60)

# 4. T+L Simple
preds = ((texture_probs + lbp_probs) / 2 > 0.5).astype(int)
acc = (preds == labels).mean() * 100
results['T+L: Simple Avg'] = acc
print(f"Simple Average:       {acc:.2f}%")

print("\n" + "="*60)
print("결론")
print("="*60)
print(f"Texture 단독:         {t_acc:.2f}%")
tf_results = [v for k, v in results.items() if 'T+F' in k]
tf_best = max(tf_results) if tf_results else 0
print(f"T+F 최고:             {tf_best:.2f}%")
print(f"T+L:                  {results['T+L: Simple Avg']:.2f}%")
print("="*60)

if max([results[k] for k in results if 'T+F' in k]) < t_acc:
    print("\n❌ Frequency 결합 → 성능 저하!")
    print("   → '약한 모델과 결합 = 독' 증명")
else:
    print("\n⚠️  Frequency 결합 → 유사")

# 시각화
fig, ax = plt.subplots(figsize=(12, 8))

methods = list(results.keys())
values = list(results.values())

colors = []
for m in methods:
    if '단독' in m:
        colors.append('#95a5a6')
    elif 'T+F' in m:
        colors.append('#e74c3c')
    else:
        colors.append('#2ecc71')

bars = ax.barh(methods, values, color=colors)
ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Texture + Frequency vs Texture + LBP', 
             fontsize=14, fontweight='bold')
ax.set_xlim(85, 100)
ax.axvline(t_acc, color='blue', linestyle='--',
           linewidth=2, alpha=0.5, label=f'Texture ({t_acc:.2f}%)')
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, values):
    ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('texture_frequency_fusion_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ 저장: texture_frequency_fusion_comparison.png")

# CSV
import csv
with open('texture_frequency_fusion_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Method', 'Accuracy(%)', 'Type'])
    for method, acc in results.items():
        if '단독' in method:
            type_ = 'Baseline'
        elif 'T+F' in method:
            type_ = 'Texture+Frequency'
        else:
            type_ = 'Texture+LBP'
        writer.writerow([method, f'{acc:.2f}', type_])

print(f"✅ 저장: texture_frequency_fusion_results.csv")
