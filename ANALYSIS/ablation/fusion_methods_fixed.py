import csv
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

device = torch.device('mps')

def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    for R, P, bins in [(1, 8, 59), (2, 16, 243), (3, 24, 299)]:
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        features.extend(hist)
    return np.array(features)

print("=== Fusion 방법 비교 (수정) ===\n")

# 모델 로드
print("모델 로드...")
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval()

with open('lbp_rf_model.pkl', 'rb') as f:
    lbp_rf = pickle.load(f)

# 데이터 (명확히 test_full)
test_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'test_full'
)

print(f"Test samples: {len(test_ds)}\n")

# 모든 확률 미리 계산
print("확률 계산 중...")
texture_probs = []
lbp_probs = []
labels = []

for i in range(len(test_ds)):
    item = test_ds[i]
    
    # Texture
    x = item['raw'].unsqueeze(0).to(device)
    with torch.no_grad():
        t_prob = F.softmax(texture(x), dim=1)[0][1].item()
    
    # LBP
    image = item['raw'].permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    l_prob = lbp_rf.predict_proba([extract_lbp(image)])[0][1]
    
    texture_probs.append(t_prob)
    lbp_probs.append(l_prob)
    labels.append(item['label'])
    
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(test_ds)}")

texture_probs = np.array(texture_probs)
lbp_probs = np.array(lbp_probs)
labels = np.array(labels)

print("✅ 완료\n")

# 다양한 Fusion 방법
methods = {}

# 1. Simple Average (Late Fusion)
avg_probs = (texture_probs + lbp_probs) / 2
preds = (avg_probs > 0.5).astype(int)
acc = (preds == labels).mean() * 100
methods['Simple Average'] = acc

# 2. Max Confidence
max_probs = np.maximum(texture_probs, lbp_probs)
preds = (max_probs > 0.5).astype(int)
acc = (preds == labels).mean() * 100
methods['Max Confidence'] = acc

# 3. Min Confidence (Both agree)
min_probs = np.minimum(texture_probs, lbp_probs)
preds = (min_probs > 0.5).astype(int)
acc = (preds == labels).mean() * 100
methods['Min Confidence'] = acc

# 4. Weighted (Texture 70%)
weighted = 0.7 * texture_probs + 0.3 * lbp_probs
preds = (weighted > 0.5).astype(int)
acc = (preds == labels).mean() * 100
methods['Weighted (0.7:0.3)'] = acc

# 5. Weighted (Texture 80%)
weighted = 0.8 * texture_probs + 0.2 * lbp_probs
preds = (weighted > 0.5).astype(int)
acc = (preds == labels).mean() * 100
methods['Weighted (0.8:0.2)'] = acc

# 6. Product (곱)
product = texture_probs * lbp_probs
preds = (product > 0.5).astype(int)
acc = (preds == labels).mean() * 100
methods['Product'] = acc

# 7. If Either (OR logic)
either = np.maximum(texture_probs, lbp_probs)
preds = (either > 0.3).astype(int)  # Lower threshold
acc = (preds == labels).mean() * 100
methods['Either (OR, t=0.3)'] = acc

# 결과 출력
print("="*60)
print("결합 방법 성능 비교")
print("="*60)
print(f"{'Method':<25} {'Accuracy':<10}")
print("-"*60)
print(f"{'Texture CNN (단독)':<25} {'98.57%':<10}")
print(f"{'Multi-LBP (단독)':<25} {'97.08%':<10}")
print("-"*60)

sorted_methods = sorted(methods.items(), key=lambda x: x[1], reverse=True)
for method, acc in sorted_methods:
    print(f"{method:<25} {acc:>7.2f}%")
print("="*60)

# 시각화 1: 막대 그래프
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 왼쪽: 전체 비교
all_methods = {
    'Texture\nCNN': 98.57,
    'Multi-\nLBP': 97.08,
}
all_methods.update(methods)

names = list(all_methods.keys())
values = list(all_methods.values())
colors = ['#2ecc71' if i < 2 else '#3498db' for i in range(len(names))]

bars = ax1.barh(names, values, color=colors)
ax1.set_xlabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax1.set_xlim(96, 100)
ax1.axvline(98.57, color='red', linestyle='--', alpha=0.5, label='Texture CNN')
ax1.grid(axis='x', alpha=0.3)
ax1.legend()

# 값 표시
for bar, val in zip(bars, values):
    ax1.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}%', va='center', fontsize=10)

# 오른쪽: Fusion만 비교
fusion_names = list(methods.keys())
fusion_values = list(methods.values())
colors2 = ['#e74c3c' if v == max(fusion_values) else '#3498db' 
           for v in fusion_values]

bars2 = ax2.barh(fusion_names, fusion_values, color=colors2)
ax2.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Fusion Methods Only', fontsize=14, fontweight='bold')
ax2.set_xlim(96, 100)
ax2.axvline(98.57, color='green', linestyle='--', alpha=0.5, 
            label='Target (Texture)')
ax2.grid(axis='x', alpha=0.3)
ax2.legend()

# 값 표시
for bar, val in zip(bars2, fusion_values):
    ax2.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('fusion_methods_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ 저장: fusion_methods_comparison.png")

# 시각화 2: 표 이미지
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('tight')
ax.axis('off')

# 데이터 준비
table_data = [
    ['Method', 'Accuracy', 'vs Texture', 'Note'],
    ['', '', '', ''],
    ['Texture CNN', '98.57%', '-', 'Baseline'],
    ['Multi-LBP', '97.08%', '-1.49%p', 'Baseline'],
    ['', '', '', ''],
]

for method, acc in sorted_methods:
    diff = acc - 98.57
    sign = '+' if diff > 0 else ''
    note = '⭐ Best' if acc == max(fusion_values) else ''
    if abs(diff) < 0.1:
        note = '≈ Same'
    elif diff < -0.5:
        note = 'Worse'
    
    table_data.append([
        method,
        f'{acc:.2f}%',
        f'{sign}{diff:.2f}%p',
        note
    ])

# 색상 설정
cell_colors = []
for i, row in enumerate(table_data):
    if i == 0:  # Header
        cell_colors.append(['#34495e'] * 4)
    elif i == 2:  # Texture
        cell_colors.append(['#2ecc71'] * 4)
    elif i == 3:  # LBP
        cell_colors.append(['#95a5a6'] * 4)
    elif i == 2:  # Texture
        cell_colors.append(['#2ecc71'] * 4)
    elif i == 3:  # LBP
        cell_colors.append(['#95a5a6'] * 4)
    else:
        acc_val = float(row[1].replace('%', ''))
        if acc_val == max(fusion_values):
            cell_colors.append(['#e74c3c'] * 4)  # Best
        elif acc_val > 98.5:
            cell_colors.append(['#f39c12'] * 4)  # Good
        else:
            cell_colors.append(['#ffffff'] * 4)  # Normal

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                cellColours=cell_colors)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Header 스타일
for i in range(4):
    cell = table[(0, i)]
    cell.set_text_props(weight='bold', color='white')

plt.title('Fusion Methods Performance Table', 
          fontsize=16, fontweight='bold', pad=20)

plt.savefig('fusion_methods_table.png', dpi=150, bbox_inches='tight')
print(f"✅ 저장: fusion_methods_table.png")
with open('fusion_methods_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Method', 'Accuracy(%)', 'Diff_vs_Texture(%p)'])
    writer.writerow(['Texture_CNN', '98.57', '0.00'])
    writer.writerow(['Multi-LBP', '97.08', '-1.49'])
    
    for method, acc in sorted_methods:
        diff = acc - 98.57
        writer.writerow([method.replace(' ', '_'), f'{acc:.2f}', f'{diff:.2f}'])

print(f"✅ 저장: fusion_methods_results.csv")

# 분석
print("\n" + "="*60)
print("분석")
print("="*60)
best_method, best_acc = max(methods.items(), key=lambda x: x[1])
print(f"최고 Fusion: {best_method} ({best_acc:.2f}%)")
print(f"Texture 단독: 98.57%")
print(f"차이: {best_acc - 98.57:.2f}%p")

if best_acc > 98.57:
    print("\n✅ Fusion이 단독보다 나음!")
elif abs(best_acc - 98.57) < 0.2:
    print("\n⚠️  Fusion과 단독 성능 유사")
    print("   → 유의미한 개선 없음")
else:
    print("\n❌ Fusion이 단독보다 못함")
    print("   → Late Fusion 필요")

print("="*60)
