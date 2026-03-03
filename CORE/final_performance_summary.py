import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

print("="*70)
print("최종 성능 요약 파일 생성")
print("="*70)

# 데이터 정리
data = {
    'Model': [
        'Texture CNN',
        'Multi-scale LBP',
        'Frequency CNN',
        'Texture2 CNN',
        'Late Fusion (T+L)',
        'Disagreement Fusion',
    ],
    'Replay-Attack (%)': [
        98.57,
        97.08,
        89.79,
        95.21,
        98.96,
        99.58,
    ],
    'MSU-MFSD (%)': [
        None,
        None,
        None,
        None,
        96.79,
        99.29,
    ],
    'Features': [
        '150,528 (RGB pixels)',
        '54 (LBP histogram)',
        '150,528 (FFT)',
        '150,528 (RGB face crop)',
        'Texture + LBP',
        'Texture + LBP + Texture2',
    ],
    'Type': [
        'CNN',
        'Traditional ML',
        'CNN',
        'CNN',
        'Ensemble',
        'Ensemble',
    ]
}
df = pd.DataFrame(data)
# CSV 저장
df.to_csv('FINAL_PERFORMANCE_SUMMARY.csv', index=False)
print("\n✅ FINAL_PERFORMANCE_SUMMARY.csv")

# 상세 성능
detail_data = {
    'Metric': [
        'Overall Accuracy',
        'Live Accuracy',
        'Spoof Accuracy',
        'FAR (False Accept Rate)',
        'FRR (False Reject Rate)',
        'HTER',
        'Failed Cases',
    ],
    'Replay-Attack': [
        '99.58% (478/480)',
        '100.00% (80/80)',
        '99.50% (398/400)',
        '0.00%',
        '0.50%',
        '0.25%',
        '2 (Client 011, 104)',
    ],
    'MSU-MFSD': [
        '99.29% (278/280)',
        '100.00% (70/70)',
        '99.05% (208/210)',
        '0.00%',
        '0.95%',
        '0.48%',
        '2',
    ]
}

df_detail = pd.DataFrame(detail_data)
df_detail.to_csv('FINAL_DETAILED_METRICS.csv', index=False)
print("✅ FINAL_DETAILED_METRICS.csv")

# 시각화
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. 모델별 성능 비교 (Replay-Attack)
ax1 = fig.add_subplot(gs[0, :])
models = df['Model'].tolist()
replay_acc = df['Replay-Attack (%)'].tolist()
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

bars = ax1.barh(models, replay_acc, color=colors)
ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Model Performance on Replay-Attack Dataset', fontsize=14, fontweight='bold')
ax1.set_xlim(85, 100)
ax1.grid(axis='x', alpha=0.3)

for bar, val in zip(bars, replay_acc):
    ax1.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
             f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

# 2. Cross-dataset 비교
ax2 = fig.add_subplot(gs[1, 0])
methods = ['Late Fusion', 'Disagreement']
replay = [98.96, 99.58]
msu = [96.79, 99.29]

x = np.arange(len(methods))
width = 0.35

bars1 = ax2.bar(x - width/2, replay, width, label='Replay-Attack', color='#3498db')
bars2 = ax2.bar(x + width/2, msu, width, label='MSU-MFSD', color='#2ecc71')

ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Cross-Dataset Generalization', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.legend()
ax2.set_ylim(95, 100)
ax2.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Fusion 진화
ax3 = fig.add_subplot(gs[1, 1])
evolution = ['Texture\nCNN', 'Late\nFusion', 'Disagreement\nFusion']
evolution_acc = [98.57, 98.96, 99.58]
evolution_colors = ['#3498db', '#9b59b6', '#1abc9c']

bars = ax3.bar(evolution, evolution_acc, color=evolution_colors, width=0.6)
ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('System Evolution', fontsize=13, fontweight='bold')
ax3.set_ylim(98, 100)
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, evolution_acc):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.05,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 화살표 추가
ax3.annotate('', xy=(1, 98.96), xytext=(0, 98.57),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax3.annotate('', xy=(2, 99.58), xytext=(1, 98.96),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

plt.suptitle('Face Anti-Spoofing System - Final Performance Summary', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('FINAL_PERFORMANCE_VISUALIZATION.png', dpi=200, bbox_inches='tight', facecolor='white')
print("✅ FINAL_PERFORMANCE_VISUALIZATION.png")

# Architecture 요약
arch_data = {
    'Component': [
        'Expert 1: Texture CNN',
        'Expert 2: Multi-scale LBP',
        'Expert 3: Texture2 CNN (Face-only)',
        'Fusion Strategy',
        'Threshold',
    ],
    'Description': [
        'EfficientNet-B0, RGB input, 98.57%',
        'R=1,2,3 + Random Forest, 97.08%',
        'EfficientNet-B0, Face crop, 95.21%',
        'Disagreement-based 3-expert voting',
        '0.2 probability difference',
    ],
    'Role': [
        'Primary RGB pattern classifier',
        'Texture pattern signal expert',
        'Disagreement backup for face crops',
        'Intelligent ensemble coordinator',
        'Trigger for Texture2 activation',
    ]
}

df_arch = pd.DataFrame(arch_data)
df_arch.to_csv('FINAL_ARCHITECTURE.csv', index=False)
print("✅ FINAL_ARCHITECTURE.csv")

print("\n" + "="*70)
print("생성된 최종 파일:")
print("="*70)
print("1. FINAL_PERFORMANCE_SUMMARY.csv - 전체 모델 성능")
print("2. FINAL_DETAILED_METRICS.csv - 상세 지표")
print("3. FINAL_PERFORMANCE_VISUALIZATION.png - 시각화")
print("4. FINAL_ARCHITECTURE.csv - 시스템 구조")
print("="*70)
