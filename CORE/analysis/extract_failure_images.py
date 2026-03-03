import torch
import torch.nn.functional as F
from utils.dataset import ReplayAttackDataset
from models.texture_expert import TextureExpert
import pickle
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

device = torch.device('mps')

def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    for R, P, bins in [(1, 8, 59), (2, 16, 243), (3, 24, 299)]:
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        features.extend(hist)
    return np.array(features)

print("="*70)
print("실패 케이스 이미지 추출")
print("="*70)

# 모델 로드
print("\n모델 로드...")
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval()

with open('lbp_rf_model.pkl', 'rb') as f:
    lbp_rf = pickle.load(f)
print("✅ 완료\n")

# 데이터
test_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'test_full'
)

# Late Fusion 실패 케이스 찾기
print("Late Fusion 실패 케이스 찾기...\n")

failures = []

for i in range(len(test_ds)):
    item = test_ds[i]
    
    # Texture
    x = item['raw'].unsqueeze(0).to(device)
    with torch.no_grad():
        t_prob = F.softmax(texture(x), dim=1)[0][1].item()
    
    # LBP
    image_np = item['raw'].permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    l_prob = lbp_rf.predict_proba([extract_lbp(image_np)])[0][1]
    
    # Late Fusion
    avg = (t_prob + l_prob) / 2
    pred = 1 if avg > 0.5 else 0
    
    if pred != item['label']:
        path = str(test_ds.samples[i])
        
        # Client 정보
        client = 'unknown'
        for part in path.split('/'):
            if 'client' in part.lower():
                client = part
                break
        
        failures.append({
            'index': i,
            'path': path,
            'client': client,
            'label': item['label'],
            'pred': pred,
            'texture_prob': t_prob,
            'lbp_prob': l_prob,
            'fusion_prob': avg,
            'image': image_np
        })
        
        print(f"실패 {len(failures)}: Index {i}")
        print(f"  Client: {client}")
        print(f"  Label: {'Spoof' if item['label'] == 1 else 'Live'}")
        print(f"  Pred: {'Spoof' if pred == 1 else 'Live'}")
        print(f"  T: {t_prob:.3f}, L: {l_prob:.3f}, Fusion: {avg:.3f}\n")

print(f"총 {len(failures)}개 실패 케이스\n")

# Client별 분류
client_011 = [f for f in failures if '011' in f['client']]
client_104 = [f for f in failures if '104' in f['client']]

print(f"Client 011: {len(client_011)}개")
print(f"Client 104: {len(client_104)}개")

# 시각화 1: Client 011 실패 케이스
if len(client_011) > 0:
    n_samples = min(3, len(client_011))
    fig = plt.figure(figsize=(16, 5*n_samples))
    gs = GridSpec(n_samples, 3, figure=fig, hspace=0.3, wspace=0.3)
    for idx, failure in enumerate(client_011[:n_samples]):
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(failure['image'])
        ax1.set_title(f"Client 011 실패 #{idx+1}\n{'Spoof' if failure['label'] == 1 else 'Live'} → Predicted: {'Spoof' if failure['pred'] == 1 else 'Live'}", 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 확률 바
        ax2 = fig.add_subplot(gs[idx, 1])
        methods = ['Texture', 'LBP', 'Fusion']
        probs = [failure['texture_prob'], failure['lbp_prob'], failure['fusion_prob']]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        
        bars = ax2.barh(methods, probs, color=colors)
        ax2.set_xlim(0, 1)
        ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_xlabel('Spoof Probability', fontsize=11, fontweight='bold')
        ax2.set_title('Model Predictions', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for bar, prob in zip(bars, probs):
            ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', va='center', fontsize=10, fontweight='bold')
        
        # 정보 텍스트
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.axis('off')
        
        info_text = f"""
Index: {failure['index']}

Ground Truth: {'Spoof' if failure['label'] == 1 else 'Live'}
Prediction: {'Spoof' if failure['pred'] == 1 else 'Live'}

Texture: {failure['texture_prob']:.4f}
LBP: {failure['lbp_prob']:.4f}
Fusion: {failure['fusion_prob']:.4f}

Path:
{failure['path'].split('/')[-1]}

Analysis:
{"Both models wrong" if (failure['texture_prob'] > 0.5) == (failure['lbp_prob'] > 0.5) else "Models disagree"}
        """
        
        ax3.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Client 011 실패 케이스', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('client_011_failures.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✅ 저장: client_011_failures.png")

# 시각화 2: Client 104 실패 케이스
if len(client_104) > 0:
    n_samples = min(2, len(client_104))
    fig = plt.figure(figsize=(16, 5*n_samples))
    gs = GridSpec(n_samples, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, failure in enumerate(client_104[:n_samples]):
        # 원본 이미지
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(failure['image'])
        ax1.set_title(f"Client 104 실패 #{idx+1}\n{'Spoof' if failure['label'] == 1 else 'Live'} → Predicted: {'Spoof' if failure['pred'] == 1 else 'Live'}", 
                     fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 확률 바
        ax2 = fig.add_subplot(gs[idx, 1])
        methods = ['Texture', 'LBP', 'Fusion']
        probs = [failure['texture_prob'], failure['lbp_prob'], failure['fusion_prob']]
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax2.barh(methods, probs, color=colors)
        ax2.set_xlim(0, 1)
        ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_xlabel('Spoof Probability', fontsize=11, fontweight='bold')
        ax2.set_title('Model Predictions', fontsize=12, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        for bar, prob in zip(bars, probs):
            ax2.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{prob:.3f}', va='center', fontsize=10, fontweight='bold')
        
        # 정보 텍스트
        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.axis('off')
        
        analysis = ("Both models wrong"
                    if (failure['texture_prob'] > 0.5) == (failure['lbp_prob'] > 0.5)
                    else "Models disagree")
        info_text = (
            f"Index: {failure['index']}\n"
            f"Ground Truth: {'Spoof' if failure['label'] == 1 else 'Live'}\n"
            f"Prediction: {'Spoof' if failure['pred'] == 1 else 'Live'}\n\n"
            f"Texture: {failure['texture_prob']:.4f}\n"
            f"LBP: {failure['lbp_prob']:.4f}\n"
            f"Fusion: {failure['fusion_prob']:.4f}\n\n"
            f"Path: {failure['path'].split('/')[-1]}\n\n"
            f"Analysis: {analysis}"
        )
        
        ax3.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Client 104 실패 케이스', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('client_104_failures.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 저장: client_104_failures.png")

# 모든 실패 케이스 한 번에
if len(failures) > 0:
    n_cols = 5
    n_rows = (len(failures) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, failure in enumerate(failures):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        ax.imshow(failure['image'])
        
        client_num = '011' if '011' in failure['client'] else '104'
        label_text = 'Spoof' if failure['label'] == 1 else 'Live'
        pred_text = 'Spoof' if failure['pred'] == 1 else 'Live'
        
        title = f"Client {client_num}\n{label_text}→{pred_text}\nF:{failure['fusion_prob']:.2f}"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # 빈 subplot 숨기기
    for idx in range(len(failures), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'모든 실패 케이스 ({len(failures)}개)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('all_failures.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 저장: all_failures.png")

print("\n생성된 이미지:")
print("1. client_011_failures.png - Client 011 상세")
print("2. client_104_failures.png - Client 104 상세")
print("3. all_failures.png - 전체 실패 케이스")
