import torch
import numpy as np
import torch.nn.functional as F
from utils.dataset import ReplayAttackDataset
from models.texture_expert import TextureExpert
from models.texture2_expert import Texture2Expert
import matplotlib.pyplot as plt

device = torch.device('mps')

print("="*60)
print("Texture vs Texture2 상보성 검증")
print("="*60)

# 모델 로드
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval()

texture2 = Texture2Expert().to(device)
texture2.load_state_dict(torch.load('checkpoints/cross/texture2.pth'))
texture2.eval()

# 데이터
test_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'test_full'
)

print(f"\nTest: {len(test_ds)}\n")

# 평가
t1_preds = []
t2_preds = []
labels = []

for i in range(len(test_ds)):
    item = test_ds[i]
    
    # Texture (전체)
    x1 = item['raw'].unsqueeze(0).to(device)
    with torch.no_grad():
        out1 = texture(x1)
        pred1 = out1.argmax(1).item()
    
    # Texture2 (얼굴)
    x2 = item['freq_raw'].unsqueeze(0).to(device)
    with torch.no_grad():
        out2 = texture2(x2)
        pred2 = out2.argmax(1).item()
    
    t1_preds.append(pred1)
    t2_preds.append(pred2)
    labels.append(item['label'])

t1_preds = np.array(t1_preds)
t2_preds = np.array(t2_preds)
labels = np.array(labels)

# 분석
t1_correct = (t1_preds == labels)
t2_correct = (t2_preds == labels)

t1_only = t1_correct & (~t2_correct)
t2_only = (~t1_correct) & t2_correct
both_correct = t1_correct & t2_correct
both_wrong = (~t1_correct) & (~t2_correct)

print("결과:")
print(f"Texture만 맞음:     {t1_only.sum()}개")
print(f"Texture2만 맞음:    {t2_only.sum()}개")
print(f"둘 다 맞음:         {both_correct.sum()}개")
print(f"둘 다 틀림:         {both_wrong.sum()}개")

print(f"\n상보성 지수: {t1_only.sum() + t2_only.sum()}개")
print(f"→ Voting하면 개선 가능한 케이스")

# Disagreement 전략
agreement = (t1_preds == t2_preds)
print(f"\n불일치 개수: {(~agreement).sum()}개 ({(~agreement).sum()/len(labels)*100:.1f}%)")

# 불일치 케이스에서 어느 쪽이 더 맞나?
disagree_idx = np.where(~agreement)[0]
t1_right_in_disagree = 0
t2_right_in_disagree = 0

for idx in disagree_idx:
    if t1_preds[idx] == labels[idx]:
        t1_right_in_disagree += 1
    if t2_preds[idx] == labels[idx]:
        t2_right_in_disagree += 1

print(f"\n불일치 시:")
print(f"  Texture 정답:  {t1_right_in_disagree}개")
print(f"  Texture2 정답: {t2_right_in_disagree}개")

if t1_right_in_disagree > t2_right_in_disagree:
    print("  → Texture가 더 신뢰성 높음")
elif t2_right_in_disagree > t1_right_in_disagree:
    print("  → Texture2가 더 신뢰성 높음")
else:
    print("  → 비슷함")

print("="*60)
