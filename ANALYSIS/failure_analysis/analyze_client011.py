import torch
import torch.nn.functional as F
from utils.dataset import ReplayAttackDataset
from models.texture2_expert import Texture2Expert
from models.texture_expert import TextureExpert
import pickle
import numpy as np
from skimage.feature import local_binary_pattern
import cv2

device = torch.device('mps')

# 모델 로드
texture = TextureExpert().to(device)
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))
texture.eval()

freq = Texture2Expert().to(device)
freq.load_state_dict(torch.load('checkpoints/cross/texture2.pth'))
freq.eval()

with open('lbp_rf_model.pkl', 'rb') as f:
    lbp_rf = pickle.load(f)

def extract_multiscale_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    lbp1 = local_binary_pattern(gray, 8, 1, method='uniform')
    hist1, _ = np.histogram(lbp1, bins=59, range=(0, 59), density=True)
    features.extend(hist1)
    lbp2 = local_binary_pattern(gray, 16, 2, method='uniform')
    hist2, _ = np.histogram(lbp2, bins=243, range=(0, 243), density=True)
    features.extend(hist2)
    lbp3 = local_binary_pattern(gray, 24, 3, method='uniform')
    hist3, _ = np.histogram(lbp3, bins=299, range=(0, 299), density=True)
    features.extend(hist3)
    return np.array(features)

test_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'test_full'
)

client_011_highdef = []
for i in range(len(test_ds)):
    item = test_ds[i]
    video_path = item['metadata']['video_path']
    if 'client011' in video_path.lower() and 'highdef' in video_path.lower() and item['label'] == 1:
        client_011_highdef.append(i)

print(f"Client 011 Highdef: {len(client_011_highdef)}개\n")

texture_correct = 0
freq_correct = 0
fusion_correct = 0

for idx in client_011_highdef:
    item = test_ds[idx]
    
    x = item['raw'].unsqueeze(0).to(device)
    with torch.no_grad():
        t_out = texture(x)
        t_pred = t_out.argmax(1).item()
        t_prob = F.softmax(t_out, dim=1)[0][1].item()
    
    if t_pred == item['label']:
        texture_correct += 1
    
    x_freq = item['freq_raw'].unsqueeze(0).to(device)
    with torch.no_grad():
        f_out = freq(x_freq)
        f_pred = f_out.argmax(1).item()
    
    if f_pred == item['label']:
        freq_correct += 1
    
    image = item['raw'].permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    lbp_feat = extract_multiscale_lbp(image)
    l_prob = lbp_rf.predict_proba([lbp_feat])[0][1]
    
    fused = (t_prob + l_prob) / 2
    fusion_pred = 1 if fused > 0.5 else 0
    
    if fusion_pred == item['label']:
        fusion_correct += 1

total = len(client_011_highdef)
print("=== Client 011 Highdef 성능 ===")
print(f"Texture CNN:    {texture_correct}/{total} = {texture_correct/total*100:.1f}%")
print(f"Frequency CNN:  {freq_correct}/{total} = {freq_correct/total*100:.1f}%")
print(f"Late Fusion:    {fusion_correct}/{total} = {fusion_correct/total*100:.1f}%")
print(f"\n개선: Texture {texture_correct}개 -> Fusion {fusion_correct}개 (+{fusion_correct - texture_correct}개)")
