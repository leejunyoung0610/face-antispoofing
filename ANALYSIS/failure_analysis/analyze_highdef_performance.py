import torch
import torch.nn.functional as F
from utils.dataset import ReplayAttackDataset
from models.texture_expert import TextureExpert
from models.texture2_expert import Texture2Expert
import pickle
import numpy as np
from skimage.feature import local_binary_pattern
import cv2
from collections import defaultdict

device = torch.device('mps')

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
    for R, P, bins in [(1, 8, 59), (2, 16, 243), (3, 24, 299)]:
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
        features.extend(hist)
    return np.array(features)

test_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'test_full'
)

print("=== Highdef Print 전체 성능 분석 ===\n")

client_stats = defaultdict(lambda: {
    'texture': {'correct': 0, 'total': 0},
    'freq': {'correct': 0, 'total': 0},
    'smart': {'correct': 0, 'total': 0}
})

all_highdef_texture = {'correct': 0, 'total': 0}
all_highdef_freq = {'correct': 0, 'total': 0}
all_highdef_smart = {'correct': 0, 'total': 0}

for i in range(len(test_ds)):
    item = test_ds[i]
    path = item['metadata']['video_path'].lower()
    
    if item['label'] == 1 and 'highdef' in path and 'print' in path:
        # Client 번호 추출
        client_id = None
        for j in range(1, 100):
            client_str = f'client{j:03d}'
            if client_str in path:
                client_id = client_str
                break
        
        if not client_id:
            continue
        
        x = item['raw'].unsqueeze(0).to(device)
        with torch.no_grad():
            t_out = texture(x)
            t_pred = t_out.argmax(1).item()
            t_prob = F.softmax(t_out, dim=1)[0][1].item()
        
        # Frequency
        x_freq = item['freq_raw'].unsqueeze(0).to(device)
        with torch.no_grad():
            f_out = freq(x_freq)
            f_pred = f_out.argmax(1).item()
        
        # LBP
        image = item['raw'].permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        lbp_feat = extract_multiscale_lbp(image)
        l_prob = lbp_rf.predict_proba([lbp_feat])[0][1]
        
        # Smart Fusion
        two_way = (t_prob + l_prob) / 2
        if abs(two_way - 0.5) < 0.2:
            x_freq = item['freq_raw'].unsqueeze(0).to(device)
            with torch.no_grad():
                f_out = freq(x_freq)
                f_prob = F.softmax(f_out, dim=1)[0][1].item()
            smart_score = (t_prob + l_prob + f_prob) / 3
        else:
            smart_score = two_way
        
        smart_pred = 1 if smart_score > 0.5 else 0
        
        # 집계
        client_stats[client_id]['texture']['total'] += 1
        client_stats[client_id]['freq']['total'] += 1
        client_stats[client_id]['smart']['total'] += 1
        
        if t_pred == 1:
            client_stats[client_id]['texture']['correct'] += 1
            all_highdef_texture['correct'] += 1
        
        if f_pred == 1:
            client_stats[client_id]['freq']['correct'] += 1
            all_highdef_freq['correct'] += 1
        
        if smart_pred == 1:
            client_stats[client_id]['smart']['correct'] += 1
            all_highdef_smart['correct'] += 1
        
        all_highdef_texture['total'] += 1
        all_highdef_freq['total'] += 1
        all_highdef_smart['total'] += 1

# 출력
print(f"총 Highdef Print: {all_highdef_texture['total']}개\n")

print("=== Client별 성능 ===")
print(f"{'Client':<12} {'Texture':<12} {'Frequency':<12} {'Smart':<12}")
print("-" * 50)

for client_id in client_stats.keys():
    s = client_stats[client_id]
    t_acc = s['texture']['correct'] / s['texture']['total'] * 100
    f_acc = s['freq']['correct'] / s['freq']['total'] * 100
    sm_acc = s['smart']['correct'] / s['smart']['total'] * 100
    
    print(f"{client_id:<12} {t_acc:>5.1f}% ({s['texture']['correct']}/{s['texture']['total']})  "
          f"{f_acc:>5.1f}% ({s['freq']['correct']}/{s['freq']['total']})  "
          f"{sm_acc:>5.1f}% ({s['smart']['correct']}/{s['smart']['total']})")

print("-" * 50)
print(f"{'Overall':<12} "
      f"{all_highdef_texture['correct']/all_highdef_texture['total']*100:>5.1f}% ({all_highdef_texture['correct']}/{all_highdef_texture['total']})  "
      f"{all_highdef_freq['correct']/all_highdef_freq['total']*100:>5.1f}% ({all_highdef_freq['correct']}/{all_highdef_freq['total']})  "
      f"{all_highdef_smart['correct']/all_highdef_smart['total']*100:>5.1f}% ({all_highdef_smart['correct']}/{all_highdef_smart['total']})")

# Client 011 특별 언급
if 'client011' in client_stats:
    print(f"\n⚠️ Client 011:")
    s = client_stats['client011']
    print(f"  Texture:   {s['texture']['correct']}/{s['texture']['total']} = {s['texture']['correct']/s['texture']['total']*100:.1f}%")
    print(f"  Frequency: {s['freq']['correct']}/{s['freq']['total']} = {s['freq']['correct']/s['freq']['total']*100:.1f}%")
    print(f"  Smart:     {s['smart']['correct']}/{s['smart']['total']} = {s['smart']['correct']/s['smart']['total']*100:.1f}%")
