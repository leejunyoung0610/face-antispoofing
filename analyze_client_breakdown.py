import os
import torch
from models.texture_expert import TextureExpert
from utils.dataset import ReplayAttackDataset, raw_transforms
from torch.utils.data import DataLoader
from collections import defaultdict

device = torch.device('mps')
model = TextureExpert().to(device)
model.load_state_dict(torch.load('checkpoints/cross/texture.pth', map_location=device))
model.eval()

ds = ReplayAttackDataset('data/replay-attack/datasets/fas_pure_data/Idiap-replayattack', 'test_full')
loader = DataLoader(ds, batch_size=1)

client_results = defaultdict(lambda: {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0})

with torch.no_grad():
    for batch in loader:
        raw = batch['raw'].to(device)
        label = batch['label'].item()
        meta = batch['metadata']
        if isinstance(meta, list):
            path_entry = meta[0]['video_path']
        else:
            path_entry = meta['video_path']
        path = path_entry[0] if isinstance(path_entry, list) else path_entry
        
        filename = os.path.basename(path)
        if filename.startswith('client'):
            client = filename.split('_')[0].replace('client', '')
        elif 'client' in filename:
            client = filename.split('client')[1].split('_')[0]
        else:
            client = 'unknown'
        
        out = model(raw)
        pred = out.argmax(1).item()
        
        if pred == 1 and label == 1:
            client_results[client]['tp'] += 1
        elif pred == 0 and label == 0:
            client_results[client]['tn'] += 1
        elif pred == 1 and label == 0:
            client_results[client]['fp'] += 1
        else:
            client_results[client]['fn'] += 1

print('=== Client별 성능 ===')
print(f"{'Client':<10} {'Live':>7} {'Spoof':>7} {'FAR':>7} {'FRR':>7}")

for client in sorted(client_results.keys()):
    r = client_results[client]
    live_acc = r['tn'] / (r['tn'] + r['fp']) if (r['tn'] + r['fp']) > 0 else 0
    spoof_acc = r['tp'] / (r['tp'] + r['fn']) if (r['tp'] + r['fn']) > 0 else 0
    far = r['fp'] / (r['fp'] + r['tn']) if (r['fp'] + r['tn']) > 0 else 0
    frr = r['fn'] / (r['fn'] + r['tp']) if (r['fn'] + r['tp']) > 0 else 0
    
    print(f"{client:<10} {live_acc*100:6.2f}% {spoof_acc*100:6.2f}% {far*100:6.2f}% {frr*100:6.2f}%")

print()
print('=== 문제 Client (011, 104) ===')
for client in ['011', '104']:
    if client in client_results:
        r = client_results[client]
        print(f"Client {client}: FN={r['fn']} (놓친 Spoof)")
