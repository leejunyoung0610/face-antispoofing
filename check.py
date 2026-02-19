import os
import torch
from utils.dataset import MSUMFSDDataset
from models.texture_expert import TextureExpert
from torch.utils.data import DataLoader

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
texture = TextureExpert().to(device)
state = torch.load('checkpoints/texture_expert/best_model.pth', map_location=device)
texture.load_state_dict(state)
texture.eval()

train_ds = MSUMFSDDataset('data', split='train')
test_ds = MSUMFSDDataset('data', split='test')

train_videos = set([s['video_path'] for s in train_ds.samples])
test_videos = set([s['video_path'] for s in test_ds.samples])
overlap = train_videos & test_videos
print(f'Train 비디오: {len(train_videos)}개')
print(f'Test 비디오: {len(test_videos)}개')
print(f'겹치는 비디오: {len(overlap)}개')
if overlap:
    print('⚠️ Data Leakage 발견!')
    for v in list(overlap)[:3]:
        print(' ', v)
else:
    print('✅ 겹치는 비디오 없음')

train_subjects = set()
test_subjects = set()

for s in train_ds.samples:
    fname = os.path.basename(s['video_path'])
    subject = fname.split('_')[1]
    train_subjects.add(subject)

for s in test_ds.samples:
    fname = os.path.basename(s['video_path'])
    subject = fname.split('_')[1]
    test_subjects.add(subject)

overlap_subjects = train_subjects & test_subjects
print(f'Train subjects: {sorted(train_subjects)}')
print(f'Test subjects: {sorted(test_subjects)}')
print(f'겹치는 subject: {overlap_subjects}')

loader = DataLoader(test_ds, batch_size=200, shuffle=True, num_workers=0)
batch = next(iter(loader))
raw = batch['raw'].to(device)
labels = batch['label'].to(device)

with torch.no_grad():
    out = texture(raw)
    preds = out.argmax(1)
    print('Live  정확도:', (preds[labels==0] == 0).float().mean().item())
    print('Spoof 정확도:', (preds[labels==1] == 1).float().mean().item())
    print('전체 정확도:', (preds == labels).float().mean().item())
    print('Spoof 예측 비율:', (preds == 1).float().mean().item())
