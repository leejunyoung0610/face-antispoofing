import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import numpy as np
import cv2
from utils.dataset import ReplayAttackDataset
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

device = torch.device('mps')

# FFT 전처리
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

# FFT Dataset
class FFTDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_ds = base_dataset
    
    def __len__(self):
        return len(self.base_ds)
    
    def __getitem__(self, idx):
        item = self.base_ds[idx]
        
        image_np = item['raw'].permute(1, 2, 0).numpy()
        image_np = (image_np * 255).astype(np.uint8)
        fft_img = image_to_fft(image_np)
        
        return {
            'fft': torch.FloatTensor(fft_img),
            'label': item['label']
        }

# 모델
class TrueTexture2Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = efficientnet_b0(pretrained=True)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Dataset 로드
print("=== True Frequency Expert 완전 학습 ===\n")
print("Dataset 로드 중...")

base_train = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'train'
)

base_test = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'test_full'
)
print(f"Train: {len(base_train)} samples")
print(f"Test:  {len(base_test)} samples")

# FFT Dataset
print("\nFFT Dataset 생성 중...")
train_fft = FFTDataset(base_train)
test_fft = FFTDataset(base_test)

# DataLoader
train_loader = DataLoader(train_fft, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_fft, batch_size=16, shuffle=False, num_workers=0)

# 모델, 손실, 옵티마이저
model = TrueTexture2Expert().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# 학습
print("\n학습 시작...")
num_epochs = 30
best_acc = 0

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch in train_loader:
        x = batch['fft'].to(device)
        y = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        pred = outputs.argmax(1)
        train_correct += (pred == y).sum().item()
        train_total += y.size(0)
        train_loss += loss.item()
    
    train_acc = train_correct / train_total * 100
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x = batch['fft'].to(device)
            y = batch['label'].to(device)
            
            outputs = model(x)
            pred = outputs.argmax(1)
            test_correct += (pred == y).sum().item()
            test_total += y.size(0)
    
    test_acc = test_correct / test_total * 100
    
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss={train_loss/len(train_loader):.4f}, "
          f"Train Acc={train_acc:.2f}%, "
          f"Test Acc={test_acc:.2f}%")
    
    # Best 저장
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'true_texture2_expert_best.pth')
        print(f"  → Best model saved! (Test Acc: {test_acc:.2f}%)")

print(f"\n {best_acc:.2f}%")

# 최종 상세 평가
print("\n=== 상세 평가 ===")

model.load_state_dict(torch.load('true_texture2_expert_best.pth'))
model.eval()

overall_correct = 0
overall_total = 0
live_correct = 0
live_total = 0
spoof_correct = 0
spoof_total = 0

client011_correct = 0
client011_total = 0

highdef_correct = 0
highdef_total = 0

for i in range(len(base_test)):
    item = base_test[i]
    path = item['metadata']['video_path'].lower()
    label = item['label']
    
    # FFT 변환
    image_np = item['raw'].permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)
    fft_img = image_to_fft(image_np)
    
    x = torch.FloatTensor(fft_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(x)
        pred = outputs.argmax(1).item()
    
    # 전체
    overall_total += 1
    if pred == label:
        overall_correct += 1
    
    # Live vs Spoof
    if label == 0:
        live_total += 1
        if pred == label:
            live_correct += 1
    else:
        spoof_total += 1
        if pred == label:
            spoof_correct += 1
    
    # Client 011 Highdef
    if 'client011' in path and 'highdef' in path and label == 1:
        client011_total += 1
        if pred == label:
            client011_correct += 1
    
    # All Highdef
    if 'highdef' in path and 'print' in path and label == 1:
        highdef_total += 1
        if pred == label:
            highdef_correct += 1

print(f"\nOverall:           {overall_correct}/{overall_total} = {overall_correct/overall_total*100:.2f}%")
print(f"Live:              {live_correct}/{live_total} = {live_correct/live_total*100:.2f}%")
print(f"Spoof:             {spoof_correct}/{spoof_total} = {spoof_correct/spoof_total*100:.2f}%")
print(f"Client 011 Highdef: {client011_correct}/{client011_total} = {client011_correct/client011_total*100:.1f}%")
print(f"전체 Highdef:       {highdef_correct}/{highdef_total} = {highdef_correct/highdef_total*100:.1f}%")

print("\n✅ 완료!")
print("저장된 모델: true_texture2_expert_best.pth")
