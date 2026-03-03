from skimage.feature import local_binary_pattern
import numpy as np
import cv2
from utils.dataset import ReplayAttackDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def extract_multiscale_lbp(image):
    """Extract multi-scale LBP features"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    features = []
    
    # Scale 1: R=1, P=8 (local fine texture)
    lbp1 = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist1, _ = np.histogram(lbp1, bins=59, range=(0, 59), density=True)
    features.extend(hist1)
    
    # Scale 2: R=2, P=16 (medium texture)
    lbp2 = local_binary_pattern(gray, P=16, R=2, method='uniform')
    hist2, _ = np.histogram(lbp2, bins=243, range=(0, 243), density=True)
    features.extend(hist2)
    
    # Scale 3: R=3, P=24 (coarse texture)
    lbp3 = local_binary_pattern(gray, P=24, R=3, method='uniform')
    hist3, _ = np.histogram(lbp3, bins=299, range=(0, 299), density=True)
    features.extend(hist3)
    
    return np.array(features)

# Train
print('Loading train data...')
train_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'train'
)

X_train = []
y_train = []

for i in range(len(train_ds)):
    item = train_ds[i]
    image = item['raw'].permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    
    features = extract_multiscale_lbp(image)
    X_train.append(features)
    y_train.append(item['label'])
    
    if (i + 1) % 50 == 0:
        print(f'Processed {i+1}/{len(train_ds)}')

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f'\nTrain: {X_train.shape}')
print(f'Feature dims: 59 + 243 + 299 = {X_train.shape[1]}')

# Test
print('\nLoading test data...')
test_ds = ReplayAttackDataset(
    'data/replay-attack/datasets/fas_pure_data/Idiap-replayattack',
    'test_full'
)

X_test = []
y_test = []

for i in range(len(test_ds)):
    item = test_ds[i]
    image = item['raw'].permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    
    features = extract_multiscale_lbp(image)
    X_test.append(features)
    y_test.append(item['label'])
    
    if (i + 1) % 50 == 0:
        print(f'Processed {i+1}/{len(test_ds)}')

X_test = np.array(X_test)
y_test = np.array(y_test)

print(f'\nTest: {X_test.shape}')

# Train
print('\n=== Training Multi-scale LBP + RF ===')
rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

pred = rf.predict(X_test)
acc = accuracy_score(y_test, pred)

cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()
live_acc = tn / (tn + fp)
spoof_acc = tp / (tp + fn)
far = fp / (fp + tn)
frr = fn / (fn + tp)

print(f'\nAccuracy: {acc:.4f} ({acc*100:.2f}%)')
print(f'Live:     {live_acc:.4f} ({live_acc*100:.2f}%)')
print(f'Spoof:    {spoof_acc:.4f} ({spoof_acc*100:.2f}%)')
print(f'FAR:      {far:.4f} ({far*100:.2f}%)')
print(f'FRR:      {frr:.4f} ({frr*100:.2f}%)')

print('\n=== Comparison ===')
print(f'Texture CNN:          98.54%')
print(f'Single-scale LBP+RF:  95.83%')
print(f'Multi-scale LBP+RF:   {acc*100:.2f}%')
print(f'\nImprovement: {(acc - 0.9583)*100:.2f}%p')
print(f'Gap to CNN:  {98.54 - acc*100:.2f}%p')

# Feature importance
print('\n=== Feature Importance (top 10) ===')
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]

for i, idx in enumerate(top_idx, 1):
    if idx < 59:
        scale = f"Scale 1 (R=1, bin {idx})"
    elif idx < 59 + 243:
        scale = f"Scale 2 (R=2, bin {idx-59})"
    else:
        scale = f"Scale 3 (R=3, bin {idx-302})"
    
    print(f'{i}. Feature {idx}: {importances[idx]:.4f} ({scale})')
