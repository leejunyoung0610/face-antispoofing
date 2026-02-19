# Face Anti-Spoofing (FAS) — MSU→Replay Lightweight Pipeline

MSU-MFSD로 학습한 뒤 Replay-Attack에서 평가/파인튜닝하며,
**Texture/Frequency 전문가 + 조건부 호출(Confidence-Adaptive)**로
성능과 비용(추가 모델 호출)을 함께 최적화하는 가벼운 FAS 파이프라인입니다.

> 핵심 컨셉: **Texture가 기본 성능을 담당**, Frequency는 항상 쓰지 않고  
> **"불확실한 샘플(Tail risk)"에서만 제한적으로 호출**하여 Spoof 탐지를 보강합니다.

---

## Highlights (Replay-Attack, Fine-tuned)

- **Texture CNN**: Overall **98.54%**, FAR **1.25%**, FRR **1.50%**, HTER **1.38%** (Test 480개)
  - Train 360 / Test 480 완전 분리
  - Video-level: Frame과 동일 (97.88%)
- **Adaptive (누수 제거)**: Overall **98.75%** (Test 240개, 1회 평가), FAR/FRR/HTER 안정적
  - Train 360 / Dev 240 / Test 240 분할
  - Threshold **0.55** (DEV only)
  - Video-level: **98.75%**
  - Frequency 호출: **1.25%**

### Component Performance
- **Frequency CNN**: Overall **95.21%**, FAR **16.25%**, FRR **2.50%**
- **Baseline (ResNet18)**: Overall **91.25%** (cross-dataset, no FT)

### Generalization
- **Domain Gap** (MSU→Replay, no fine-tuning):
  - Baseline: **-28.08%** (94.75%→66.67%)
  - Texture: **-10.50%** (99.88%→89.38%)
  - → Texture shows **17.58%p smaller degradation**

### Key Findings
- Tail risk: Client 011(FRR 20%), 104(FAR 25%)
- Texture FN avg confidence: **79.56%** vs TP: **98.23%** (18.67%p gap)
- Frequency vulnerable to simple backgrounds (16 Live false positives)

> ⚠️ 수치는 `create_final_tables.py` / `plot_roc_curves.py` 결과 기준.

---

## Key Contributions
1. **Cross-dataset Generalization**: Texture domain gap -10% vs Baseline -28%
2. **Leak-Free Evaluation**: Train/Dev/Test split, Dev-only threshold tuning
3. **Tail Risk Analysis**: Client 011(FRR 20%), 104(FAR 25%)
4. **Video-level Validation**: Frame≈Video consistency
5. **Confidence-Adaptive Framework**: Texture first + conditional Frequency calls

---

## Data Splits (Leakage-Free)

- **train**: 360 videos (fine-tuning)
- **dev**: 240 videos (threshold selection only)
- **test**: 240 videos (final evaluation, exactly once)
- **test_full**: 480 videos (reference, 원본 test set)

- **Threshold Selection**
  - Grid search on **dev** only
  - **test**는 딱 한 번만 평가 (threshold 미사용)
  - 정보 누수 없이 Texture/Adaptive 평가 분리

## Figures
- `roc_curves.png` — Baseline/Texture/Frequency ROC comparison
- `confusion_matrices.png` — 4-system confusion matrices
- `texture_confidence_distribution.png` — TP vs FN confidence gap
- `frequency_failure_comparison.png` — Live FP visual analysis

---

## Environment
- Python 3.9+
- macOS / Linux (Windows 미검증)

### Install
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation

### 1. Download Datasets
- **MSU-MFSD**: [Download link](http://biometrics.cse.msu.edu/Publications/Databases/MSU_MFSD/)
- **Replay-Attack**: [Download link](https://www.idiap.ch/en/scientific-research/data/replayattack)

### 2. Extract to `data/` folder
```bash
data/
├── MSU-MFSD/
│   ├── scene01/
│   │   ├── real/
│   │   └── attack/
│   └── ...
└── replay-attack/
    └── datasets/fas_pure_data/Idiap-replayattack/
        ├── train/
        └── test/
```

### 3. (Optional) Precompute Feature Cache
```bash
python utils/precompute_features.py
```
This creates `data/feature_cache.pt` (2.3MB) for faster multi-expert training.
```

---

## 🚀 최종 .gitignore
```
# Data (exclude large datasets)
data/MSU-MFSD/
data/replay-attack/
data/*.pt

# Checkpoints
checkpoints/
*.pth

# Results (auto-generated)
results/
*.png
*.jpg

# Python
venv/
__pycache__/
*.pyc
.DS_Store
*.log

# Jupyter
.ipynb_checkpoints/
*.ipynb

# macOS
.DS_Store
.AppleDouble
.LSOverride