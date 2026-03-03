# Face Anti-Spoofing System

**Disagreement-Based 3-Expert Ensemble for Robust Spoof Detection**

## 🎯 Overview

얼굴 위조 공격(Print, Replay, Mask 등)을 탐지하는 시스템으로, 3개의 전문가 모델을 지능적으로 조합하여 높은 정확도와 일반화 성능을 달성했습니다.

### Key Achievements

- **Replay-Attack**: 99.58% (478/480)
- **MSU-MFSD** (Cross-dataset): 99.29% (278/280)
- **FAR**: 0.00%, **FRR**: 0.50%, **HTER**: 0.25%
- **Disagreement-based Smart Fusion**: 효율적인 3차 검증 (13%만 추가 호출)

## 🏗️ System Architecture

### 3-Expert Ensemble

```
Input Image
     │
     ├─→ Expert 1: Texture CNN (EfficientNet-B0)
     │   98.57% - RGB 패턴 인식
     │   
     ├─→ Expert 2: Multi-scale LBP (Random Forest)
     │   97.08% - 텍스처 패턴 분석 (R=1,2,3)
     │
     └─→ Expert 3: Texture2 CNN (Face-only)
         95.21% - Disagreement 시에만 투입
```

### Disagreement Fusion Strategy

```python
if |P_texture - P_lbp| > 0.2:
    # 불일치 시 3차 검증
    Final = (Texture + LBP + Texture2) / 3
else:
    # 일치 시 2-way Late Fusion
    Final = (Texture + LBP) / 2
```

**장점:**
- 불일치 케이스만 추가 검증 (13% 케이스)
- 계산 비용 최소화
- 정확도 향상 (98.96% → 99.58%)

## 📊 Performance Summary

### In-domain (Replay-Attack)

| Model | Overall | Live | Spoof | FAR | FRR | HTER |
|-------|---------|------|-------|-----|-----|------|
| Texture CNN | 98.57% | 98.75% | 98.50% | 1.25% | 1.50% | 1.38% |
| Multi-LBP | 97.08% | 86.25% | 99.25% | 13.75% | 0.75% | 7.25% |
| Late Fusion | 98.96% | 98.75% | 99.25% | 1.25% | 0.75% | 1.00% |
| **Disagreement** | **99.58%** | **100.00%** | **99.50%** | **0.00%** | **0.50%** | **0.25%** |

### Cross-dataset (MSU-MFSD)

| Method | Replay-Attack | MSU-MFSD | Gap |
|--------|---------------|----------|-----|
| Late Fusion | 98.96% | 96.79% | -2.17%p |
| **Disagreement** | **99.58%** | **99.29%** | **-0.29%p** |

→ 뛰어난 일반화 성능 (도메인 갭 최소화)

## 🔬 Key Findings

### 1. Multi-scale LBP의 중요성
- Single-scale (R=1만): 77%
- Multi-scale (R=1,2,3): 97.08%
- **R=3 (거친 텍스처)가 핵심**: 전역 구조 차이 포착

### 2. Late Fusion vs Early Fusion
- **Early Fusion** (RGB+LBP 6채널): 88.54% ❌
  - LBP의 강한 신호가 CNN 학습 방해
  - Live 32.50% (심각한 편향)
  
- **Late Fusion** (독립 학습 후 결합): 98.96% ✅
  - 각 모델의 편향 없이 독립 최적화
  - 상호보완 극대화

### 3. Tail Risk Analysis
- **Client 011**: Perfect Print Vulnerability
  - 고품질 프린트 + 단순 배경 + 매끄러운 피부
  - FRR 20% (4/5 실패)
- **Client 104**: 특정 조명 조건 취약
  - FAR 25%

→ 평균 성능 ≠ 개인별 성능, Worst-case 관리 필요

## 📁 Project Structure

```
face_antispoofing_3weeks/
│
├── FINAL/                    # ⭐ 최종 결과물
│   ├── README.md            # 상세 문서
│   ├── FINAL_PERFORMANCE_SUMMARY.csv
│   ├── FINAL_DETAILED_METRICS.csv
│   ├── FINAL_PERFORMANCE_VISUALIZATION.png
│   └── complete_fusion_comparison.png
│
├── CORE/                     # 핵심 시스템 코드
│   ├── training/
│   │   ├── train_texture.py
│   │   ├── train_texture2.py
│   │   └── train_multiscale_lbp.py
│   ├── evaluation/
│   │   ├── complete_fusion_test.py
│   │   ├── test_disagreement_strategy.py
│   │   ├── evaluate_msu_disagreement.py
│   │   └── evaluate_msu_final.py
│   ├── models/
│   │   ├── texture_expert.py
│   │   ├── texture2_expert.py
│   │   └── ...
│   ├── utils/
│   │   └── dataset.py
│   ├── checkpoints/
│   │   ├── cross/
│   │   │   ├── texture.pth
│   │   │   └── texture2.pth
│   │   └── lbp_rf_model.pkl
│   └── requirements.txt
│
├── ANALYSIS/                 # 실험 분석
│   ├── ablation/
│   ├── failure_analysis/
│   ├── complementary/
│   └── cross_dataset/
│
└── README.md                 # This file

```

## 🚀 Quick Start

### 1. Environment Setup

```bash
cd face_antispoofing_3weeks
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r CORE/requirements.txt
```

### 2. Data Preparation

```bash
data/
├── replay-attack/
│   └── datasets/fas_pure_data/Idiap-replayattack/
│       ├── train/
│       ├── devel/
│       └── test/
└── msu-mfsd/
    └── scene01/
        ├── real/
        └── attack/
```

### 3. Run Evaluation

```bash
# Replay-Attack (In-domain)
python CORE/evaluation/complete_fusion_test.py

# MSU-MFSD (Cross-dataset)
python CORE/evaluation/evaluate_msu_disagreement.py

# Threshold 탐색
python CORE/evaluation/test_disagreement_strategy.py
```

### 4. Train from Scratch

```bash
# Expert 1: Texture CNN
python CORE/training/train_texture.py

# Expert 2: Multi-scale LBP
python CORE/training/train_multiscale_lbp.py

# Expert 3: Texture2 (Face-only)
python CORE/training/train_texture2.py
```

## 📖 Documentation

**상세 문서는 `FINAL/README.md`를 참고하세요.**

주요 내용:
- 시스템 구조 상세 설명
- 실험 결과 및 분석
- Ablation Study
- 실패 케이스 분석
- Cross-dataset 검증
- 핵심 발견 및 교훈

## 🔑 Key Contributions

1. **Disagreement-based Smart Fusion**
   - 3-expert ensemble with conditional activation
   - 13%만 추가 검증으로 99.58% 달성
   
2. **Multi-scale의 재발견**
   - R=1,2,3 조합이 R=1,2,4보다 우수
   - 거친 텍스처(R=3)의 중요성 입증
   
3. **명시적 vs 암묵적 특징 비교**
   - LBP (명시적): 97.08%
   - CNN (암묵적): 98.57%
   - Gap 1.49%p만 → 해석 가능성 + 고성능 양립
   
4. **Tail Risk Analysis**
   - Client-specific 취약성 발견
   - Perfect Print Vulnerability 정의
   - Worst-case 관리 필요성 제시

5. **Cross-dataset 일반화**
   - Domain gap -0.29%p (Late Fusion 대비 -2.17%p)
   - 실전 배포 가능성 입증

## 🛠️ Technical Stack

- **Framework**: PyTorch 2.0+
- **Models**: EfficientNet-B0, Random Forest
- **Features**: RGB, Multi-scale LBP (R=1,2,3)
- **Platform**: macOS (Apple Silicon), Linux
- **Backend**: MPS (Metal Performance Shaders)

## 📝 Citation

본 프로젝트를 참고하실 경우 아래와 같이 인용해주세요:

```
Face Anti-Spoofing System with Disagreement Fusion
3-Expert Ensemble achieving 99.58% on Replay-Attack
Repository: https://github.com/leejunyoung0610/face-antispoofing
```


## 📄 License

개인 프로젝트 진행입니다

---

## 📞 Contact

이슈 제기 환영합니다.

---

**Face Anti-Spoofing: 99.58% with Disagreement Fusion**

*Robust, Generalizable, Production-Ready*
