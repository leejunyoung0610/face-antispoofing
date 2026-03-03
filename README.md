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
- **Tail Risk**: Client 011/104 동일 100% 오류 → Texture confidence gap 존재
- **Multi-Expert**: Frequency 호출이 증가할수록 실 성능 악화, 복잡도만 늘어남
- **Domain Gap**: Texture **-10.50%** (Baseline **-28.08%**)

### Component Performance
- **Frequency CNN**: Overall **95.21%**, FAR **16.25%**, FRR **2.50%** (Live False Positive 16%)
- **Baseline (ResNet18)**: Overall **91.25%** (cross-dataset, no FT)

### Key Findings
- **Texture 단독**이 가장 강건: 98.54%로 Multi-Expert보다 우수하거나 동일한 결과
- **Client 011 취약성**: 고품질 print·단순 배경에서 FRR 폭증, 개인별 취약성 고려 필요
- **Multi-Expert 한계**: Frequency 호출률 4~5% 넘으면 Live 오인 급증 → 전체 FAR/FRR 악화

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
## Key Findings
1. **Texture 단독이 최적**: 98.54%를 단순한 구조로 달성, Multi-Expert 대비 FAR/FRR도 동등하거나 우수
2. **Client 011 취약성**: 고품질 프린트·단순 배경에서 FRR 폭증, 개인별 대응 필요
3. **Multi-Expert 한계**: Frequency 호출률 4~5% 넘으면 Live 오인(FAR) 증가 → 전체 성능 악화

## Final Recommendation
- **Texture CNN 단독 사용**을 권장합니다 (98.54%, 단순함, 강건함).
- Multi-Expert는 Tail risk나 보안 요구가 특별히 강한 환경에만 제한적 적용.

## Future Work
- 배경 복잡도 기반 Frequency 호출 조건 정밀화
- Client-specific threshold (011/104) 및 모니터링 경고
- Depth/geometry 정보 결합으로 Texture/Adaptive 신뢰도 증강

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

6-channel input:
- RGB (3 channels)
- Multi-scale LBP (3 channels: R=1,2,3)

→ EfficientNet-B0 학습
```

### 6.2 실패 결과
```
Train: 100% (빠르게 과적합)
Test:  88.54%
  Live:  32.50% (심각!)
  Spoof: 99.75%
```

### 6.3 실패 원인 분석

**가설: LBP가 너무 강한 신호**
```
LBP 특징:
→ "Spoof = 특정 패턴" (명확한 규칙)
→ 매우 강한 신호

학습 과정:
1. CNN이 6개 채널 입력 받음
2. LBP 채널이 Spoof 강한 신호 제공
3. CNN이 LBP 신호에만 의존
4. RGB 정보 무시!

결과:
→ LBP의 편향(Spoof 선호) 그대로 학습
→ "조금이라도 의심스러우면 Spoof"
→ Live 대량 오인
```

**인사이트:**
> "Early Fusion은 **강한 신호가 약한 신호를 지배**한다. 명시적 특징의 편향이 CNN 학습을 왜곡시켰다."

---

## 7. Late Fusion 성공: 독립성이 핵심

### 7.1 전략 전환

**깨달음:**
> "함께 학습시키면 간섭한다. 따로 학습하고 나중에 결합하자!"

**Late Fusion 설계:**
```
독립 학습:
1. Texture CNN → 98.54% (RGB 학습)
2. Multi-LBP RF → 97.08% (LBP 학습)

결합:
Score_final = (Score_CNN + Score_LBP) / 2
```

### 7.2 성공 결과
```
Frame-level: 98.96%
Video-level: 99.29% 🏆

개별 비교:
- Texture CNN: 98.57% (Video)
- Late Fusion:  99.29% (Video)
→ +0.72%p 개선!
```

### 7.3 상세 분석: 왜 성공했는가?

**Live 성능 (동일):**
```
Texture CNN:  79/80 = 98.75%
Late Fusion:  79/80 = 98.75%
→ 완전 동일!
```

**Spoof 성능 (개선!):**
```
Texture CNN:  197/200 = 98.50%
Late Fusion:  199/200 = 99.50%
→ +1.0%p 개선!

잡은 2개:
→ LBP의 Spoof 강점 활용!
```

**상호보완 메커니즘:**
```
각 모델의 강점:
- Texture CNN: Live 인식 강함, Spoof 보통
- Multi-LBP:   Live 약함 (86%), Spoof 강함 (99%)

Average 시:
→ Live: CNN 주도 (98.75%)
→ Spoof: LBP 보조 (99.50%)

→ 완벽한 Complementary!
```

**인사이트:**
> "Late Fusion은 각 모델이 **독립적으로 최적화**되어 편향 없이 강점을 보존한다. 결합 시 상호보완으로 약점을 메운다."

---

## 8. Tail Risk 분석: 평균 너머의 진실

### 8.1 문제 제기

> "99.29%면 거의 완벽한가? 모든 사람에게 동등하게 작동하는가?"

### 8.2 Client별 성능 분석

**전체 FN (False Negative) 6개:**
```
Client 011: 4개 (67%)
Client 104: 2개 (33%)
Others (18명): 0개

→ 2명이 전체 오류의 100%!
```

### 8.3 Client 011 심층 분석

**실패 케이스 특징:**
```
모두 Highdef Photo Print:
- attack_print_client011_highdef_photo_controlled
- attack_highdef_client011_highdef_photo_adverse

Texture 확신도: 0.19, 0.29 (매우 낮음!)
Frequency 확신도: 0.75, 0.99 (높음)

→ Texture가 "확신 없이" 틀림
```

**Temporal 분석 (10 프레임):**
```
Frame  0: Spoof 0.514 (유일하게 의심)
Frame 23: Spoof 0.310
...
Frame 207: Spoof 0.089

다수결: 1/10 Spoof → Live (실패!)

→ 거의 모든 프레임이 Live처럼 보임
→ Temporal 정보로도 못 잡음!
```

**시각적 분석:**

비교 실험:
- fail_client011.jpg (Texture 0.19)
- success_client014.jpg (Texture 0.81)

**차이점:**
```
Client 011 (실패):
✅ 매우 고품질 프린트 (300+ DPI)
✅ 피부 텍스처 극도로 균일
✅ 배경 단순 (회색 벽)
✅ 조명 완벽
✅ 프린트 artifact 거의 없음

→ 4가지가 완벽히 결합
→ 실제 Live와 구별 불가능!

Client 014 (성공):
⚠️ 배경 복잡 (창문, 블라인드)
⚠️ 프린트 artifact 존재
⚠️ 엣지 약간 뭉개짐
```

**정의: "Perfect Print Vulnerability"**

특징:
1. 개인 피부 특성 (원래 매끄러움)
2. 고품질 프린트 (Highdef 300+ DPI)
3. 단순 배경 (회색 벽)
4. 좋은 조명 (Controlled)

**인사이트:**
> "평균 99.29%는 모든 개인에게 동등하지 않다. **개인별 취약성**이 존재하며, 고품질 프린트 + 단순 배경 + 매끄러운 피부의 조합은 현재 시스템의 한계다."

---

## 9. 검증: 환상이 아닌 진짜 성능

### 9.1 Data Leakage 제거

**Week 3 발견:**
```
초기 Adaptive 98.96%:
→ Threshold를 Test로 선택 (누수!)

해결:
Train/Dev/Test 3분할
- Train (360): 학습
- Dev (240): Threshold 선택만
- Test (240): 최종 평가 1회

→ Leak-free 달성!
```

### 9.2 Video-level 검증

**목적:** Frame memorization 배제
```
Frame-level vs Video-level:

Texture CNN:
Frame: 98.54%
Video: 98.57% (+0.03%p)

Late Fusion:
Frame: 98.96%
Video: 99.29% (+0.33%p)

→ 거의 동일!
→ 과적합 없음!
```

### 9.3 Cross-dataset 검증
```
Domain Gap 분석:

Baseline (ResNet-18):
MSU → Replay: -28.08% (collapse!)

Texture CNN:
MSU → Replay: -10.50%
→ 17.58%p 개선!

→ 물리적 특징의 일반화 능력 우수!
```

### 9.4 Stress Test

**실전 환경 시뮬레이션:**
```
Transform         Performance  평가
Original          98.54%       Baseline
JPEG Q=90         98.12%       ✅ 실용적
JPEG Q=70         96.67%       ✅ 허용
Brightness ×1.3   92.71%       ✅ 양호
Blur σ=1.0        91.46%       ⚠️ 보통
Brightness ×0.7   87.29%       ⚠️ 취약
Noise σ=0.01      87.29%       ⚠️ 취약

→ 실용적 조건(Q≥70)에서 96%+
→ 극단적 조건에서만 취약
```

---

## 10. 최종 시스템 아키텍처

### 10.1 Late Fusion Ensemble
```
Input Image
     |
     ├─→ [RGB] → Texture CNN (EfficientNet-B0)
     |              |
     |           98.57% (암묵적 특징)
     |              ↓
     |          Live 강점
     |
     └─→ [Multi-scale LBP] → Random Forest
                  |
              97.08% (명시적 특징)
                  ↓
              Spoof 강점
              
              ↓
         Simple Average
              ↓
      Video-level: 99.29% 🏆
```

### 10.2 성능 비교
```
=== 최종 성능 비교 (Video-level) ===

1위 🥇 Late Fusion:         99.29%
   Live:  98.75%, Spoof: 99.50%
   FN: 2개 (Live 1, Spoof 1)
   
2위 🥈 Texture CNN:         98.57%
   Live:  98.75%, Spoof: 98.50%
   FN: 4개 (Live 1, Spoof 3)
   
3위 🥉 LBP + Laplacian:     97.50%
   명시적 특징 결합
   
4위    Multi-scale LBP:     97.08%
   명시적 특징 단독
   
5위    Laplacian:           88.75%
6위    FFT:                 75.62%
```

---

## 11. 핵심 기여 및 인사이트

### 11.1 Multi-scale의 재발견 ⭐⭐⭐
```
발견 과정:
Single-scale 77% (실패)
→ "LBP는 한계가 있다"
→ 3개월 후 재시도
→ Multi-scale 97.08% (성공!)

핵심:
R=3 (거친 텍스처)가 가장 중요!
→ 전역적 구조 차이가 핵심

기여:
"Scale이 중요하다"는 것을 체계적으로 검증
```

### 11.2 명시적 vs 암묵적 특징 비교 ⭐⭐
```
명시적 (Multi-LBP): 97.08%
암묵적 (CNN):       98.57%
Gap:                1.49%p만!

의미:
→ 명시적 특징도 충분히 효과적
→ 해석 가능성 + 고성능 양립 가능
→ "CNN이 모든 답"은 아님
```

### 11.3 물리적 변수 체계적 검증 ⭐⭐
```
5개 정의 → 3개 구현 → 기여도 측정:

텍스처:   76.2% (핵심!)
평면성:   23.2% (보조)
주파수:   0.6%  (무의미)

발견:
→ 국소 특징 >> 전역 특징
→ 텍스처가 본질
→ 주파수는 배경에 과의존
```

### 11.4 Late Fusion의 효과 검증 ⭐
```
Early Fusion: 88.54% (실패)
→ 강한 신호가 약한 신호 지배
→ LBP가 CNN 학습 방해

Late Fusion: 99.29% (성공!)
→ 독립 학습 → 편향 없음
→ 상호보완 극대화

원칙:
"서로 다른 편향을 가진 모델은
 독립적으로 학습하고 나중에 결합하라"
```

### 11.5 Tail Risk 발견 ⭐⭐⭐
```
2명(Client 011, 104)이 전체 오류 100%

"Perfect Print Vulnerability":
- 고품질 프린트
- 단순 배경
- 매끄러운 피부
- 좋은 조명

의미:
평균 ≠ 개인별 성능
→ 실전 배포 시 개인별 검증 필요
→ Worst-case 관리 필수
```

---

## 12. 한계 및 Future Work

### 12.1 시스템 한계

**1. 고품질 프린트 취약 (Client 011)**

현상: 300+ DPI highdef print + 단순 배경
해결 방향:
- 배경 복잡도 기반 재검증
- Client-specific threshold
- Depth 정보 추가

**2. Temporal 정보 미활용**

현상: 단일 프레임 기반 판단
발견: Client 011의 10 프레임 봐도 못 잡음 (고품질은 모든 프레임 완벽)
해결 방향:
- LSTM, 3D CNN (움직임 패턴)
- 미세 떨림, 깜빡임 감지

**3. 주파수 특징의 배경 의존성**

현상: 단순 배경 Live를 Spoof로 오인
해결 방향:
- 얼굴만 Crop (배경 제거)
- 국소 FFT (패치 단위)
- 방향별 FFT (Radial 대신)

### 12.2 Future Work

**단기 (1-2개월):**
- Depth 정보 통합 (MiDaS 등)
- 배경 복잡도 기반 Adaptive system
- OULU, CASIA 추가 검증

**중기 (3-6개월):**
- Temporal 모델링 (LSTM/3D CNN)
- Attention mechanism (어디를 보는가?)
- Client-specific adaptation

**장기 (6개월+):**
- Self-supervised learning
- Vision Transformer
- Cross-modal fusion (RGB + IR + Depth)

---

## 13. 결론

본 프로젝트는 **"왜 Live와 Spoof가 다른가?"**라는 근본적 질문에서 시작하여, 물리적 관찰 → 변수 정의 → 명시적 구현 → 체계적 검증의 전 과정을 수행했다.

### 13.1 핵심 성과

**기술적 성과:**
- Video-level 99.29% 달성
- 명시적 특징(Multi-LBP) 97.08%
- CNN과의 Gap 1.49%p만

**방법론적 기여:**
- Multi-scale의 중요성 재발견 (R=3 거친 텍스처 핵심)
- 명시적 vs 암묵적 체계적 비교
- Late Fusion의 독립성 원칙 검증
- Tail Risk 발견 (개인별 취약성)

### 13.2 학습한 교훈

**1. "실패는 과정이다"**
```
Single-scale LBP 77% (실패)
→ 3개월 후
→ Multi-scale LBP 97.08% (성공)

→ 실패에서 배워 개선하는 것이 핵심
```

**2. "Scale matters"**
```
R=1 (미세) vs R=3 (거친)
→ 거친 텍스처가 핵심!
→ 세밀함보다 전역 구조
```

**3. "물리적 근거의 힘"**
```
명시적 특징 97.08%
vs
CNN 98.57%

→ Gap 1.49%p만
→ 해석 가능 + 고성능 양립
```

**4. "평균 너머를 보라"**
```
99.29% 평균
→ 하지만 Client 011은 80%
→ Tail Risk 관리 필수
```

### 13.3 최종 메시지

> **"단순히 높은 정확도를 추구한 것이 아니다. 물리적 세계를 관찰하고, 차이를 정의하고, 측정 가능하게 만들고, 체계적으로 검증했다. 실패를 분석하고, 원인을 찾고, 개선했다. 이것이 진짜 엔지니어링이다."**

---

## Appendix

### A. 데이터셋

- MSU-MFSD: 280 train, 200 test
- Replay-Attack: 360 train, 240 dev, 240 test (480 test_full)

### B. 하드웨어

- Apple M4, 16GB RAM
- PyTorch 2.0, MPS backend

### C. 코드 구조
```
face_antispoofing_3weeks/
├── models/
│   ├── texture_expert.py
│   ├── frequency_expert.py
│   └── ...
├── utils/
│   └── dataset.py
├── checkpoints/
└── experiments/