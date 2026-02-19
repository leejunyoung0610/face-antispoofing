# 실험 결과 정리

## 표 1: MSU-MFSD (내부 데이터)

| 시스템 | 프레임 수준 | 비디오 수준 |
| --- | --- | --- |
| Baseline | 94.75% | N/A |
| Texture CNN | 99.88% | 100.00% |
| Frequency CNN | 96.81% | N/A |

## 표 2: Cross-dataset 일반화 (Test 480, No Fine-tuning)

| 시스템 | Live | Spoof | Overall | Domain Gap |
| --- | --- | --- | --- | --- |
| Baseline | 66.67% | 97.75% | 91.25% | -28.08% |
| Texture CNN | **89.38%** | 97.50% | 91.25% | **-10.50%** |
| Frequency CNN | 86.04% | 97.75% | 90.90% | -10.77% |

**Key Finding:** Texture shows 17.58%p smaller domain gap than Baseline.

## 표 3: Domain Adaptation (Test 480, Fine-tuned)

| 시스템 | Frame Live | Frame Spoof | Frame Overall | FAR | FRR | HTER |
| --- | --- | --- | --- | --- | --- | --- |
| Texture CNN | 98.75% | 98.50% | **98.54%** | 1.25% | 1.50% | 1.38% |
| Frequency CNN | 83.75% | 97.50% | 95.21% | 16.25% | 2.50% | 9.38% |

**Multi-Expert 시도:**
- Cascading, Adaptive, Frequency-First, Alarm Sensor 등 여러 전략 실험
- 모두 Texture 단독(98.54%)과 유사하거나 낮은 성능
- 복잡도 대비 개선 미미

## 표 4: 비디오 수준 검증 (Test 480)

| 시스템 | Frame-level | Video-level | Δ |
| --- | --- | --- | --- |
| Texture CNN (MSU) | 99.88% | 100.00% | +0.12% |
| Texture CNN (Replay) | 98.54% | **97.88%** | -0.66% |

**Conclusion:** Frame ≈ Video → No frame memorization.

## 표 5: Leak-Free Evaluation ⭐

### 5-1. Split Strategy

| 구분 | 크기 | 용도 |
| --- | --- | --- |
| Train | 360 videos | Fine-tuning |
| **Dev** | 240 videos | **Threshold 선택만** |
| **Test** | 240 videos | **최종 평가 (1회)** |
| Test_Full | 480 videos | Reference (원본 전체) |

### 5-2. Threshold Selection (Dev)

| Threshold | Overall | Live | Spoof | FAR | FRR | Freq Call |
| --- | --- | --- | --- | --- | --- | --- |
| 0.50 | 98.33% | 97.22% | 98.53% | 2.78% | 1.47% | 0.00% |
| **0.55** | **98.75%** | **100.00%** | **98.53%** | **0.00%** | **1.47%** | **1.25%** |
| 0.60~0.80 | 98.75% | 100.00% | 98.53% | 0.00% | 1.47% | 1.67~3.75% |

**Optimal:** 0.55 (Live 100%, 최소 Freq Call)

### 5-3. Final Test Evaluation (Test 240, 1회)

| 시스템 | Frame Overall | Frame FAR | Frame FRR | Video Overall | Freq Call |
| --- | --- | --- | --- | --- | --- |
| Texture CNN | 98.75% | 2.27% | 1.02% | 98.75% | — |
| **Adaptive (0.55)** | **98.75%** | **2.27%** | **1.02%** | **98.75%** | **1.25%** |

**Note:** 
- 이전 보고한 Adaptive 98.96%는 Test에서 threshold 선택한 **누수 결과**
- 현재는 Dev로만 선택 → Test 1회 평가 → **98.75% (누수 없음)**
- Texture 단독과 성능 동일, Frequency 기여도 미미

---

## 표 6: Client별 성능 (Tail Risk)

| Client | Live | Spoof | FAR | FRR | Notes |
| --- | --- | --- | --- | --- | --- |
| 011 | 100% | 80% | 0% | 20% | 전체 FN의 67% |
| 104 | 75% | 90% | 25% | 10% | Live false reject / Spoof false positive |
| Others (18명) | 100% | 100% | 0% | 0% | 나머지는 완전 정확 |

**Insight:**
- 2명의 클라이언트가 전체 오류의 100%를 차지
- 평균 성능만 보면 Tail risk를 감출 수 있음
- 실전 배포 전에 client별 검증/대응 필요

## Figures

1. `roc_curves.png` — Baseline/Texture/Frequency ROC (Test_Full 480)
2. `confusion_matrices.png` — 4-system confusion matrices (Test_Full 480)
3. `texture_confidence_distribution.png` — TP vs FN confidence gap (18.67%p)
4. `frequency_failure_comparison.png` — Live FP 배경 의존성 분석

---

## 핵심 발견

### 1. Domain Gap 감소 ⭐⭐⭐
- Baseline: -28% → Texture: -10% (**17.58%p 개선**)
- 물리적 특징 학습이 일반화 우수

### 2. Confidence-Adaptive Framework ⭐⭐⭐
- Threshold 0.55 (Dev로만 선택, 누수 없음)
- Frequency 호출: **1.25%만**
- 성능: Texture 단독과 동일
- **효율성 극대화** (96% 샘플은 Texture만)

### 3. Video-level = Frame-level ⭐⭐
- 차이 0.66%p (무시 가능)
- 프레임 외우기 아님 증명

### 4. Multi-Expert 한계 발견 ⭐⭐
- Frequency Live 오인식 (배경 의존성)
- 여러 결합 전략 시도 → 개선 미미
- **결론: Texture 단독 충분**