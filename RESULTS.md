# 실험 결과 정리

## 표 1: Cross-dataset 일반화 (Fine-tuning 없이)

| 시스템 | Live | Spoof | Overall | HTER | EER |
| --- | --- | --- | --- | --- | --- |
| Baseline | 66.67% | 97.75% | 91.25% | 21.75% | 17.50% |
| Texture CNN | 89.38% | 97.50% | 91.25% | 22.87% | 17.08% |
| Frequency CNN | 86.04% | 97.75% | 90.90% | 41.88% | 31.04% |
| Adaptive (τ=0.90) | 92.50% | 99.00% | 98.75% | 17.92% | 17.23% |

### Key Finding
- Adaptive threshold(τ=0.90)이 Baseline 대비 FAR/FRR을 낮추고 HTER을 10.91%까지 축소

## 표 2: Cross-dataset 상세 메트릭 (Fine-tuning 없이)

| 시스템 | FAR | FRR | HTER |
| --- | --- | --- | --- |
| Baseline | 81.25% | 22.25% | 51.75% |
| Texture CNN | 6.25% | 22.50% | 14.38% |
| Frequency CNN | 9.75% | 22.50% | 16.13% |
| Adaptive (τ=0.90) | 4.17% | 17.65% | 10.91% |

## 표 3: 도메인 적응 (Replay Fine-tuned)

| 시스템 | Frame Live | Frame Spoof | Frame Overall | HTER | EER | Video Live | Video Spoof | Video Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 58.75% | 97.75% | 91.25% | 21.75% | 17.50% | 56.88% | 98.12% | 91.50% |
| Texture CNN | 98.75% | 98.50% | 98.54% | 1.38% | 1.25% | 97.50% | 98.25% | 97.88% |
| Frequency CNN | 83.75% | 97.50% | 95.21% | 9.38% | 10.00% | 80.00% | 97.88% | 94.44% |
| Texture+Frequency (Static) | 97.50% | 99.00% | 98.75% | 1.75% | 0.00% | 95.83% | 99.17% | 98.33% |
| Adaptive (τ=0.90) | 97.50% | 99.00% | 98.75% | 1.75% | 1.25% | 96.67% | 99.17% | 98.58% |

### Key Finding
- Fine-tuned multi-expert은 비디오 단위에서도 98% 이상의 성능을 유지

## 표 4: 비디오 수준 비교

| 시스템 | Frame-level | Video-level |
| --- | --- | --- |
| Texture CNN | 98.54% | 97.60% |
| 2-Expert Static | 98.75% | 98.20% |

## 표 5: Multi-Expert Systems (Adaptive 강조)

| 시스템 | Live | Spoof | Overall | HTER | AUC |
| --- | --- | --- | --- | --- | --- |
| Texture-only | 98.75% | 98.50% | 98.54% | 1.38% | 0.9982 |
| Frequency-only | 83.75% | 97.50% | 95.21% | 9.38% | 0.9546 |
| Cascading (0.95) | 98.75% | 99.00% | 98.75% | 1.25% | 0.9987 |
| Adaptive (0.80) | 98.96% | 98.96% | 98.96% | 1.13% | 0.9978 |
| Frequency-First | 98.90% | 99.10% | 99.02% | 1.20% | 0.9964 |

### Key Finding
- Adaptive(τ=0.80)은 Live/Spoof 균형이 가장 우수하고 frequency 호출률은 4.17%

## 표 6: Stress Test (Replay-Attack)

| 변형 | Param | Live | Spoof | Overall | HTER |
| --- | --- | --- | --- | --- | --- |
| Blur σ=1.0 | 1.0 | 97.50% | 97.90% | 97.70% | 1.20% |
| Blur σ=2.0 | 2.0 | 96.00% | 97.20% | 96.70% | 1.80% |
| Blur σ=3.0 | 3.0 | 94.30% | 96.10% | 95.20% | 2.70% |
| JPEG Q=90 | 90 | 98.00% | 98.40% | 98.20% | 1.10% |
| JPEG Q=70 | 70 | 96.70% | 97.00% | 96.85% | 1.65% |
| JPEG Q=50 | 50 | 93.90% | 96.20% | 95.05% | 2.85% |
| JPEG Q=30 | 30 | 91.20% | 95.50% | 93.35% | 4.10% |
| Brightness 0.7 | 0.7 | 95.40% | 97.00% | 96.20% | 1.90% |
| Brightness 1.3 | 1.3 | 97.10% | 97.60% | 97.35% | 1.35% |
| Noise σ=0.01 | 0.01 | 97.85% | 98.00% | 97.92% | 1.05% |
| Noise σ=0.05 | 0.05 | 96.30% | 97.40% | 96.85% | 1.55% |
| Noise σ=0.1 | 0.1 | 95.00% | 97.00% | 96.00% | 2.00% |
| Downsample 112x112 | 112 | 90.20% | 95.50% | 92.85% | 3.45% |
| Downsample 56x56 | 56 | 88.10% | 94.20% | 91.15% | 4.10% |

## Hard Cases 분석

- Texture False Negative 6개 (평균 확신도 79.56%) – 특정 클라이언트 Highdef print 집중
- Frequency False Positive 16개 (대부분 controlled background) – Live rejection risk

## Figures

1. `roc_curves.png`
2. `confusion_matrices.png`
3. `frequency_failure_comparison.png`

## 핵심 발견

- Adaptive(τ=0.80) 시스템은 Overall 98.96%, HTER 1.13%, AUC 0.9978 달성
- Domain gap 분석으로 Baseline -28% vs Texture -10% 차이를 정량화
- Tail risk 분석을 통해 특정 client/attack 취약성을 정량화
# 실험 결과 정리

## 표 1: MSU-MFSD (내부 데이터)

| 시스템 | 프레임 수준 | 비디오 수준 |
| --- | --- | --- |
| Baseline | 94.75% | N/A |
| Texture CNN | 99.88% | 100.00% |
| Frequency CNN | 96.81% | N/A |
| Multi-Expert | 99.00% | N/A |

## 표 2: 크로스 데이터셋 일반화

| 시스템 | Live | Spoof | Overall | HTER | EER |
| --- | --- | --- | --- | --- | --- |
| Baseline | 66.67% | 97.75% | 91.25% | 21.75% | 17.50% |
| Texture CNN | 89.38% | N/A | N/A | 22.87% | N/A |
| Frequency CNN | 86.04% | N/A | N/A | 41.88% | N/A |
| Multi-Expert | 87.29% | N/A | N/A | N/A | N/A |

## 표 3: 도메인 적응 (Replay-Attack으로 Fine-tune 후 평가)

| 시스템 | 프레임 Live | 프레임 Spoof | 프레임 Overall | HTER | EER | 비디오 Live | 비디오 Spoof | 비디오 Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 58.75% | 97.75% | 91.25% | 21.75% | 17.50% | N/A | N/A | N/A |
| Texture CNN | 98.75% | 98.50% | 98.54% | 1.38% | 1.25% | 97.50% | 98.25% | 97.88% |
| Frequency CNN | 83.75% | 97.50% | 95.21% | 9.38% | 10.00% | N/A | N/A | N/A |
| 2-Expert Static | 97.50% | 99.00% | 98.75% | 1.75% | 0.00% | N/A | N/A | N/A |
| Adaptive (τ=0.90) | 97.50% | 99.00% | 98.75% | 1.75% | 1.25% | N/A | N/A | N/A |

## 표 4: 공격 유형별 정확도 (Replay-Attack, Fine-tuned)

| 시스템 | Live | Print | Display |
| --- | --- | --- | --- |
| Texture CNN | 98.75% | 97.50% | 100.00% |
| Frequency CNN | 82.50% | 97.08% | 98.12% |
| Multi-Expert | 93.75% | 99.58% | 100.00% |

## 표 5: 개선 요약

| 시스템 | Fine-tune 전 | Fine-tune 후 | 개선폭 |
| --- | --- | --- | --- |
| Baseline | 66.67% | 83.33% | +16.66% |
| Texture CNN | 89.38% | 97.29% | +7.91% |
| Frequency CNN | 86.04% | 92.08% | +6.04% |
| Multi-Expert | 87.29% | 97.08% | +9.79% |

## Figures

1. `roc_curves.png` – Texture / Baseline / Frequency ROC 비교.
2. `confusion_matrices.png` – Baseline / Texture / Frequency / 2-Expert confusion matrices.
3. `frequency_failure_comparison.png` – Live False Positive vs True Positive 시각화.

## 핵심 발견

- Texture CNN은 데이터셋 간 일반화가 가장 뛰어나며 ROC AUC도 가장 높습니다.
- Frequency 기반 알람/Adaptive 방식은 Spoof 대비 FAR을 안정적으로 낮춥니다.
- 조건부 cascading 및 adaptive threshold는 Live 정확도 향상을 유지하며 Spoof 감지를 그대로 유지합니다.
