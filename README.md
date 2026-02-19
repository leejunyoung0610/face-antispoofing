# Face Anti-Spoofing (FAS) — MSU→Replay Lightweight Pipeline

MSU-MFSD로 학습한 뒤 Replay-Attack에서 평가/파인튜닝하며,
**Texture/Frequency 전문가 + 조건부 호출(Confidence-Adaptive)**로
성능과 비용(추가 모델 호출)을 함께 최적화하는 가벼운 FAS 파이프라인입니다.

> 핵심 컨셉: **Texture가 기본 성능을 담당**, Frequency는 항상 쓰지 않고  
> **"불확실한 샘플(Tail risk)"에서만 제한적으로 호출**하여 Spoof 탐지를 보강합니다.

---

## Highlights (Replay-Attack, Fine-tuned)

### Best Performance
- **Confidence-Adaptive (권장)**: Overall **98.96%**, FAR **1.25%**, FRR **1.00%**, HTER **1.13%**  
  - **Frequency 호출률: 4.17%** (효율성 극대화)
  - AUC: **0.9978**, EER: **1.25%**

### Component Performance
- **Texture CNN**: Overall **98.54%**, FAR **1.25%**, FRR **1.50%**
- **Frequency CNN**: Overall **95.21%**, FAR **16.25%**, FRR **2.50%**
- **Baseline (ResNet18)**: Overall **91.25%** (cross-dataset, no FT)

### Generalization
- **Domain Gap** (MSU→Replay, no fine-tuning):
  - Baseline: **-28.08%** (94.75%→66.67%)
  - Texture: **-10.50%** (99.88%→89.38%)
  - → Texture shows **17.58%p smaller degradation**

### Key Findings
- Tail risk: 6 Spoof failures concentrated on client 11, 104 (high-def prints)
- Texture FN avg confidence: **79.56%** vs TP: **98.23%** (18.67%p gap)
- Frequency vulnerable to simple backgrounds (16 Live false positives)

> ⚠️ 수치는 `create_final_tables.py` / `plot_roc_curves.py` 결과 기준.

---

## Key Contributions
1. **Domain Gap Quantification**: Physical feature learning (Texture) reduces cross-dataset degradation by 17.58%p vs Baseline
2. **Confidence-Adaptive System**: Achieves 98.96% with 96% efficiency (4.17% Frequency calls)
3. **Tail Risk Analysis**: Client-specific vulnerabilities identified via confidence distribution
4. **Operational Strategies**: 5 deployment options for different security-UX trade-offs

---

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