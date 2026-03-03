# Face Anti-Spoofing with Smart Late Fusion

## 📊 최종 성능

| Metric | Value |
|--------|-------|
| Overall Accuracy | 99.29% (278/280) |
| Live Accuracy | 97.50% (78/80) |
| Spoof Accuracy | 100.00% (200/200) |
| FAR (False Accept) | 2.50% |
| FRR (False Reject) | 0.00% |

## 🎯 시스템 구성

### Smart Late Fusion
1. **Texture CNN** (EfficientNet-B0)
   - RGB 이미지 기반
   - 성능: 98.57%

2. **Multi-scale LBP** (Random Forest)
   - R=1,2,3 다중 스케일
   - 601 features
   - 성능: 97.08%

3. **Adaptive Integration**
   - 확실한 경우 (95.8%): Texture + LBP
   - 불확실한 경우 (4.2%): + Frequency
   - Threshold: |P - 0.5| < 0.2

## 🔑 주요 기여

1. **Multi-scale LBP 재발견**
   - Single-scale: 77%
   - Multi-scale: 97.08%
   - 거친 텍스처(R=3)가 핵심: 47.4%

2. **Late Fusion 설계**
   - Early Fusion: 88.54% (실패)
   - Late Fusion: 99.29% (성공)
   - 독립성 원칙

3. **Smart Fusion**
   - 적응적 특징 통합
   - Frequency 4.2%만 사용
   - Spoof 100% 달성

4. **Highdef Print 해결**
   - 문제: Client 011 (50%)
   - 해결: Smart Fusion (100%)
   - 전체 Highdef: 100% (48/48)

## 📁 파일 구조

```
face_antispoofing_3weeks/
├── models/
│   ├── texture_expert.py
│   ├── frequency_expert.py (앙상블용)
│   └── baseline.py
├── checkpoints/
│   ├── cross/
│   │   ├── texture.pth
│   │   └── frequency.pth
│   └── lbp_rf_model.pkl
├── images/
│   ├── performance_comparison.png
│   ├── highdef_analysis.png
│   ├── client011_solution.png
│   ├── attack_type_performance.png
│   ├── performance_table.png
│   ├── dataset_composition.png
│   ├── feature_importance.png
│   ├── lbp_analysis.png
│   ├── laplacian_analysis.png
│   └── frequency_failure_comparison.png
└── README.md
```

## 🧪 실험 로그

### 시도했으나 제외
- **FFT CNN**: 89.79%
  - Live: 52.50% (FAR 47.5% - 치명적)
  - Spoof: 97.25%
  - Highdef: 97.5% (우수)
  - 결론: Live 오인이 너무 심각, 제외

### 학습된 교훈
- 주파수 특징만으론 부족 (공간 정보 손실)
- 텍스처 특징이 핵심
- 균형이 중요 (Live vs Spoof)

## 🎓 인사이트

1. **Scale Matters**
   - 거친 텍스처(R=3)가 47.4% 기여
   - 프린트 도트, 종이 섬유 감지

2. **Independence is Key**
   - Early Fusion 실패 (간섭)
   - Late Fusion 성공 (독립)

3. **Adaptive is Better**
   - 모든 케이스에 모든 특징 X
   - 필요할 때만 추가 특징 ○

## 📊 검증

- ✅ Data Leakage: 없음 (Overlap 0)
- ✅ Frame/Video 일관: ±0.5%p
- ✅ Train/Test 분리: 15명 / 35명
- ✅ 재현성: Seed 동일 시 결과 동일

## 🚀 사용 방법

```python
from models.texture_expert import TextureExpert
from models.frequency_expert import FrequencyExpert
import pickle

# 모델 로드
texture = TextureExpert()
texture.load_state_dict(torch.load('checkpoints/cross/texture.pth'))

with open('lbp_rf_model.pkl', 'rb') as f:
    lbp_rf = pickle.load(f)

# 예측
t_prob = get_texture_prob(image)
l_prob = get_lbp_prob(image)

# Smart Fusion
two_way = (t_prob + l_prob) / 2
if abs(two_way - 0.5) < 0.2:
    freq = FrequencyExpert()
    freq.load_state_dict(torch.load('checkpoints/cross/frequency.pth'))
    f_prob = get_freq_prob(image)
    final = (t_prob + l_prob + f_prob) / 3
else:
    final = two_way

prediction = 'Spoof' if final > 0.5 else 'Live'
```

## 📅 프로젝트 타임라인

- Week 1-2: 물리적 변수 정의, Baseline 구축
- Week 3 초반: Multi-Expert 실험
- Week 3 중반: Multi-scale 재발견, Late Fusion
- Week 3 말: Smart Fusion, FFT 실험

## 👤 Author
- 2026.02.21

## 📜 License
MIT

