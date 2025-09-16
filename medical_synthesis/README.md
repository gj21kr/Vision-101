# Medical Image Synthesis Examples 🏥

이 폴더는 최신 생성 모델을 활용한 의료 영상·볼륨 합성 실험을 한 곳에서 실행할 수 있도록 구성된 통합 예제를 제공합니다.

## 📁 파일 구조

```
medical_synthesis/
├── medical_synthesis_example.py  # 2D/3D 통합 합성 스크립트
└── README.md                     # 이 문서
```

`medical_synthesis_example.py`는 데이터 차원(2D/3D)과 모델 유형(Diffusion, StyleGAN, VAE)을 옵션으로 선택할 수 있는 공통 파이프라인을 제공합니다.

## 🔬 지원 기능 개요

| 구분 | 지원 모델 | 주요 기능 |
| --- | --- | --- |
| 2D | Diffusion, StyleGAN | 흉부 X-ray, CT, MRI 등 다양한 모달리티 합성 · 조건부 병리 생성 · 의료 지표 평가 (PSNR/SSIM) |
| 3D | VAE, Diffusion | CT/MRI 볼륨 생성 · 해부학적 구조 보존 · 3D 병변 합성 · 3D 품질 지표 (PSNR, Cosine Similarity) |

공통 파이프라인은 로깅, 데이터 분할, 모델 초기화, 학습 루프, 샘플 저장을 하나의 인터페이스로 처리하며 데이터 차원에 따른 전처리와 모델 구성만 전략 객체로 분리되어 있습니다.

## ⚙️ 파이썬 API 사용 예시

```python
from medical_synthesis.medical_synthesis_example import (
    MedicalModality,
    VolumeType,
    train_medical_synthesis,
    build_test_loader,
    evaluate_medical_synthesis,
)
import torch

# 1) 2D Diffusion 모델 학습
model_2d, pipeline_2d = train_medical_synthesis(
    data_dim="2d",
    model_type="diffusion",
    modality=MedicalModality.CHEST_XRAY,
    num_epochs=10,
    batch_size=4,
    image_size=128,
)

test_loader_2d = build_test_loader(
    data_dim="2d",
    batch_size=4,
    image_size=128,
    volume_size=(32, 32, 32),  # 2D에서는 사용되지 않음
    modality=MedicalModality.CHEST_XRAY,
    volume_type=VolumeType.CT_CHEST,
)
metrics_2d = evaluate_medical_synthesis(
    model_2d,
    data_dim="2d",
    test_loader=test_loader_2d,
    device=pipeline_2d.device,
    model_type="diffusion",
)
print(metrics_2d)

# 2) 3D VAE 모델 학습
model_3d, pipeline_3d = train_medical_synthesis(
    data_dim="3d",
    model_type="vae",
    volume_type=VolumeType.MRI_BRAIN,
    volume_size=(32, 32, 32),
    num_epochs=10,
    batch_size=2,
)

test_loader_3d = build_test_loader(
    data_dim="3d",
    batch_size=2,
    image_size=128,
    volume_size=(32, 32, 32),
    modality=MedicalModality.CHEST_XRAY,
    volume_type=VolumeType.MRI_BRAIN,
)
metrics_3d = evaluate_medical_synthesis(
    model_3d,
    data_dim="3d",
    test_loader=test_loader_3d,
    device=pipeline_3d.device,
    model_type="vae",
)
print(metrics_3d)
```

## 🚀 CLI 실행 예시

통합 스크립트는 `--data-dim`과 `--model-type` 옵션으로 학습 구성을 지정할 수 있습니다.

```bash
# 2D Diffusion 예제 (흉부 X-ray)
python medical_synthesis/medical_synthesis_example.py \
    --data-dim 2d \
    --model-type diffusion \
    --modality chest_xray \
    --num-epochs 10 \
    --image-size 128

# 2D StyleGAN 예제
python medical_synthesis/medical_synthesis_example.py \
    --data-dim 2d \
    --model-type stylegan \
    --modality mri \
    --image-size 128 \
    --latent-dim 512

# 3D VAE 예제 (뇌 MRI)
python medical_synthesis/medical_synthesis_example.py \
    --data-dim 3d \
    --model-type vae \
    --volume-type mri_brain \
    --volume-size 32 32 32 \
    --num-epochs 10

# 3D Diffusion 예제
python medical_synthesis/medical_synthesis_example.py \
    --data-dim 3d \
    --model-type diffusion \
    --volume-type ct_chest
```

각 실행은 자동으로 학습 로그와 생성 샘플(이미지 또는 NumPy 볼륨)을 `results/medical_synthesis_*` 디렉터리에 저장합니다.

## 🛠️ 통합 파이프라인 구조

- **전략 패턴 적용**: `Diffusion2DStrategy`, `StyleGAN2DStrategy`, `VAE3DStrategy`, `Diffusion3DStrategy`가 데이터 전처리 및 학습 스텝을 담당합니다.
- **공통 학습 루프**: `MedicalSynthesisPipeline`이 장치 선택, 데이터 분할, 로깅, 샘플 저장을 일괄 처리합니다.
- **평가 도구**: `evaluate_medical_synthesis` 함수가 2D/3D에 맞춰 의료 영상 품질 지표를 계산합니다.

## 📊 평가 지표

- **2D**: PSNR, SSIM
- **3D**: 3D PSNR, Volume Cosine Similarity

## 🧱 의존성

```
torch >= 1.9.0
torchvision >= 0.10.0
numpy >= 1.21.0
matplotlib >= 3.4.0
scipy >= 1.7.0
Pillow >= 8.0.0
```

## 📚 추가 참고

- `visualize_3d_volume` 함수를 사용하면 3D 볼륨 샘플을 2D 슬라이스 그리드 형태로 저장할 수 있습니다.
- 로그 및 샘플 저장 형식은 `result_logger.ResultLogger`와 동일하게 유지됩니다.

통합 스크립트를 활용하여 2D/3D 의료 합성 실험을 손쉽게 반복 실행하고 비교할 수 있습니다.
