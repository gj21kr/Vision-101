# Medical Image Usage Guide

Vision-101의 의료 이미지 특화 기능 사용법 가이드입니다.

## 🏥 개요

Vision-101에는 의료 이미지에 특화된 다음 기능들이 구현되어 있습니다:

### 📂 파일 구조
```
Vision-101/
├── medical_data_utils.py           # 의료 데이터 로더
├── result_logger.py               # 결과 자동 저장 시스템
├── run_medical_tests.py           # 통합 테스트 스크립트
├── generating/
│   ├── vae_medical_example.py     # 의료 이미지 VAE
│   └── gan_medical_example.py     # 의료 이미지 GAN
├── 3d/
│   └── nerf_medical_example.py    # 의료 볼륨 NeRF
└── results/                       # 모든 결과가 여기에 저장됨
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 기본 패키지 설치
pip install torch torchvision numpy matplotlib pillow scikit-learn

# 선택적 의료 이미지 라이브러리 (DICOM, NIfTI 지원)
pip install pydicom nibabel

# OpenCV (이미지 전처리용)
pip install opencv-python
```

### 2. 전체 알고리즘 테스트

```bash
# 모든 알고리즘을 chest X-ray 데이터로 테스트
python run_medical_tests.py --algorithms all --dataset chest_xray

# 빠른 테스트 (각 알고리즘 5 epochs만)
python run_medical_tests.py --algorithms all --dataset chest_xray --quick-test

# 특정 알고리즘만 실행
python run_medical_tests.py --algorithms vae gan --dataset brain_mri
```

### 3. 개별 알고리즘 실행

#### VAE (Variational Autoencoder)
```bash
cd generating
python vae_medical_example.py
```

#### GAN (Generative Adversarial Network)
```bash
cd generating
python gan_medical_example.py
```

#### NeRF (Neural Radiance Fields)
```bash
cd 3d
python nerf_medical_example.py
```

## 📊 지원하는 의료 이미지 타입

### 1. 흉부 X-ray (Chest X-ray)
- **특징**: 폐, 심장, 늑골 구조
- **용도**: 폐렴, 결핵 진단 보조
- **형식**: Grayscale (1채널)
- **예시 데이터셋**: ChestX-ray14, Montgomery County

```python
from medical_data_utils import load_chest_xray_data

# 합성 데이터 생성
images = load_chest_xray_data(num_samples=1000, image_size=256)

# 실제 데이터 로드 (경로 지정 시)
images = load_chest_xray_data('/path/to/chest_xray_data', num_samples=1000)
```

### 2. 뇌 MRI (Brain MRI)
- **특징**: 뇌 조직, 뇌실, 백질/회질
- **용도**: 종양, 뇌졸중 진단 보조
- **형식**: Grayscale (1채널)
- **예시 데이터셋**: BraTS, ADNI

```python
from medical_data_utils import load_brain_mri_data

images = load_brain_mri_data(num_samples=1000, image_size=256)
```

### 3. 피부 병변 (Skin Lesion)
- **특징**: 피부 표면, 색소 침착
- **용도**: 멜라노마, 피부암 진단 보조
- **형식**: RGB (3채널)
- **예시 데이터셋**: ISIC 2020

```python
from medical_data_utils import load_skin_lesion_data

images = load_skin_lesion_data(num_samples=1000, image_size=256)
```

## 🛠 커스텀 의료 데이터 사용법

### DICOM 파일 로드
```python
from medical_data_utils import MedicalImageLoader

loader = MedicalImageLoader('custom', image_size=512)

# DICOM 디렉토리 로드
images = loader.load_real_dataset('/path/to/dicom/files', max_samples=500)

# 통계 확인
stats = loader.get_data_statistics(images)
print(f"Loaded {stats['num_images']} images")
print(f"Image shape: {stats['image_shape']}")
```

### NIfTI 파일 로드 (3D 볼륨)
```python
# 3D 볼륨 데이터 생성
volumes = loader.create_synthetic_medical_data(num_samples=10, data_type='3d')

# NIfTI 파일에서 로드
# loader는 자동으로 중간 슬라이스를 선택하여 2D로 변환
images = loader.load_real_dataset('/path/to/nifti/files')
```

## 📈 결과 확인 및 해석

### 자동 저장되는 결과물

모든 알고리즘 실행 시 다음이 자동으로 저장됩니다:

#### 1. 훈련 진행 상황
- `logs/training.log`: 상세 훈련 로그
- `metrics/metrics.json`: 손실값, 정확도 등 수치 데이터
- `plots/training_curves.png`: 훈련 곡선 그래프

#### 2. 생성 결과물
- `images/original_samples.png`: 원본 의료 이미지 샘플
- `images/generated_samples_epoch_XXX.png`: 에포크별 생성 결과
- `images/final_generated_samples.png`: 최종 생성 결과

#### 3. 모델 파일
- `models/XXX_final_model.pth`: 최종 훈련된 모델
- `models/XXX_checkpoint_epoch_XXX.pth`: 중간 체크포인트

#### 4. 설정 및 요약
- `logs/config.json`: 실험 설정
- `experiment_summary.json`: 실험 요약

### 결과 디렉토리 구조 예시
```
results/
├── generating_vae_chest_xray_20241214_143022/
│   ├── images/
│   │   ├── original_samples.png
│   │   ├── generated_samples_epoch_005.png
│   │   ├── generated_samples_epoch_010.png
│   │   └── final_generated_samples.png
│   ├── models/
│   │   ├── vae_checkpoint_epoch_010.pth
│   │   └── vae_final_model.pth
│   ├── logs/
│   │   ├── training.log
│   │   └── config.json
│   ├── metrics/
│   │   └── metrics.json
│   └── plots/
│       └── training_curves.png
└── 3d_nerf_brain_mri_20241214_143525/
    └── ... (similar structure)
```

## 🎯 의료 이미지별 최적 파라미터

### VAE 파라미터
```python
# 흉부 X-ray
config = {
    'latent_dim': 32,
    'hidden_dim': 512,
    'learning_rate': 1e-3,
    'epochs': 50
}

# 뇌 MRI
config = {
    'latent_dim': 64,  # 더 복잡한 구조로 인해 큰 latent space
    'hidden_dim': 1024,
    'learning_rate': 5e-4,
    'epochs': 100
}

# 피부 병변 (컬러 이미지)
config = {
    'latent_dim': 128,  # RGB 채널로 인해 가장 큰 latent space
    'hidden_dim': 1024,
    'learning_rate': 1e-3,
    'epochs': 80
}
```

### GAN 파라미터
```python
# 의료 이미지 GAN 공통 설정
config = {
    'generator_lr': 2e-4,
    'discriminator_lr': 2e-4,
    'batch_size': 64,
    'beta1': 0.5,  # Adam optimizer momentum
    'epochs': 200
}
```

## 📋 품질 평가 메트릭

### 자동 계산되는 메트릭
1. **Inception Score (IS)**: 생성 이미지의 다양성과 품질
2. **Reconstruction Error**: VAE의 재구성 품질
3. **Discriminator Accuracy**: GAN의 훈련 균형
4. **PSNR/SSIM**: 이미지 품질 측정 (해당하는 경우)

### 의료 이미지 특화 평가
```python
# 생성된 의료 이미지의 해부학적 구조 평가
def evaluate_medical_image_quality(generated_images, image_type):
    if image_type == 'chest_xray':
        # 폐 영역, 심장 윤곽, 늑골 구조 평가
        return evaluate_chest_structures(generated_images)
    elif image_type == 'brain_mri':
        # 뇌 조직 대비, 뇌실 구조 평가
        return evaluate_brain_structures(generated_images)
    # ... 등등
```

## ⚠ 주의사항 및 한계

### 1. 윤리적 고려사항
- ⚠️ **이 구현은 교육 목적으로만 사용하세요**
- ⚠️ **실제 의료 진단에 사용하지 마세요**
- ⚠️ **환자 데이터 사용 시 IRB 승인 필요**
- ⚠️ **생성된 이미지의 의료적 유효성은 검증되지 않음**

### 2. 기술적 한계
- Synthetic 데이터는 실제 의료 데이터와 차이가 있음
- 작은 데이터셋으로 인한 overfitting 가능성
- GPU 메모리 사용량이 높을 수 있음

### 3. 개선 방향
- 더 큰 실제 의료 데이터셋 사용
- 도메인 전문가와의 협업
- 의료 이미지 특화 loss function 개발
- 규제 요구사항 준수

## 🔧 문제 해결

### 일반적인 오류들

#### 1. 메모리 부족
```bash
# GPU 메모리 부족 시
export CUDA_VISIBLE_DEVICES=""  # CPU만 사용

# 또는 배치 사이즈 줄이기
python vae_medical_example.py --batch_size 32
```

#### 2. 의존성 설치 오류
```bash
# DICOM 지원이 필요하지 않다면
pip install Vision-101 --no-dicom

# 또는 conda 사용
conda install -c conda-forge pydicom nibabel
```

#### 3. 권한 오류 (결과 저장)
```bash
# results 디렉토리 권한 확인
chmod 755 results/
mkdir -p results/
```

## 📞 지원 및 기여

### 버그 리포트
- GitHub Issues에 상세한 오류 로그와 함께 제보
- 사용한 데이터셋과 파라미터 정보 포함

### 기여 방법
1. 새로운 의료 이미지 타입 지원 추가
2. 의료 이미지 특화 전처리 기법 구현
3. 평가 메트릭 개선
4. 문서화 개선

## 📚 추가 리소스

### 권장 의료 데이터셋
- **ChestX-ray14**: NIH에서 공개한 흉부 X-ray 데이터셋
- **MIMIC-CXR**: MIT에서 공개한 대규모 흉부 X-ray
- **BraTS**: 뇌종양 세분화 챌린지 데이터
- **ISIC**: 국제 피부 이미징 협력 아카이브

### 관련 논문
- Esteva, A., et al. (2017). "Dermatologist-level classification of skin cancer with deep neural networks."
- Rajpurkar, P., et al. (2017). "CheXNet: Radiologist-level pneumonia detection on chest X-rays."
- Litjens, G., et al. (2017). "A survey on deep learning in medical image analysis."

---

**⚠️ 면책 조항**: 이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다. 실제 의료 진단이나 치료에 사용해서는 안 됩니다.