# Medical Image Synthesis Examples 🏥

이 폴더는 최신 생성 모델을 활용한 의료 영상 합성 예제들을 포함합니다.

## 📁 파일 구조

```
medical_synthesis/
├── medical_2d_synthesis_example.py    # 2D 의료 영상 합성
├── medical_3d_synthesis_example.py    # 3D 의료 볼륨 합성
└── README.md                          # 이 문서
```

## 🔬 2D Medical Image Synthesis

### 지원 기능
- **최신 Diffusion Models**: DDPM을 활용한 고품질 의료 영상 생성
- **Medical-specific StyleGAN**: 의료 영상에 특화된 StyleGAN 구현
- **다중 모달리티 지원**:
  - 흉부 X-ray
  - CT 스캔
  - MRI
  - 유방촬영술 (Mammography)
  - 초음파
  - 망막 영상

### 조건부 생성 특성
- 병리학적 상태 (정상/폐렴/결절 등)
- 중증도 (0-1 범위)
- 병변 위치 및 크기
- 환자 정보 (나이, 성별)

### 의료 영상 특화 기능
- **Edge Preservation Loss**: 의료 영상의 세부 구조 보존
- **Perceptual Loss**: 시각적 품질 향상
- **Medical Quality Metrics**: PSNR, SSIM 등 의료 영상 평가 지표

### 사용 예제
```python
from medical_synthesis.medical_2d_synthesis_example import *

# 흉부 X-ray 생성 모델 훈련
model = train_medical_2d_synthesis(
    modality=MedicalModality.CHEST_XRAY,
    model_type="diffusion",  # 또는 "stylegan"
    num_epochs=100,
    batch_size=8,
    image_size=256
)

# 평가
metrics = evaluate_medical_2d_synthesis(model, test_loader, device)
print(f"PSNR: {metrics['psnr']:.2f} dB")
```

## 🧠 3D Medical Volume Synthesis

### 지원 기능
- **3D Diffusion Models**: 볼륨 데이터를 위한 3D DDPM
- **3D Variational Autoencoders**: 고해상도 3D 볼륨 생성
- **NeRF-Medical**: 3D 의료 장면 합성 (향후 확장)

### 지원 볼륨 타입
- **CT 스캔**:
  - 흉부 CT
  - 복부 CT
  - 두부 CT
- **MRI**:
  - 뇌 MRI
  - 심장 MRI
  - 척추 MRI

### 3D 특화 기능
- **해부학적 구조 보존**: 실제 의료 구조와 일치하는 생성
- **다중 장기 볼륨**: 여러 장기가 포함된 복합 볼륨
- **3D 병변 시뮬레이션**: 종양, 병변, 이상 소견 생성
- **메모리 효율적 처리**: 대용량 3D 데이터 처리 최적화

### 조건부 생성 특성
- 해부학적 부위
- 병리학적 상태
- 환자 정보
- 조영제 사용 여부
- 슬라이스 두께
- 공간 해상도

### 사용 예제
```python
from medical_synthesis.medical_3d_synthesis_example import *

# 3D 뇌 MRI 생성 모델 훈련
model = train_medical_3d_synthesis(
    volume_type=VolumeType.MRI_BRAIN,
    model_type="vae",  # 또는 "diffusion"
    num_epochs=50,
    batch_size=2,
    volume_size=(64, 64, 64)
)

# 3D 볼륨 시각화
volume = generate_sample_volume(model)
visualize_3d_volume(volume, "Generated Brain MRI")
```

## 🚀 실행 방법

### 2D 의료 영상 합성 실행
```bash
cd /workspace/Vision-101
python medical_synthesis/medical_2d_synthesis_example.py
```

### 3D 의료 볼륨 합성 실행
```bash
cd /workspace/Vision-101
python medical_synthesis/medical_3d_synthesis_example.py
```

## 🛠️ 기술적 특징

### 최신 아키텍처
1. **Medical Diffusion U-Net**:
   - 의료 영상에 특화된 U-Net 구조
   - Attention 메커니즘 통합
   - 조건부 생성을 위한 embedding 레이어

2. **Medical StyleGAN**:
   - 의료 영상을 위한 Style-based GAN
   - Progressive growing 지원
   - 의료 영상 품질에 맞춘 손실 함수

3. **3D VAE-GAN**:
   - 3D 볼륨을 위한 변분 오토인코더
   - 잠재 공간에서의 3D 볼륨 조작
   - 메모리 효율적인 3D 처리

### 의료 영상 특화 손실 함수
- **Medical MSE Loss**: 기본 재구성 손실
- **Edge Preservation Loss**: 의료 영상의 경계 보존
- **Perceptual Loss**: 시각적 품질 향상
- **Anatomical Consistency Loss**: 해부학적 구조 일관성

## 📊 평가 지표

### 2D 영상 품질 평가
- **PSNR**: Peak Signal-to-Noise Ratio
- **SSIM**: Structural Similarity Index
- **Medical Specificity**: 의료 영상 특화 평가

### 3D 볼륨 품질 평가
- **3D PSNR**: 3차원 PSNR
- **Volume Similarity**: 볼륨 간 유사도
- **Anatomical Accuracy**: 해부학적 정확도

## 🎯 응용 분야

### 의료 AI 데이터 증강
- 부족한 의료 데이터 보완
- 희귀 질환 데이터 생성
- 다양한 병리학적 상태 시뮬레이션

### 의료진 교육
- 교육용 의료 영상 생성
- 다양한 케이스 스터디 자료
- 시뮬레이션 기반 학습

### 개인정보 보호
- 실제 환자 데이터 대체
- 프라이버시 보호 연구 데이터
- 익명화된 의료 영상 생성

## 🔧 설정 및 요구사항

### 하드웨어 요구사항
- **2D 모델**: GPU 4GB+ 권장
- **3D 모델**: GPU 8GB+ 권장
- RAM: 16GB+ 권장

### 소프트웨어 요구사항
```
torch >= 1.9.0
torchvision >= 0.10.0
numpy >= 1.21.0
matplotlib >= 3.4.0
scipy >= 1.7.0
PIL >= 8.0.0
```

## 📈 성능 최적화

### 메모리 최적화
- Gradient checkpointing 활용
- Mixed precision training 지원
- 배치 크기 동적 조정

### 훈련 최적화
- Progressive training 지원
- Learning rate scheduling
- Early stopping 구현

## 🚨 주의사항

1. **의료 규정 준수**: 생성된 이미지는 실제 진단 목적으로 사용하지 마세요
2. **윤리적 고려**: 의료 데이터 생성 시 윤리적 가이드라인 준수
3. **검증 필요**: 실제 응용 전 의료 전문가 검증 필수

## 📚 참고 문헌

- Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis.
- Karras, T., et al. (2019). Analyzing and improving the image quality of stylegan.
- Ronneberger, O., et al. (2015). U-net: Convolutional networks for biomedical image segmentation.
- Kingma, D. P., & Welling, M. (2013). Auto-encoding variational bayes.

## 🤝 기여하기

새로운 의료 모달리티나 개선사항이 있으시면 언제든 기여해주세요!

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

**면책조항**: 이 코드는 연구 및 교육 목적으로만 제작되었습니다. 실제 의료 진단이나 치료에 사용하지 마세요.