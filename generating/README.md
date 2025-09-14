# Image Generation Examples

Vision-101 프로젝트의 이미지 생성 알고리즘 예제 모음입니다.

## 📖 문서 구성

- **README.md** (현재 파일): 빠른 시작 가이드 및 개요
- **[TECHNICAL_README.md](TECHNICAL_README.md)**: 상세한 기술 문서 및 수학적 배경
- **각 디렉토리의 README.md**: 특정 모델 카테고리별 상세 설명

## 포함된 알고리즘

### 1. GAN (Generative Adversarial Networks) - `gan_example.py`
- **설명**: 생성자와 판별자가 서로 경쟁하며 학습하는 생성 모델
- **특징**:
  - Generator와 Discriminator 네트워크 구현
  - MNIST 데이터셋 기반 학습
  - 실시간 손실 함수 모니터링
- **사용법**: `python gan_example.py`
- **고급 GAN 모델들**: `gan_variants/` 디렉토리에서 DCGAN, WGAN, StyleGAN 등 확인

### 2. VAE (Variational Autoencoder) - `vae_example.py`
- **설명**: 확률적 인코더-디코더 구조의 생성 모델
- **특징**:
  - 잠재 공간(Latent Space) 샘플링
  - 이미지 재구성 및 생성
  - 잠재 공간 보간(Interpolation) 기능
- **사용법**: `python vae_example.py`

### 3. Diffusion Model - `diffusion_example.py`
- **설명**: 노이즈 제거 과정을 통한 이미지 생성 모델 (DDPM)
- **특징**:
  - U-Net 기반 노이즈 예측 네트워크
  - Forward/Reverse 확산 과정
  - 타임스텝 조건부 생성
- **사용법**: `python diffusion_example.py`
- **고급 Diffusion 모델들**: `diffusion_variants/` 디렉토리에서 DDIM, Latent Diffusion, Score-based 등 확인

### 4. Style Transfer - `style_transfer_example.py`
- **설명**: 콘텐츠 이미지에 스타일 이미지의 특징을 적용하는 기법
- **특징**:
  - Neural Style Transfer (Gatys 방법)
  - Fast Style Transfer 네트워크
  - VGG19 기반 특징 추출
  - Gram Matrix를 이용한 스타일 손실
- **사용법**: `python style_transfer_example.py`

### 5. Super Resolution - `super_resolution_example.py`
- **설명**: 저해상도 이미지를 고해상도로 변환하는 기법
- **특징**:
  - SRCNN (Super-Resolution CNN)
  - ESPCN (Efficient Sub-Pixel CNN)
  - SRGAN (Super-Resolution GAN)
  - PSNR 성능 평가 메트릭
- **사용법**: `python super_resolution_example.py`

## 요구사항

```bash
pip install torch torchvision
pip install matplotlib pillow numpy
```

## 공통 기능

- 모든 예제는 GPU 가속을 지원합니다 (CUDA 사용 가능시)
- 학습 과정의 시각화 기능 제공
- 사전 훈련된 모델 활용 (Style Transfer, Super Resolution)
- 다양한 손실 함수 구현

## 실행 방법

각 파일을 개별적으로 실행할 수 있으며, 기본적으로 데모 모드로 작동합니다.
실제 학습을 위해서는 각 파일의 주석을 해제하고 실행하세요.

```python
# 예시: GAN 학습 실행
if __name__ == "__main__":
    train_gan()  # 주석 해제 후 실행
```

## 성능 최적화 팁

1. **GPU 메모리 관리**: 배치 크기를 조절하여 메모리 사용량 최적화
2. **학습률 스케줄링**: 각 모델에 맞는 학습률 스케줄러 사용
3. **데이터 증강**: 더 다양한 데이터셋으로 모델 성능 향상
4. **정규화**: Batch Normalization, Instance Normalization 등 활용

## 추가 리소스

- [PyTorch 공식 문서](https://pytorch.org/docs/)
- [Computer Vision 논문 리스트](https://paperswithcode.com/area/computer-vision)
- [GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)

## 라이센스

이 예제들은 교육 목적으로 제작되었습니다.