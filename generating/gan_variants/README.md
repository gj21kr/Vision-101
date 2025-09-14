# GAN Variants Examples

다양한 GAN (Generative Adversarial Networks) 변형 모델들의 구현 예제 모음입니다.

## 포함된 GAN 모델들

### 1. DCGAN (Deep Convolutional GAN) - `dcgan_example.py`
- **논문**: Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
- **특징**:
  - 완전 연결층을 합성곱층으로 대체
  - Batch Normalization 사용
  - LeakyReLU 활성화 함수
  - Transposed Convolution을 통한 업샘플링
  - 적절한 가중치 초기화

### 2. WGAN (Wasserstein GAN) - `wgan_example.py`
- **논문**: Wasserstein GAN
- **특징**:
  - Wasserstein 거리 사용
  - Critic 네트워크 (Sigmoid 없음)
  - Weight Clipping으로 Lipschitz 제약 조건 적용
  - RMSprop 옵티마이저
  - 여러 번의 Critic 업데이트 per Generator 업데이트

### 3. WGAN-GP (WGAN with Gradient Penalty) - `wgan_gp_example.py`
- **논문**: Improved Training of Wasserstein GANs
- **특징**:
  - Weight clipping 대신 Gradient Penalty 사용
  - 더 안정적인 훈련
  - Adam 옵티마이저 사용 가능
  - 기울기 소실 문제 해결

### 4. CycleGAN - `cyclegan_example.py`
- **논문**: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
- **특징**:
  - 페어링되지 않은 데이터셋에서 도메인 변환
  - 두 개의 생성자 (G_AB, G_BA)
  - 사이클 일관성 손실 (Cycle Consistency Loss)
  - Identity 손실 (선택적)
  - 이미지 풀 사용

### 5. StyleGAN - `stylegan_example.py`
- **논문**: A Style-Based Generator Architecture for Generative Adversarial Networks
- **특징**:
  - 매핑 네트워크 (Z → W 공간)
  - Adaptive Instance Normalization (AdaIN)
  - 스타일 믹싱 (Style Mixing)
  - 노이즈 주입 (Noise Injection)
  - Equalized Learning Rate
  - Progressive Growing (단순화됨)

### 6. Conditional GAN - `conditional_gan_example.py`
- **논문**: Conditional Generative Adversarial Nets
- **특징**:
  - 클래스 레이블을 조건으로 하는 생성
  - 레이블 임베딩
  - 제어된 이미지 생성
  - FC와 Convolutional 버전 모두 구현

## 주요 개념 비교

| 모델 | 주요 혁신 | 장점 | 단점 |
|------|----------|------|------|
| DCGAN | CNN 아키텍처 | 안정적 훈련, 고품질 이미지 | 모드 붕괴 가능 |
| WGAN | Wasserstein 거리 | 훈련 안정성, 의미있는 손실 | Weight clipping 문제 |
| WGAN-GP | Gradient Penalty | WGAN 개선, 더 나은 수렴 | 계산 복잡도 증가 |
| CycleGAN | 사이클 일관성 | 무페어링 데이터 변환 | 기하학적 변화 제한 |
| StyleGAN | 스타일 기반 생성 | 고품질, 제어 가능한 생성 | 복잡한 아키텍처 |
| Conditional GAN | 조건부 생성 | 제어된 생성 | 추가 레이블 정보 필요 |

## 훈련 팁

### 일반적인 GAN 훈련 팁
1. **배치 크기**: 보통 32-128 사이
2. **학습률**: 0.0001-0.0002 권장
3. **옵티마이저**: Adam (β1=0.5, β2=0.999)
4. **가중치 초기화**: 정규분포 (평균=0, 표준편차=0.02)

### 모델별 특화 팁
- **DCGAN**: Batch Normalization 사용, 적절한 필터 수
- **WGAN**: Critic 업데이트 5:1 비율, 낮은 학습률
- **WGAN-GP**: λ=10 gradient penalty 계수
- **CycleGAN**: λ=10 cycle consistency 가중치
- **StyleGAN**: Progressive growing, 스타일 믹싱 확률 0.9
- **Conditional GAN**: 레이블 임베딩 차원 조정

## 평가 메트릭

1. **FID (Fréchet Inception Distance)**: 이미지 품질
2. **IS (Inception Score)**: 다양성과 품질
3. **LPIPS**: 지각적 유사도
4. **SSIM**: 구조적 유사도

## 응용 분야

- **이미지 생성**: 얼굴, 자연 이미지, 예술 작품
- **이미지 변환**: 스타일 전송, 도메인 변환
- **데이터 증강**: 훈련 데이터 확장
- **이미지 편집**: 속성 조작, 인페인팅
- **의료 영상**: 합성 의료 이미지 생성

## 실행 방법

각 파일을 개별적으로 실행할 수 있습니다:

```python
# 예시: DCGAN 실행
python dcgan_example.py

# 훈련 시작하려면 파일 내 주석 해제
# generator, discriminator = train_dcgan()
```

## 요구사항

```bash
pip install torch torchvision
pip install matplotlib numpy pillow
```

## 추가 리소스

- [GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo) - 다양한 GAN 모델들
- [Papers with Code - GANs](https://paperswithcode.com/methods/category/generative-adversarial-networks)
- [PyTorch GAN 튜토리얼](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

## 주의사항

1. **GPU 메모리**: StyleGAN과 CycleGAN은 많은 메모리 필요
2. **훈련 시간**: 모델에 따라 수 시간에서 수일 소요
3. **하이퍼파라미터**: 데이터셋에 따라 조정 필요
4. **모드 붕괴**: 다양한 정규화 기법 사용 권장