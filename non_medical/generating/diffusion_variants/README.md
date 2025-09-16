# Diffusion Model Variants Examples

다양한 Diffusion Model 변형들의 구현 예제 모음입니다. 각 모델은 고유한 특징과 장점을 가지며, 다양한 응용 분야에서 활용됩니다.

## 포함된 Diffusion 모델들

### 1. DDPM (Denoising Diffusion Probabilistic Models) - `ddpm_example.py`
- **논문**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **특징**:
  - 마르코프 체인 기반 확산 과정
  - U-Net 아키텍처로 노이즈 예측
  - 시간 임베딩과 어텐션 메커니즘
  - 1000 스텝의 완전한 역확산 과정
  - 높은 품질의 이미지 생성

### 2. DDIM (Denoising Diffusion Implicit Models) - `ddim_example.py`
- **논문**: Denoising Diffusion Implicit Models (Song et al., 2021)
- **특징**:
  - 결정론적 샘플링 (η=0일 때)
  - 20-250 스텝으로 빠른 샘플링
  - 동일한 노이즈 입력 → 동일한 출력
  - 잠재 공간에서의 부드러운 보간
  - DDPM과 동일한 훈련 과정

### 3. Latent Diffusion Model - `latent_diffusion_example.py`
- **논문**: High-Resolution Image Synthesis with Latent Diffusion Models (Rombach et al., 2022)
- **특징**:
  - VAE 잠재 공간에서의 확산
  - 16배 적은 계산량 (8×8 잠재 vs 32×32 이미지)
  - 인코더-디코더 아키텍처
  - 고해상도 이미지 생성 가능
  - 메모리 효율성

### 4. Conditional Diffusion - `conditional_diffusion_example.py`
- **논문**: Diffusion Models Beat GANs on Image Synthesis (Dhariwal & Nichol, 2021)
- **특징**:
  - 클래스 레이블 조건부 생성
  - Classifier-free Guidance
  - 시간과 클래스 임베딩 결합
  - 특정 클래스 이미지 생성 가능
  - 가이던스 스케일로 품질 조절

### 5. Score-based Diffusion - `score_based_diffusion_example.py`
- **논문**: Score-Based Generative Modeling through Stochastic Differential Equations (Song et al., 2021)
- **특징**:
  - SDE (확률 미분방정식) 프레임워크
  - 연속 시간 공식화
  - 다중 샘플링 알고리즘 (SDE, ODE, PC)
  - Euler-Maruyama 샘플러
  - Predictor-Corrector 샘플러

## 모델 비교표

| 모델 | 샘플링 스텝 | 샘플링 시간 | 품질 | 다양성 | 제어 가능성 |
|------|-------------|-------------|------|--------|-------------|
| DDPM | 1000 | 느림 | 매우 높음 | 높음 | 낮음 |
| DDIM | 20-250 | 빠름 | 높음 | 중간 | 높음 |
| Latent Diffusion | 50-250 | 빠름 | 높음 | 높음 | 중간 |
| Conditional | 50-250 | 빠름 | 높음 | 중간 | 매우 높음 |
| Score-based | 100-500 | 중간 | 높음 | 높음 | 중간 |

## 핵심 개념

### 1. Forward 확산 과정
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
```
- 점진적으로 노이즈를 추가
- 최종적으로 순수 가우시안 노이즈

### 2. Reverse 확산 과정
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t² I)
```
- 신경망으로 학습된 역과정
- 노이즈에서 데이터로 복원

### 3. 훈련 목표
```
L = E_t,x_0,ε [||ε - ε_θ(x_t, t)||²]
```
- 노이즈 예측 손실 함수
- 실제 노이즈와 예측 노이즈 간의 MSE

### 4. 샘플링 알고리즘
- **DDPM**: 완전한 마르코프 체인
- **DDIM**: 결정론적 ODE 해결
- **Score-based**: SDE/ODE 샘플러
- **PC**: Predictor-Corrector 결합

## 응용 분야

### 1. 이미지 생성
- 고품질 자연 이미지 생성
- 얼굴, 풍경, 예술 작품
- 무조건부/조건부 생성

### 2. 이미지 편집
- 인페인팅 (결손 부분 복원)
- 아웃페인팅 (이미지 확장)
- 스타일 변환
- 속성 편집

### 3. 의료 영상
- MRI, CT 스캔 생성
- 이미지 복원 및 향상
- 데이터 증강

### 4. 3D 생성
- NeRF와 결합한 3D 객체 생성
- 포인트 클라우드 생성
- 메쉬 생성

## 실행 방법

각 파일을 개별적으로 실행할 수 있습니다:

```bash
# DDPM 실행
python ddpm_example.py

# DDIM 빠른 샘플링
python ddim_example.py

# 잠재 공간 확산
python latent_diffusion_example.py

# 조건부 생성
python conditional_diffusion_example.py

# Score-based 모델
python score_based_diffusion_example.py
```

### 훈련 시작하기
각 파일에서 주석을 해제하고 실행:

```python
# 예시: DDPM 훈련
if __name__ == "__main__":
    model, diffusion = train_ddpm()  # 주석 해제
    samples = diffusion.sample(model, n=16)
```

## 하이퍼파라미터 가이드

### 일반적인 설정
- **배치 크기**: 16-128 (GPU 메모리에 따라)
- **학습률**: 1e-4 ~ 3e-4
- **옵티마이저**: AdamW (weight decay 포함)
- **노이즈 스케줄**: Linear (β: 1e-4 ~ 0.02)

### 모델별 특화 설정

**DDPM**:
- 노이즈 스텝: 1000
- U-Net 채널: 128, 256, 512, 512
- 어텐션: 16×16, 8×8 해상도

**DDIM**:
- η: 0.0 (결정론적) ~ 1.0 (확률적)
- 샘플링 스텝: 20-250
- 동일한 네트워크 아키텍처

**Latent Diffusion**:
- 잠재 채널: 4
- VAE 압축 비율: 8×
- U-Net 채널: 320, 640, 1280

**Conditional**:
- 클래스 임베딩 차원: 512
- 가이던스 스케일: 3.0-15.0
- Null 클래스 확률: 0.1

**Score-based**:
- σ_min: 0.01, σ_max: 50.0
- 기하학적 노이즈 스케줄
- Predictor-Corrector 스텝: 5

## 성능 최적화

### 1. 메모리 최적화
- Gradient checkpointing
- Mixed precision training (fp16)
- 배치 크기 조정

### 2. 속도 최적화
- DDIM 빠른 샘플링
- 잠재 공간 확산
- 병렬 샘플링

### 3. 품질 향상
- Classifier-free guidance
- 더 많은 어텐션 레이어
- 더 큰 모델 크기

## 평가 메트릭

- **FID (Fréchet Inception Distance)**: 생성 품질
- **IS (Inception Score)**: 다양성과 선명도
- **LPIPS**: 지각적 유사도
- **CLIP Score**: 텍스트-이미지 일치도

## 요구사항

```bash
pip install torch torchvision
pip install matplotlib numpy tqdm
pip install scipy  # Score-based 모델용
```

## 주요 참고 문헌

1. **DDPM**: Ho, J. et al. "Denoising Diffusion Probabilistic Models" (2020)
2. **DDIM**: Song, J. et al. "Denoising Diffusion Implicit Models" (2021)
3. **Latent Diffusion**: Rombach, R. et al. "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)
4. **Score-based**: Song, Y. et al. "Score-Based Generative Modeling through Stochastic Differential Equations" (2021)
5. **Classifier-free**: Ho, J. & Salimans, T. "Classifier-Free Diffusion Guidance" (2022)

## 추가 리소스

- [Papers with Code - Diffusion](https://paperswithcode.com/methods/category/diffusion-models)
- [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
- [Yang Song's Blog on Score-based Models](https://yang-song.github.io/blog/2021/score/)

## 주의사항

⚠️ **계산 복잡도**: Diffusion 모델은 매우 계산집약적입니다
- GPU 메모리: 최소 8GB 권장
- 훈련 시간: 수 시간에서 수일
- 샘플링 시간: DDPM > Score-based > Latent > DDIM

🎯 **실용적 권장사항**:
- 빠른 프로토타이핑: DDIM 또는 Latent Diffusion
- 최고 품질: DDPM 또는 Score-based
- 제어된 생성: Conditional Diffusion
- 고해상도: Latent Diffusion