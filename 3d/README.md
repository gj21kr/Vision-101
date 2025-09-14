# 3D Reconstruction Algorithms

이 디렉토리는 2D 이미지로부터 3D 장면을 재구성하는 다양한 알고리즘들의 구현을 포함합니다. 전통적인 방법부터 최신 딥러닝 기반 접근법까지 포괄적으로 다룹니다.

## 📁 파일 구조

```
3d/
├── README.md                          # 이 파일
├── nerf_example.py                     # Neural Radiance Fields
├── gaussian_splatting_example.py       # 3D Gaussian Splatting
├── instant_ngp_example.py             # Instant Neural Graphics Primitives
├── pifu_example.py                    # Pixel-aligned Implicit Functions
├── structure_from_motion_example.py   # Structure from Motion
└── multi_view_stereo_example.py       # Multi-View Stereo
```

## 🎯 알고리즘 분류

### 1. Neural Rendering 방법들
최신 딥러닝 기반 접근법으로, 높은 품질의 새로운 시점 합성이 가능합니다.

#### 🌟 **NeRF (Neural Radiance Fields)** [`nerf_example.py`]
- **용도**: 고품질 새로운 시점 합성
- **특징**: 연속적 5D 장면 표현 (x,y,z,θ,φ)
- **장점**: 뛰어난 렌더링 품질, 복잡한 조명/반사 처리
- **단점**: 긴 훈련 시간, 느린 렌더링
- **적용 분야**: VR/AR, 영화 제작, 디지털 트윈

#### ⚡ **Instant-NGP** [`instant_ngp_example.py`]
- **용도**: 실시간 NeRF 훈련 및 렌더링
- **특징**: Multi-resolution hash encoding
- **장점**: 극도로 빠른 훈련 (초 단위), 실시간 렌더링
- **단점**: CUDA 의존성, hash collision 아티팩트
- **적용 분야**: 실시간 애플리케이션, 프로토타이핑

#### 🎨 **3D Gaussian Splatting** [`gaussian_splatting_example.py`]
- **용도**: 실시간 고품질 3D 렌더링
- **특징**: 3D 가우시안 primitives로 장면 표현
- **장점**: 실시간 렌더링, 높은 품질, 빠른 훈련
- **단점**: 많은 가우시안 수, 복잡한 최적화
- **적용 분야**: 게임, 실시간 시각화

### 2. Implicit Surface 방법들
3D 형태를 implicit function으로 표현하는 방법들입니다.

#### 👤 **PIFu (Pixel-aligned Implicit Functions)** [`pifu_example.py`]
- **용도**: 단일 이미지로부터 3D 인간 모델 재구성
- **특징**: 2D CNN 특성과 3D 좌표 결합
- **장점**: 단일 이미지만 필요, 복잡한 의상 처리
- **단점**: 가려진 부분 추정 한계
- **적용 분야**: 가상 피팅, 게임 캐릭터, 의료

### 3. 전통적 Multi-View 방법들
컴퓨터 비전의 고전적이면서 강건한 방법들입니다.

#### 📐 **Structure from Motion (SfM)** [`structure_from_motion_example.py`]
- **용도**: 카메라 포즈와 sparse 3D 구조 동시 추정
- **특징**: Feature matching과 Bundle Adjustment
- **장점**: 검증된 기술, 넓은 장면 처리, 보정 불필요
- **단점**: 텍스처 의존적, 긴 처리 시간
- **적용 분야**: 사진측량, 지도 제작, 3D 스캐닝

#### 🔍 **Multi-View Stereo (MVS)** [`multi_view_stereo_example.py`]
- **용도**: Dense 3D reconstruction
- **특징**: Plane sweep과 photo-consistency
- **장점**: Dense한 결과, 높은 정확도
- **단점**: 계산 비용 높음, 조명 변화 민감
- **적용 분야**: 건축 모델링, 산업 검사, 문화재

## 🚀 빠른 시작

### 환경 설정

```bash
# 필요한 패키지 설치
pip install torch torchvision opencv-python numpy matplotlib scipy scikit-learn

# 추가 도구 (선택사항)
pip install open3d trimesh plotly  # 3D 시각화용
```

### 기본 사용법

각 알고리즘은 독립적으로 실행 가능한 예제로 구성되어 있습니다:

```bash
# NeRF 예제 실행
python nerf_example.py

# Gaussian Splatting 예제 실행
python gaussian_splatting_example.py

# SfM 파이프라인 실행
python structure_from_motion_example.py
```

## 📊 성능 비교

| 알고리즘 | 훈련 시간 | 렌더링 속도 | 품질 | 메모리 사용량 | 용도 |
|---------|----------|------------|------|-------------|------|
| NeRF | 수 시간 | 느림 (초) | ⭐⭐⭐⭐⭐ | 보통 | 고품질 렌더링 |
| Instant-NGP | 수 초 | 실시간 | ⭐⭐⭐⭐ | 적음 | 실시간 앱 |
| Gaussian Splatting | 수 분 | 실시간 | ⭐⭐⭐⭐⭐ | 많음 | 실시간 고품질 |
| PIFu | 수 시간 | 빠름 | ⭐⭐⭐ | 보통 | 인간 재구성 |
| SfM | 분~시간 | N/A | ⭐⭐⭐ | 적음 | 카메라 추정 |
| MVS | 분~시간 | N/A | ⭐⭐⭐⭐ | 많음 | Dense 재구성 |

## 🎓 학습 로드맵

### 초급자 (3D 비전 입문)
1. **SfM** → 기본적인 multi-view geometry 이해
2. **MVS** → Dense reconstruction 개념 학습
3. **PIFu** → Deep learning 기반 3D 재구성 이해

### 중급자 (Neural Rendering 탐구)
1. **NeRF** → Neural rendering 기본 개념
2. **Instant-NGP** → 효율적 구현 기법
3. **Gaussian Splatting** → 최신 실시간 렌더링

### 고급자 (연구/개발)
- 각 방법의 장단점 비교 분석
- 하이브리드 방법 개발
- 특정 도메인 적용 (의료, 자율주행, 등)

## 💡 실제 적용 예시

### 1. VR/AR 애플리케이션
```python
# Gaussian Splatting for real-time rendering
model = GaussianModel(num_gaussians=100000)
trainer = GaussianSplattingTrainer(model)

# Train on multi-view images
for epoch in range(training_epochs):
    trainer.train_step(images, camera_poses)

# Real-time rendering
rendered_image = model.render(new_camera_pose)
```

### 2. 디지털 트윈 생성
```python
# SfM + MVS pipeline for industrial scanning
sfm = StructureFromMotion()
mvs = MultiViewStereo()

# Estimate camera poses
poses = sfm.run_full_pipeline(industrial_images)

# Generate dense point cloud
dense_cloud = mvs.run_mvs_pipeline(industrial_images, poses)
```

### 3. 가상 피팅 시스템
```python
# PIFu for single image 3D human reconstruction
pifu_model = PIFuModel()
occupancy = pifu_model(person_image, query_points, camera_matrix)
mesh = marching_cubes(occupancy)
```

## 🔧 고급 설정

### 커스텀 데이터셋 사용

```python
# 실제 이미지 데이터로 NeRF 훈련
class CustomDataset:
    def __init__(self, image_paths, camera_poses):
        self.images = [load_image(path) for path in image_paths]
        self.poses = camera_poses

    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx]

# COLMAP 결과 로드
poses = load_colmap_poses("colmap_output/")
dataset = CustomDataset(image_paths, poses)
```

### GPU 가속 설정

```python
# CUDA 최적화
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
```

## 📚 추가 리소스

### 논문 및 참고 자료
- **NeRF**: [Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934)
- **Instant-NGP**: [Instant Neural Graphics Primitives](https://nvlabs.github.io/instant-ngp/)
- **Gaussian Splatting**: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **PIFu**: [PIFu: Pixel-Aligned Implicit Function](https://shunsukesaito.github.io/PIFu/)

### 오픈소스 구현
- [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) - NeRF 구현 모음
- [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) - Instant-NGP 최적화 라이브러리
- [COLMAP](https://colmap.github.io/) - SfM/MVS 표준 구현
- [OpenMVG](https://openmvg.readthedocs.io/) - 오픈소스 MVG 라이브러리

### 데이터셋
- [NeRF Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) - NeRF 벤치마크
- [DTU Dataset](https://roboimagedata.compute.dtu.dk/) - MVS 벤치마크
- [RenderPeople](https://renderpeople.com/) - 인간 3D 스캔 데이터

## 🤝 기여하기

이 예제들을 개선하거나 새로운 알고리즘을 추가하고 싶다면:

1. 각 구현의 교육적 목적을 유지해주세요
2. 자세한 주석과 수학적 배경 설명 포함
3. 실행 가능한 예제 코드 제공
4. 성능보다는 이해하기 쉬운 코드 작성

## ⚠️ 주의사항

- 이 구현들은 **교육 목적**으로 작성되었습니다
- 실제 프로덕션에서는 최적화된 라이브러리 사용을 권장합니다
- GPU 메모리 사용량에 주의하세요 (특히 Gaussian Splatting)
- 일부 알고리즘은 CUDA 의존성이 있습니다

## 📞 문의사항

구현과 관련된 질문이나 개선 사항이 있으면 언제든 연락해주세요!

---

**Happy 3D Reconstruction!** 🚀✨