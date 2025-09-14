"""
NeRF (Neural Radiance Fields) Implementation

NeRF는 Mildenhall et al. (2020)에서 제안된 혁신적인 3D 장면 표현 방법으로,
2D 이미지들로부터 고품질 3D 장면을 학습하고 새로운 시점에서 렌더링할 수 있습니다.

핵심 개념:

1. **Neural Scene Representation**:
   - 3D 장면을 MLP(Multi-Layer Perceptron)로 표현
   - 입력: 3D 위치 (x,y,z)와 시점 방향 (θ,φ)
   - 출력: 색상 (RGB)와 밀도 (σ)

2. **Volume Rendering**:
   - Ray casting을 통한 픽셀별 색상 계산
   - C(r) = ∫ T(t)σ(r(t))c(r(t),d)dt
   - T(t) = exp(-∫σ(r(s))ds): 투과율 (transparency)

3. **Positional Encoding**:
   - 고주파 디테일 학습을 위한 주파수 인코딩
   - PE(p) = [sin(2^0πp), cos(2^0πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]

4. **Hierarchical Sampling**:
   - Coarse network와 Fine network의 2단계 샘플링
   - 중요한 영역에 더 많은 샘플 할당으로 효율성 향상

수학적 원리:
- Volume Rendering Equation을 미분 가능한 형태로 구현
- Neural network를 통한 연속적 장면 표현
- Differentiable rendering으로 end-to-end 학습

장점:
- 고품질 새로운 시점 합성 (Novel View Synthesis)
- 연속적 장면 표현으로 임의 해상도 렌더링 가능
- 기하학적 일관성과 시점 dependent 효과 모두 처리

단점:
- 긴 학습 시간 (장면당 수 시간)
- 느린 렌더링 속도
- 장면별 개별 학습 필요

Reference:
- Mildenhall, B., et al. (2020).
  "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis."
  European Conference on Computer Vision (ECCV).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

class PositionalEncoder(nn.Module):
    def __init__(self, d_input=3, n_freqs=10, log_space=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)

        # Create frequency bands
        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs-1, self.n_freqs)
        else:
            freq_bands = torch.linspace(1., 2.**(self.n_freqs-1), self.n_freqs)

        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        """
        Positional Encoding: 고주파 디테일 학습을 위한 주파수 변환

        NeRF의 핵심 혁신 중 하나:
        - 일반 MLP는 저주파 패턴만 잘 학습 (spectral bias)
        - 위치를 여러 주파수로 인코딩하여 고주파 디테일 학습 가능
        - 인간의 푸리에 변환과 유사한 개념

        Args:
            x (torch.Tensor): 입력 좌표 [batch, d_input]

        Returns:
            torch.Tensor: 인코딩된 좌표 [batch, d_output]

        수학적 정의:
            PE(p) = [p, sin(2^0πp), cos(2^0πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]

        효과:
        - 세밀한 텍스처와 기하학적 디테일 표현 가능
        - Network의 표현 능력을 극적으로 향상
        - 다양한 스케일의 패턴 동시 학습
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        # Original coordinates
        out = [x]

        # Encode with different frequencies
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                out.append(func(freq * x))

        return torch.cat(out, dim=-1)

class NeRF(nn.Module):
    def __init__(self, d_pos=3, d_dir=3, n_layers=8, d_filter=256, skip=4,
                 use_viewdirs=True, pos_freqs=10, dir_freqs=4):
        super().__init__()

        self.d_pos = d_pos
        self.d_dir = d_dir
        self.skip = skip
        self.use_viewdirs = use_viewdirs

        # Positional encoders
        self.pos_encoder = PositionalEncoder(d_pos, pos_freqs, log_space=True)
        self.dir_encoder = PositionalEncoder(d_dir, dir_freqs, log_space=True) if use_viewdirs else None

        # Dimensions after encoding
        d_pos_encoded = self.pos_encoder.d_output
        d_dir_encoded = self.dir_encoder.d_output if use_viewdirs else 0

        # Position network (for density and features)
        layers = []
        for i in range(n_layers):
            if i == 0:
                layer = nn.Linear(d_pos_encoded, d_filter)
            elif i == skip:
                layer = nn.Linear(d_filter + d_pos_encoded, d_filter)
            else:
                layer = nn.Linear(d_filter, d_filter)
            layers.append(layer)

        self.pos_layers = nn.ModuleList(layers)

        # Density head
        self.density_head = nn.Linear(d_filter, 1)

        # Feature layer
        self.feature_layer = nn.Linear(d_filter, d_filter)

        # Color network (view-dependent)
        if use_viewdirs:
            self.color_layer1 = nn.Linear(d_filter + d_dir_encoded, d_filter // 2)
            self.color_layer2 = nn.Linear(d_filter // 2, 3)
        else:
            self.color_head = nn.Linear(d_filter, 3)

    def forward(self, pos, dirs=None):
        """
        NeRF Network Forward Pass: 3D 위치와 시점에서 색상과 밀도 예측

        NeRF의 핵심 아이디어:
        - 3D 공간의 각 점을 (x,y,z,θ,φ) → (r,g,b,σ)로 매핑
        - σ(밀도): 해당 위치에 물체가 존재할 확률
        - c(색상): 해당 방향에서 보이는 색상 (view-dependent)

        Args:
            pos (torch.Tensor): 3D 위치 좌표 [batch, 3]
            dirs (torch.Tensor): 시점 방향 [batch, 3] (optional)

        Returns:
            tuple: (colors, densities)
                - colors: RGB 색상 [batch, 3]
                - densities: 밀도 값 [batch, 1]

        Network 구조:
        1. Position encoding으로 고주파 정보 보존
        2. 8층 MLP로 위치 특성 학습
        3. Skip connection (4번째 층)으로 gradient flow 개선
        4. 밀도는 위치에만 의존 (기하학적 일관성)
        5. 색상은 위치+방향 의존 (시점별 변화, 반사 등)
        """
        # Encode positions
        pos_encoded = self.pos_encoder(pos)

        # Position network with skip connection
        x = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            if i == self.skip:
                x = torch.cat([x, pos_encoded], dim=-1)
            x = F.relu(layer(x))

        # Density (view-independent)
        density = F.relu(self.density_head(x))

        # Feature extraction
        feature = self.feature_layer(x)

        # Color (view-dependent or independent)
        if self.use_viewdirs and dirs is not None:
            dirs_encoded = self.dir_encoder(dirs)
            color_input = torch.cat([feature, dirs_encoded], dim=-1)
            color = F.relu(self.color_layer1(color_input))
            color = torch.sigmoid(self.color_layer2(color))
        else:
            color = torch.sigmoid(self.color_head(feature))

        return color, density

def get_rays(height, width, focal_length, pose):
    """
    Ray generation: 카메라 파라미터로부터 광선 생성

    컴퓨터 비전의 기본 개념:
    - 각 픽셀은 3D 공간으로의 광선(ray)을 나타냄
    - Ray: origin + t * direction (t는 깊이)
    - 카메라 좌표계에서 월드 좌표계로 변환

    Args:
        height, width: 이미지 크기
        focal_length: 카메라 초점거리
        pose: 카메라 pose matrix [4, 4]

    Returns:
        tuple: (ray_origins, ray_directions)
            - ray_origins: 광선 시작점 [H, W, 3]
            - ray_directions: 광선 방향 [H, W, 3]

    수학적 배경:
    - Pinhole camera model 사용
    - 픽셀 (i,j) → 카메라 좌표 → 월드 좌표 변환
    - 모든 광선은 카메라 중심에서 시작
    """
    # Create pixel coordinates
    i, j = torch.meshgrid(torch.linspace(0, width-1, width),
                         torch.linspace(0, height-1, height))
    i, j = i.t(), j.t()

    # Convert to camera coordinates
    dirs = torch.stack([(i - width * 0.5) / focal_length,
                       -(j - height * 0.5) / focal_length,
                       -torch.ones_like(i)], dim=-1)

    # Transform to world coordinates
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
    rays_o = pose[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def volume_render(colors, densities, dists):
    """
    Volume Rendering: 광선을 따라 색상을 적분하여 최종 픽셀 색상 계산

    NeRF의 핵심 렌더링 방정식:
    C(r) = ∫[t_near to t_far] T(t) * σ(r(t)) * c(r(t), d) dt

    여기서:
    - T(t) = exp(-∫[t_near to t] σ(r(s)) ds): 투과율 (transparency)
    - σ(r(t)): 위치 r(t)에서의 밀도
    - c(r(t), d): 위치 r(t)와 방향 d에서의 색상

    Args:
        colors (torch.Tensor): 샘플점들의 색상 [batch, n_samples, 3]
        densities (torch.Tensor): 샘플점들의 밀도 [batch, n_samples, 1]
        dists (torch.Tensor): 샘플점들 간의 거리 [batch, n_samples]

    Returns:
        tuple: (rgb, depth, acc, weights)
            - rgb: 렌더링된 색상 [batch, 3]
            - depth: 예상 깊이 [batch, 1]
            - acc: 투명도 누적 [batch, 1]
            - weights: 각 샘플의 기여도 [batch, n_samples]

    물리적 의미:
    - 광선이 물질을 통과하며 흡수되고 산란됨
    - 각 점에서의 기여도는 그 점까지의 투과율과 해당 점의 밀도에 비례
    - 최종 색상은 모든 점의 가중 평균
    """
    # Calculate transparency weights
    alpha = 1 - torch.exp(-densities * dists.unsqueeze(-1))

    # Calculate transmittance T(t)
    transmittance = torch.cumprod(torch.cat([
        torch.ones_like(alpha[..., :1]),  # T(t_0) = 1
        1 - alpha + 1e-10                # Avoid log(0)
    ], dim=-1), dim=-1)[..., :-1]

    # Calculate weights w_i = T_i * α_i
    weights = transmittance * alpha

    # Render color
    rgb = torch.sum(weights * colors, dim=-2)

    # Calculate expected depth
    depth = torch.sum(weights[..., 0] * dists, dim=-1, keepdim=True)

    # Calculate accumulated transmittance (opacity)
    acc = torch.sum(weights[..., 0], dim=-1, keepdim=True)

    return rgb, depth, acc, weights.squeeze(-1)

class NeRFTrainer:
    def __init__(self, model, optimizer, device='cuda'):
        """
        NeRF 훈련 관리 클래스

        NeRF 훈련의 특징:
        - 장면별 개별 훈련 (scene-specific)
        - 다양한 시점의 이미지들 필요
        - 매우 긴 훈련 시간 (수십만 iteration)
        - Hierarchical sampling으로 효율성 향상
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)

    def train_step(self, rays_o, rays_d, target_colors, near=0.0, far=1.0, n_samples=64):
        """
        Single training step: 한 배치의 광선들에 대해 훈련

        Args:
            rays_o: 광선 원점 [batch, 3]
            rays_d: 광선 방향 [batch, 3]
            target_colors: 실제 픽셀 색상 [batch, 3]
            near, far: 렌더링 범위
            n_samples: 광선당 샘플 수

        Returns:
            dict: 손실 및 메트릭 정보
        """
        self.optimizer.zero_grad()

        # Sample points along rays
        t_vals = torch.linspace(near, far, n_samples, device=self.device)
        t_vals = t_vals.expand(rays_o.shape[0], n_samples)

        # Add noise for regularization
        if self.model.training:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand

        # Calculate sample points
        pts = rays_o.unsqueeze(1) + t_vals.unsqueeze(-1) * rays_d.unsqueeze(1)

        # Get colors and densities
        dirs = rays_d.unsqueeze(1).expand_as(pts)
        colors, densities = self.model(pts.reshape(-1, 3), dirs.reshape(-1, 3))
        colors = colors.reshape(*pts.shape)
        densities = densities.reshape(pts.shape[:-1] + (1,))

        # Calculate distances between samples
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        # Volume rendering
        rgb, depth, acc, weights = volume_render(colors, densities, dists)

        # Loss calculation
        loss = F.mse_loss(rgb, target_colors)

        loss.backward()
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'rgb': rgb.detach(),
            'depth': depth.detach(),
            'acc': acc.detach()
        }

def hierarchical_sampling(weights, t_vals, n_samples):
    """
    Hierarchical Sampling: 중요도 기반 샘플링으로 효율성 향상

    NeRF의 2단계 샘플링 전략:
    1. Coarse network: 균등 샘플링으로 대략적 구조 파악
    2. Fine network: 중요한 영역(높은 weight)에 추가 샘플링

    이 방법의 장점:
    - 빈 공간에 불필요한 샘플링 방지
    - 물체 표면 근처에 더 많은 샘플 집중
    - 전체 샘플 수 대비 높은 품질 달성

    Args:
        weights: Coarse network의 weight 분포 [batch, n_coarse]
        t_vals: Coarse network의 t 값들 [batch, n_coarse]
        n_samples: Fine network용 추가 샘플 수

    Returns:
        torch.Tensor: Fine network용 t 값들 [batch, n_coarse + n_samples]
    """
    # PDF로 변환 (정규화)
    weights = weights + 1e-5  # Prevent zero weights
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # Uniform samples
    u = torch.rand(weights.shape[0], n_samples, device=weights.device)

    # Inverse transform sampling
    indices = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
    above = torch.clamp(indices, 0, cdf.shape[-1] - 1)

    # Linear interpolation
    indices_g = torch.stack([below, above], dim=-1)
    matched_shape = list(indices_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), -1, indices_g)
    bins_g = torch.gather(t_vals.unsqueeze(-2).expand(matched_shape), -1, indices_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def render_image(model, height, width, focal_length, pose, device='cuda', chunk=1024):
    """
    Complete image rendering: 전체 이미지 렌더링

    메모리 효율성을 위해 chunk 단위로 처리:
    - 전체 이미지를 한번에 처리하면 GPU 메모리 부족
    - 작은 배치들로 나누어 순차 처리
    - 결과를 합쳐서 최종 이미지 생성
    """
    model.eval()
    rays_o, rays_d = get_rays(height, width, focal_length, pose)
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)

    all_rgb = []

    with torch.no_grad():
        for i in range(0, rays_o.shape[0], chunk):
            rays_o_chunk = rays_o[i:i+chunk]
            rays_d_chunk = rays_d[i:i+chunk]

            # Sample points along rays
            t_vals = torch.linspace(0.0, 1.0, 64, device=device)
            t_vals = t_vals.expand(rays_o_chunk.shape[0], 64)

            pts = rays_o_chunk.unsqueeze(1) + t_vals.unsqueeze(-1) * rays_d_chunk.unsqueeze(1)
            dirs = rays_d_chunk.unsqueeze(1).expand_as(pts)

            # Forward pass
            colors, densities = model(pts.reshape(-1, 3), dirs.reshape(-1, 3))
            colors = colors.reshape(*pts.shape)
            densities = densities.reshape(pts.shape[:-1] + (1,))

            # Volume rendering
            dists = torch.full((rays_o_chunk.shape[0], 64), 1.0/64, device=device)
            rgb, _, _, _ = volume_render(colors, densities, dists)

            all_rgb.append(rgb.cpu())

    rgb_map = torch.cat(all_rgb, dim=0).reshape(height, width, 3)
    return rgb_map

# Example usage and dataset
class SyntheticDataset(Dataset):
    """
    합성 데이터셋 예제

    실제 NeRF 훈련에는 다음이 필요:
    - 다양한 시점의 RGB 이미지들
    - 각 이미지의 카메라 포즈 (외부 파라미터)
    - 카메라 내부 파라미터 (초점거리 등)
    - 옵션: 깊이 정보, 세그멘테이션 마스크
    """
    def __init__(self, n_views=100, height=64, width=64):
        self.n_views = n_views
        self.height = height
        self.width = width
        self.focal_length = 50.0

        # Generate random camera poses (simplified)
        self.poses = self.generate_poses()

    def generate_poses(self):
        poses = []
        for i in range(self.n_views):
            # Simple circular camera trajectory
            angle = 2 * np.pi * i / self.n_views
            x = 2 * np.cos(angle)
            z = 2 * np.sin(angle)
            y = 0.5

            # Look at origin
            pose = np.eye(4)
            pose[:3, 3] = [x, y, z]  # Translation
            # Simplified rotation (실제로는 더 복잡한 계산 필요)

            poses.append(torch.FloatTensor(pose))
        return poses

    def __len__(self):
        return self.n_views

    def __getitem__(self, idx):
        pose = self.poses[idx]
        # 실제로는 여기서 해당 pose의 실제 이미지를 로드
        # 지금은 더미 데이터 반환
        target_img = torch.rand(self.height, self.width, 3)
        return pose, target_img

if __name__ == "__main__":
    print("NeRF Example - Neural Radiance Fields")
    print("This implementation demonstrates:")
    print("- Positional encoding for high-frequency details")
    print("- Volume rendering with differentiable integration")
    print("- Hierarchical sampling for efficiency")
    print("- View-dependent color prediction")

    # Example model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeRF(d_pos=3, d_dir=3, n_layers=8, d_filter=256)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Note: This is a simplified implementation.")
    print("Real NeRF training requires:")
    print("- Multi-view dataset with camera poses")
    print("- Coarse-to-fine hierarchical sampling")
    print("- Long training time (hours to days)")
    print("- Proper camera pose estimation (COLMAP, etc.)")