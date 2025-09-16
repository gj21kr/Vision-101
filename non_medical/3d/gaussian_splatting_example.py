"""
3D Gaussian Splatting Implementation

3D Gaussian Splatting은 Kerbl et al. (2023)에서 제안된 실시간 고품질 3D 렌더링 방법으로,
NeRF의 속도 문제를 해결하면서도 높은 품질을 유지하는 혁신적인 접근법입니다.

핵심 개념:

1. **3D Gaussian Primitives**:
   - 3D 장면을 수백만 개의 3D 가우시안으로 표현
   - 각 가우시안: 위치(μ), 공분산(Σ), 색상(c), 투명도(α)
   - G(x) = exp(-½(x-μ)ᵀΣ⁻¹(x-μ))

2. **Differentiable Splatting**:
   - 3D 가우시안을 2D 이미지 평면에 투영
   - EWA (Elliptical Weighted Average) splatting 사용
   - 전체 파이프라인이 미분 가능하여 end-to-end 학습

3. **Adaptive Density Control**:
   - 훈련 중 가우시안 추가/제거/분할/병합
   - 필요한 영역에 더 높은 밀도로 가우시안 배치
   - 불필요한 가우시안 제거로 효율성 향상

4. **Real-time Rendering**:
   - GPU 최적화된 tile-based rasterization
   - 실시간 렌더링 가능 (>30 FPS)
   - 메모리 효율적인 구현

수학적 원리:

투영 방정식:
- 3D 공분산 Σ를 2D로 투영: Σ' = JΣJᵀ
- 여기서 J는 투영 변환의 야코비안

렌더링 방정식:
- C = Σᵢ cᵢαᵢ ∏_{j<i}(1-αⱼ)
- 깊이 순서대로 alpha blending

장점:
- 실시간 렌더링 속도 (vs NeRF의 수 분)
- 고품질 결과 (NeRF와 비슷하거나 더 좋음)
- 메모리 효율성
- 빠른 훈련 시간 (NeRF 대비 10-100배 빠름)

단점:
- 복잡한 최적화 과정
- 가우시안 수가 매우 많을 수 있음
- 일부 geometric details 손실 가능

Reference:
- Kerbl, B., et al. (2023).
  "3D Gaussian Splatting for Real-Time Radiance Field Rendering."
  ACM Transactions on Graphics (SIGGRAPH).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import math

class GaussianModel(nn.Module):
    def __init__(self, num_gaussians=100000):
        """
        3D Gaussian Splatting 모델

        각 가우시안의 매개변수:
        - positions: 3D 위치 μ ∈ ℝ³
        - scales: 스케일 s ∈ ℝ³ (각 축별 크기)
        - rotations: 회전 q ∈ ℝ⁴ (quaternion)
        - colors: SH coefficients (구면 조화 함수)
        - opacity: 투명도 α ∈ [0,1]

        Args:
            num_gaussians: 초기 가우시안 개수
        """
        super().__init__()
        self.num_gaussians = num_gaussians

        # Gaussian parameters (learnable)
        self.positions = nn.Parameter(torch.randn(num_gaussians, 3) * 0.1)
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3) * 0.01)
        self.rotations = nn.Parameter(torch.randn(num_gaussians, 4))
        self.colors = nn.Parameter(torch.randn(num_gaussians, 3))
        self.opacity = nn.Parameter(torch.ones(num_gaussians, 1) * 0.1)

        # Spherical harmonics degree (for view-dependent colors)
        self.sh_degree = 3
        self.colors_sh = nn.Parameter(torch.randn(num_gaussians, (self.sh_degree + 1) ** 2, 3) * 0.001)

    def get_covariance_matrix(self, scales, rotations):
        """
        3D 공분산 행렬 계산: 스케일과 회전으로부터 공분산 구성

        3D 가우시안의 수학적 표현:
        G(x) = exp(-½(x-μ)ᵀΣ⁻¹(x-μ))

        공분산 행렬 구성:
        Σ = R S Sᵀ Rᵀ
        - R: 회전 행렬 (quaternion으로부터)
        - S: 스케일 행렬 (diagonal)

        Args:
            scales: 스케일 매개변수 [N, 3]
            rotations: quaternion 회전 [N, 4]

        Returns:
            torch.Tensor: 공분산 행렬들 [N, 3, 3]

        중요한 점:
        - 각 가우시안마다 다른 모양과 방향 가능
        - 타원체 형태로 다양한 geometry 표현
        - 미분 가능한 연산으로 end-to-end 학습
        """
        # Normalize quaternions
        rotations = F.normalize(rotations, dim=-1)

        # Convert quaternion to rotation matrix
        w, x, y, z = rotations.unbind(-1)
        rotation_matrix = torch.stack([
            torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1)
        ], dim=-2)

        # Create scale matrix
        scale_matrix = torch.diag_embed(torch.clamp(scales, min=1e-7))

        # Compute covariance: Σ = R S Sᵀ Rᵀ
        covariance = torch.bmm(torch.bmm(rotation_matrix, scale_matrix**2),
                              rotation_matrix.transpose(-1, -2))

        return covariance

    def project_to_2d(self, positions, covariances, camera_matrix):
        """
        3D 가우시안을 2D 이미지 평면에 투영

        투영 과정:
        1. 3D 가우시안 → 2D 투영 (perspective projection)
        2. 3D 공분산 → 2D 공분산 (야코비안 활용)
        3. EWA splatting으로 픽셀 기여도 계산

        Args:
            positions: 3D 위치 [N, 3]
            covariances: 3D 공분산 [N, 3, 3]
            camera_matrix: 카메라 투영 행렬 [3, 4]

        Returns:
            tuple: (2D positions, 2D covariances, depths)

        수학적 배경:
        - 투영 변환의 야코비안 J를 통해 공분산 변환
        - Σ₂D = J Σ₃D Jᵀ
        - Perspective projection의 비선형성 고려
        """
        # Homogeneous coordinates
        positions_homo = torch.cat([positions, torch.ones_like(positions[:, :1])], dim=-1)

        # Project to 2D
        projected = torch.matmul(camera_matrix, positions_homo.T).T
        depths = projected[:, 2:3]
        positions_2d = projected[:, :2] / (depths + 1e-7)

        # Compute Jacobian of projection
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        # Simplified Jacobian (assuming small translations)
        jacobian = torch.zeros(positions.shape[0], 2, 3, device=positions.device)
        jacobian[:, 0, 0] = fx / depths.squeeze()
        jacobian[:, 1, 1] = fy / depths.squeeze()
        jacobian[:, 0, 2] = -fx * positions_2d[:, 0] / depths.squeeze()
        jacobian[:, 1, 2] = -fy * positions_2d[:, 1] / depths.squeeze()

        # Project covariance to 2D: Σ₂D = J Σ₃D Jᵀ
        covariances_2d = torch.bmm(torch.bmm(jacobian, covariances), jacobian.transpose(-1, -2))

        return positions_2d, covariances_2d, depths

    def spherical_harmonics(self, directions):
        """
        구면 조화 함수를 이용한 view-dependent 색상 계산

        구면 조화 함수 (Spherical Harmonics):
        - 구면 위의 함수를 푸리에 급수처럼 표현
        - 조명과 반사를 효율적으로 모델링
        - 각 방향별로 다른 색상 표현 가능

        Args:
            directions: 시점 방향 벡터 [N, 3]

        Returns:
            torch.Tensor: SH 기저 함수 값들 [N, num_sh_bases]

        수학적 배경:
        Y₀⁰ = 1/2 √(1/π)
        Y₁⁻¹ = 1/2 √(3/π) y
        Y₁⁰ = 1/2 √(3/π) z
        Y₁¹ = 1/2 √(3/π) x
        ... (고차 항들)

        장점:
        - 효율적인 계산
        - 부드러운 view-dependent 효과
        - 적은 매개변수로 복잡한 조명 표현
        """
        x, y, z = directions.unbind(-1)

        # SH basis functions (up to degree 3)
        sh_0_0 = torch.ones_like(x) * 0.28209479177387814  # 1/2 * sqrt(1/π)

        sh_1_neg1 = -0.48860251190291987 * y  # -1/2 * sqrt(3/π) * y
        sh_1_0 = 0.48860251190291987 * z      # 1/2 * sqrt(3/π) * z
        sh_1_pos1 = -0.48860251190291987 * x  # -1/2 * sqrt(3/π) * x

        sh_2_neg2 = 1.0925484305920792 * x * y
        sh_2_neg1 = -1.0925484305920792 * y * z
        sh_2_0 = 0.31539156525252005 * (2 * z**2 - x**2 - y**2)
        sh_2_pos1 = -1.0925484305920792 * x * z
        sh_2_pos2 = 0.5462742152960396 * (x**2 - y**2)

        # Higher order terms would continue here...

        sh_bases = torch.stack([
            sh_0_0,
            sh_1_neg1, sh_1_0, sh_1_pos1,
            sh_2_neg2, sh_2_neg1, sh_2_0, sh_2_pos1, sh_2_pos2
        ], dim=-1)

        return sh_bases

    def forward(self, camera_matrix, camera_position, image_height, image_width):
        """
        전체 3D Gaussian Splatting 파이프라인

        과정:
        1. 3D 가우시안 매개변수로부터 공분산 계산
        2. 2D 이미지 평면으로 투영
        3. View-dependent 색상 계산 (SH)
        4. 2D splatting으로 이미지 렌더링

        Args:
            camera_matrix: 카메라 내부 파라미터 [3, 4]
            camera_position: 카메라 위치 [3]
            image_height, image_width: 출력 이미지 크기

        Returns:
            torch.Tensor: 렌더링된 이미지 [height, width, 3]
        """
        # Get 3D covariance matrices
        covariances_3d = self.get_covariance_matrix(
            torch.exp(self.scales), F.normalize(self.rotations, dim=-1))

        # Project to 2D
        positions_2d, covariances_2d, depths = self.project_to_2d(
            self.positions, covariances_3d, camera_matrix)

        # Calculate view directions for SH
        view_dirs = F.normalize(self.positions - camera_position.unsqueeze(0), dim=-1)

        # Get SH basis values
        sh_bases = self.spherical_harmonics(view_dirs)

        # Calculate view-dependent colors
        colors = self.colors.clone()
        for i in range(min(sh_bases.shape[-1], self.colors_sh.shape[1])):
            colors += self.colors_sh[:, i] * sh_bases[:, i:i+1]

        colors = torch.sigmoid(colors)  # Ensure [0,1] range

        # Perform 2D splatting (simplified version)
        rendered_image = self.render_2d_splats(
            positions_2d, covariances_2d, colors,
            torch.sigmoid(self.opacity), depths,
            image_height, image_width)

        return rendered_image

    def render_2d_splats(self, positions_2d, covariances_2d, colors, opacity, depths,
                        image_height, image_width):
        """
        2D Gaussian Splatting 렌더링

        각 픽셀에서의 색상 계산:
        C(x,y) = Σᵢ αᵢ cᵢ exp(-½(p-μᵢ)ᵀ Σᵢ⁻¹ (p-μᵢ)) ∏_{j<i}(1-αⱼ)

        Args:
            positions_2d: 2D 가우시안 중심 [N, 2]
            covariances_2d: 2D 공분산 [N, 2, 2]
            colors: 색상 [N, 3]
            opacity: 투명도 [N, 1]
            depths: 깊이 (정렬용) [N, 1]
            image_height, image_width: 이미지 크기

        Returns:
            torch.Tensor: 렌더링된 이미지 [height, width, 3]

        실제 구현에서는:
        - GPU tile-based rasterization 사용
        - 메모리 효율적인 구현 필요
        - 깊이 정렬 최적화
        """
        device = positions_2d.device

        # Create pixel grid
        y_coords, x_coords = torch.meshgrid(
            torch.arange(image_height, device=device, dtype=torch.float),
            torch.arange(image_width, device=device, dtype=torch.float),
            indexing='ij'
        )
        pixels = torch.stack([x_coords, y_coords], dim=-1).reshape(-1, 2)

        # Sort gaussians by depth (back-to-front)
        sorted_indices = torch.argsort(depths.squeeze(), descending=True)

        positions_2d = positions_2d[sorted_indices]
        covariances_2d = covariances_2d[sorted_indices]
        colors = colors[sorted_indices]
        opacity = opacity[sorted_indices]

        # Initialize output
        rendered_color = torch.zeros(image_height * image_width, 3, device=device)
        accumulated_alpha = torch.zeros(image_height * image_width, 1, device=device)

        # Render each gaussian (simplified - in practice use efficient GPU kernels)
        for i in range(min(1000, len(positions_2d))):  # Limit for demo
            # Calculate Gaussian weights for all pixels
            diff = pixels - positions_2d[i]  # [H*W, 2]

            # Compute inv(covariance) * diff
            try:
                cov_inv = torch.inverse(covariances_2d[i] + torch.eye(2, device=device) * 1e-6)
                weights = torch.exp(-0.5 * torch.sum(diff @ cov_inv * diff, dim=-1))
            except:
                continue  # Skip invalid covariances

            # Apply opacity
            alpha = opacity[i] * weights
            alpha = alpha.unsqueeze(-1)  # [H*W, 1]

            # Alpha blending
            contribution = (1 - accumulated_alpha) * alpha
            rendered_color += contribution * colors[i]
            accumulated_alpha += contribution

            # Early termination if fully opaque
            if torch.all(accumulated_alpha > 0.99):
                break

        return rendered_color.reshape(image_height, image_width, 3)

class GaussianSplattingTrainer:
    def __init__(self, model, device='cuda'):
        """
        3D Gaussian Splatting 훈련 관리자

        훈련 과정의 특징:
        - Adaptive density control (가우시안 추가/제거)
        - 다양한 loss 함수 (L1, SSIM, perceptual)
        - Regularization (가우시안 크기, 투명도 등)
        """
        self.model = model
        self.device = device
        self.model.to(device)

        # Optimizers with different learning rates
        self.optimizer_positions = torch.optim.Adam([model.positions], lr=1.6e-4)
        self.optimizer_colors = torch.optim.Adam([model.colors, model.colors_sh], lr=2.5e-3)
        self.optimizer_opacity = torch.optim.Adam([model.opacity], lr=5e-2)
        self.optimizer_scales = torch.optim.Adam([model.scales], lr=5e-3)
        self.optimizer_rotations = torch.optim.Adam([model.rotations], lr=1e-3)

    def densification_and_pruning(self, iteration):
        """
        적응적 밀도 제어: 가우시안 추가/제거/분할

        전략:
        1. 큰 가우시안 → 분할 (높은 gradient 영역)
        2. 작은 가우시안 → 복제 (under-reconstruction 영역)
        3. 투명한 가우시안 → 제거 (기여도 낮음)
        4. 화면 밖 가우시안 → 제거 (불필요)

        이 과정이 3D GS의 핵심 혁신:
        - 자동으로 최적의 가우시안 분포 찾기
        - 효율성과 품질 사이의 균형
        - 장면 복잡도에 적응적 대응
        """
        if iteration % 100 == 0 and iteration > 500:
            with torch.no_grad():
                # Calculate gradients (simplified)
                grad_threshold = 0.0002

                # Find gaussians with high gradients
                if hasattr(self.model.positions, 'grad') and self.model.positions.grad is not None:
                    grad_norm = torch.norm(self.model.positions.grad, dim=-1)
                    high_grad_mask = grad_norm > grad_threshold

                    # Split large gaussians with high gradients
                    large_scale_mask = torch.max(self.model.scales.data, dim=-1)[0] > 0.01
                    split_mask = high_grad_mask & large_scale_mask

                    if split_mask.sum() > 0:
                        self.split_gaussians(split_mask)

                # Remove transparent gaussians
                transparent_mask = self.model.opacity.data.squeeze() < 0.01
                if transparent_mask.sum() > 0:
                    self.prune_gaussians(transparent_mask)

    def split_gaussians(self, mask):
        """가우시안 분할: 큰 가우시안을 두 개로 나누기"""
        n_split = mask.sum().item()
        if n_split == 0:
            return

        # Create new gaussian parameters
        new_positions = self.model.positions.data[mask] + torch.randn_like(self.model.positions.data[mask]) * 0.01
        new_scales = self.model.scales.data[mask] * 0.8
        new_rotations = self.model.rotations.data[mask]
        new_colors = self.model.colors.data[mask]
        new_opacity = self.model.opacity.data[mask]

        # Concatenate with existing gaussians
        self.model.positions.data = torch.cat([self.model.positions.data, new_positions])
        self.model.scales.data = torch.cat([self.model.scales.data, new_scales])
        self.model.rotations.data = torch.cat([self.model.rotations.data, new_rotations])
        self.model.colors.data = torch.cat([self.model.colors.data, new_colors])
        self.model.opacity.data = torch.cat([self.model.opacity.data, new_opacity])

        # Update original gaussians
        self.model.scales.data[mask] *= 0.8

    def prune_gaussians(self, mask):
        """가우시안 제거: 불필요한 가우시안 삭제"""
        keep_mask = ~mask

        self.model.positions.data = self.model.positions.data[keep_mask]
        self.model.scales.data = self.model.scales.data[keep_mask]
        self.model.rotations.data = self.model.rotations.data[keep_mask]
        self.model.colors.data = self.model.colors.data[keep_mask]
        self.model.opacity.data = self.model.opacity.data[keep_mask]

    def train_step(self, camera_matrix, camera_position, target_image, iteration):
        """
        Single training step

        Loss functions:
        1. L1 loss: 픽셀별 절대 차이
        2. SSIM loss: 구조적 유사도
        3. Perceptual loss: 고수준 특성 비교

        Args:
            camera_matrix: 카메라 파라미터
            camera_position: 카메라 위치
            target_image: 목표 이미지
            iteration: 현재 iteration 수

        Returns:
            dict: 손실 정보
        """
        # Clear gradients
        self.optimizer_positions.zero_grad()
        self.optimizer_colors.zero_grad()
        self.optimizer_opacity.zero_grad()
        self.optimizer_scales.zero_grad()
        self.optimizer_rotations.zero_grad()

        # Forward pass
        rendered_image = self.model(camera_matrix, camera_position,
                                  target_image.shape[0], target_image.shape[1])

        # L1 loss
        l1_loss = F.l1_loss(rendered_image, target_image)

        # SSIM loss (simplified)
        ssim_loss = self.compute_ssim_loss(rendered_image, target_image)

        # Total loss
        loss = 0.8 * l1_loss + 0.2 * ssim_loss

        # Regularization
        opacity_reg = torch.mean(self.model.opacity**2)
        scale_reg = torch.mean(torch.exp(self.model.scales)**2)
        loss += 0.01 * opacity_reg + 0.001 * scale_reg

        # Backward pass
        loss.backward()

        # Update parameters
        self.optimizer_positions.step()
        self.optimizer_colors.step()
        self.optimizer_opacity.step()
        self.optimizer_scales.step()
        self.optimizer_rotations.step()

        # Adaptive density control
        self.densification_and_pruning(iteration)

        return {
            'total_loss': loss.item(),
            'l1_loss': l1_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'num_gaussians': len(self.model.positions)
        }

    def compute_ssim_loss(self, img1, img2):
        """
        SSIM (Structural Similarity Index) Loss 계산

        SSIM은 인간의 시각적 인지와 더 잘 맞는 품질 측정:
        - 밝기 비교 (luminance)
        - 대비 비교 (contrast)
        - 구조 비교 (structure)

        Args:
            img1, img2: 비교할 이미지들 [H, W, 3]

        Returns:
            torch.Tensor: SSIM loss (1 - SSIM)
        """
        # Convert to grayscale for SSIM calculation
        def rgb_to_gray(img):
            return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

        gray1 = rgb_to_gray(img1)
        gray2 = rgb_to_gray(img2)

        # Constants for numerical stability
        C1 = 0.01**2
        C2 = 0.03**2

        # Calculate means
        mu1 = F.avg_pool2d(gray1.unsqueeze(0).unsqueeze(0), 11, stride=1, padding=5).squeeze()
        mu2 = F.avg_pool2d(gray2.unsqueeze(0).unsqueeze(0), 11, stride=1, padding=5).squeeze()

        # Calculate variances and covariance
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d((gray1**2).unsqueeze(0).unsqueeze(0), 11, stride=1, padding=5).squeeze() - mu1_sq
        sigma2_sq = F.avg_pool2d((gray2**2).unsqueeze(0).unsqueeze(0), 11, stride=1, padding=5).squeeze() - mu2_sq
        sigma12 = F.avg_pool2d((gray1 * gray2).unsqueeze(0).unsqueeze(0), 11, stride=1, padding=5).squeeze() - mu1_mu2

        # SSIM calculation
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim.mean()

# Example dataset
class MultiViewDataset(Dataset):
    """
    Multi-view 데이터셋 예제

    실제 3D GS 훈련에 필요한 데이터:
    - 다양한 시점의 고해상도 이미지들
    - 정확한 카메라 포즈 (COLMAP 등으로 추정)
    - 카메라 내부 파라미터
    - 선택적: 초기 point cloud (SfM으로 생성)
    """
    def __init__(self, n_views=100, image_size=(64, 64)):
        self.n_views = n_views
        self.image_size = image_size

        # Generate synthetic data
        self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """합성 데이터 생성 (실제로는 실제 이미지 로드)"""
        self.images = []
        self.camera_matrices = []
        self.camera_positions = []

        for i in range(self.n_views):
            # Synthetic image (실제로는 실제 이미지)
            img = torch.rand(*self.image_size, 3)

            # Synthetic camera parameters
            focal_length = 50.0
            camera_matrix = torch.tensor([
                [focal_length, 0, self.image_size[1]/2, 0],
                [0, focal_length, self.image_size[0]/2, 0],
                [0, 0, 1, 0]
            ], dtype=torch.float)

            # Circular camera trajectory
            angle = 2 * np.pi * i / self.n_views
            camera_pos = torch.tensor([2*np.cos(angle), 0.5, 2*np.sin(angle)])

            self.images.append(img)
            self.camera_matrices.append(camera_matrix)
            self.camera_positions.append(camera_pos)

    def __len__(self):
        return self.n_views

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'camera_matrix': self.camera_matrices[idx],
            'camera_position': self.camera_positions[idx]
        }

if __name__ == "__main__":
    print("3D Gaussian Splatting Example")
    print("Key innovations:")
    print("- 3D scene representation with millions of 3D Gaussians")
    print("- Differentiable 2D splatting for real-time rendering")
    print("- Adaptive density control during training")
    print("- View-dependent colors via spherical harmonics")

    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GaussianModel(num_gaussians=10000)
    trainer = GaussianSplattingTrainer(model, device)

    print(f"Initial number of Gaussians: {model.num_gaussians}")
    print("Note: This is a simplified educational implementation.")
    print("Production implementations use:")
    print("- Efficient CUDA kernels for splatting")
    print("- Tile-based rasterization")
    print("- Advanced culling and sorting")
    print("- Memory-optimized data structures")