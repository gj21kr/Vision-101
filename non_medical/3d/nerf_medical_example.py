"""
Medical NeRF Example with Result Logging

의료 영상 데이터에 특화된 NeRF 구현으로, 3D 의료 볼륨 데이터나
다중 시점 의료 이미지로부터 3D reconstruction을 수행합니다.

특화 기능:
- 의료 볼륨 데이터 (CT, MRI) 지원
- 훈련 과정 상세 로깅
- 렌더링 결과 자동 저장
- 3D 구조 시각화 및 저장
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT.name not in {"medical", "non_medical"} and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if PROJECT_ROOT.name in {"medical", "non_medical"}:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ensure project root is on sys.path
from medical.medical_data_utils import MedicalImageLoader
from non_medical.result_logger import create_logger_for_3d

class PositionalEncoder(nn.Module):
    def __init__(self, d_input=3, n_freqs=10, log_space=False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)

        if self.log_space:
            freq_bands = 2.**torch.linspace(0., self.n_freqs-1, self.n_freqs)
        else:
            freq_bands = torch.linspace(1., 2.**(self.n_freqs-1), self.n_freqs)

        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        out = [x]
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                out.append(func(freq * x))

        return torch.cat(out, dim=-1)

class MedicalNeRF(nn.Module):
    def __init__(self, d_pos=3, d_dir=3, n_layers=8, d_filter=256, skip=4,
                 use_viewdirs=True, pos_freqs=10, dir_freqs=4, density_activation='relu'):
        super().__init__()

        self.d_pos = d_pos
        self.d_dir = d_dir
        self.skip = skip
        self.use_viewdirs = use_viewdirs
        self.density_activation = density_activation

        # Positional encoders
        self.pos_encoder = PositionalEncoder(d_pos, pos_freqs, log_space=True)
        self.dir_encoder = PositionalEncoder(d_dir, dir_freqs, log_space=True) if use_viewdirs else None

        # Dimensions after encoding
        d_pos_encoded = self.pos_encoder.d_output
        d_dir_encoded = self.dir_encoder.d_output if use_viewdirs else 0

        # Position network
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

        # Color network
        if use_viewdirs:
            self.color_layer1 = nn.Linear(d_filter + d_dir_encoded, d_filter // 2)
            self.color_layer2 = nn.Linear(d_filter // 2, 3)
        else:
            self.color_head = nn.Linear(d_filter, 3)

    def forward(self, pos, dirs=None):
        # Encode positions
        pos_encoded = self.pos_encoder(pos)

        # Position network with skip connection
        x = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            if i == self.skip:
                x = torch.cat([x, pos_encoded], dim=-1)
            x = F.relu(layer(x))

        # Density
        if self.density_activation == 'relu':
            density = F.relu(self.density_head(x))
        else:  # softplus for medical data
            density = F.softplus(self.density_head(x))

        # Feature extraction
        feature = self.feature_layer(x)

        # Color
        if self.use_viewdirs and dirs is not None:
            dirs_encoded = self.dir_encoder(dirs)
            color_input = torch.cat([feature, dirs_encoded], dim=-1)
            color = F.relu(self.color_layer1(color_input))
            color = torch.sigmoid(self.color_layer2(color))
        else:
            color = torch.sigmoid(self.color_head(feature))

        return color, density

def create_medical_volume_dataset(volume_type='brain_mri', volume_size=64):
    """
    의료 볼륨 데이터셋 생성

    Args:
        volume_type: 볼륨 타입 ('brain_mri', 'ct_scan', 'organ')
        volume_size: 볼륨 크기

    Returns:
        3D volume data and corresponding camera poses
    """
    loader = MedicalImageLoader(volume_type, volume_size)

    # Create 3D volume
    volume = loader.create_synthetic_medical_data(1, data_type='3d')[0]

    # Generate camera poses for multi-view rendering
    poses = []
    n_views = 32  # Number of views around the volume

    for i in range(n_views):
        # Circular camera trajectory
        angle = 2 * np.pi * i / n_views
        radius = 3.0

        # Camera position
        cam_x = radius * np.cos(angle)
        cam_y = radius * np.sin(angle)
        cam_z = 1.0

        # Look at center
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])

        # Camera position
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # Forward direction
        forward = look_at - cam_pos
        forward = forward / np.linalg.norm(forward)

        # Right direction
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        # Up direction
        up = np.cross(right, forward)

        # Create pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = cam_pos

        poses.append(torch.FloatTensor(pose))

    return volume, poses

def volume_render_medical(colors, densities, dists, background_color=None):
    """
    의료 데이터에 특화된 볼륨 렌더링

    Args:
        colors: RGB colors [batch, n_samples, 3]
        densities: Volume densities [batch, n_samples, 1]
        dists: Sample distances [batch, n_samples]
        background_color: 배경색 (의료 영상의 경우 보통 검은색)
    """
    # Alpha compositing
    alpha = 1 - torch.exp(-densities * dists.unsqueeze(-1))

    # Transmittance
    transmittance = torch.cumprod(torch.cat([
        torch.ones_like(alpha[..., :1]),
        1 - alpha + 1e-10
    ], dim=-1), dim=-1)[..., :-1]

    # Weights
    weights = transmittance * alpha

    # Rendered color
    rgb = torch.sum(weights * colors, dim=-2)

    # Add background
    if background_color is not None:
        acc_weights = torch.sum(weights[..., 0], dim=-1, keepdim=True)
        rgb = rgb + (1 - acc_weights) * background_color

    # Depth and opacity
    depth = torch.sum(weights[..., 0] * dists, dim=-1, keepdim=True)
    acc = torch.sum(weights[..., 0], dim=-1, keepdim=True)

    return rgb, depth, acc, weights.squeeze(-1)

def get_rays_medical(height, width, focal_length, pose, near=0.1, far=5.0):
    """의료 데이터용 ray generation"""
    i, j = torch.meshgrid(torch.linspace(0, width-1, width),
                         torch.linspace(0, height-1, height))
    i, j = i.t(), j.t()

    # Camera coordinates
    dirs = torch.stack([(i - width * 0.5) / focal_length,
                       -(j - height * 0.5) / focal_length,
                       -torch.ones_like(i)], dim=-1)

    # Transform to world coordinates
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], dim=-1)
    rays_o = pose[:3, -1].expand(rays_d.shape)

    # Normalize direction vectors
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    return rays_o, rays_d

class MedicalNeRFTrainer:
    def __init__(self, model, logger, device='cuda'):
        self.model = model.to(device)
        self.logger = logger
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    def train_step(self, rays_o, rays_d, target_colors, near=0.1, far=5.0, n_samples=64):
        """단일 훈련 스텝"""
        self.optimizer.zero_grad()

        batch_size = rays_o.shape[0]

        # Sample points along rays
        t_vals = torch.linspace(near, far, n_samples, device=self.device)
        t_vals = t_vals.expand(batch_size, n_samples)

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

        # Calculate distances
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)

        # Volume rendering
        rgb, depth, acc, weights = volume_render_medical(colors, densities, dists)

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

def train_medical_nerf(volume_type='brain_mri', num_epochs=100, save_interval=20):
    """
    Medical NeRF 훈련 함수

    Args:
        volume_type: 의료 볼륨 타입
        num_epochs: 훈련 에포크 수
        save_interval: 결과 저장 간격
    """
    # Create logger
    logger = create_logger_for_3d("nerf", volume_type)

    # Configuration
    config = {
        'algorithm': 'NeRF',
        'volume_type': volume_type,
        'num_epochs': num_epochs,
        'image_size': 64,
        'n_samples': 64,
        'learning_rate': 5e-4
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Create medical volume dataset
    logger.log(f"Creating {volume_type} volume dataset...")
    volume, poses = create_medical_volume_dataset(volume_type, volume_size=64)

    # Save volume visualization
    mid_slice = volume.shape[0] // 2
    logger.save_image(volume[mid_slice], "original_volume_slice",
                     title=f"{volume_type} - Middle Slice")

    # Create 3D visualization of volume
    fig = plt.figure(figsize=(12, 4))

    # Axial view
    ax1 = fig.add_subplot(131)
    ax1.imshow(volume[mid_slice], cmap='gray')
    ax1.set_title('Axial View')
    ax1.axis('off')

    # Coronal view
    ax2 = fig.add_subplot(132)
    ax2.imshow(volume[:, volume.shape[1]//2, :], cmap='gray')
    ax2.set_title('Coronal View')
    ax2.axis('off')

    # Sagittal view
    ax3 = fig.add_subplot(133)
    ax3.imshow(volume[:, :, volume.shape[2]//2], cmap='gray')
    ax3.set_title('Sagittal View')
    ax3.axis('off')

    plt.suptitle(f'{volume_type} - Original Volume')
    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['images'], 'original_volume_views.png'), dpi=150)
    plt.close()

    logger.log(f"Volume shape: {volume.shape}")
    logger.log(f"Generated {len(poses)} camera poses")

    # Initialize NeRF model
    model = MedicalNeRF(d_pos=3, d_dir=3, n_layers=8, d_filter=256,
                       density_activation='softplus')  # softplus for medical

    # Create trainer
    trainer = MedicalNeRFTrainer(model, logger, device)

    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate target images from volume (simplified rendering)
    logger.log("Generating target images from volume...")
    height, width = 64, 64
    focal_length = 50.0
    target_images = []

    # Simple volume ray casting for targets (simplified)
    for i, pose in enumerate(poses[:8]):  # Use first 8 poses for training
        # This is a simplified version - real implementation would do proper ray casting
        target_img = np.random.rand(height, width, 3) * 0.5  # Dummy target
        target_images.append(target_img)

    # Training loop
    logger.log("Starting Medical NeRF training...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        # Train on multiple views
        for view_idx in range(min(8, len(poses))):
            pose = poses[view_idx]
            target_img = torch.FloatTensor(target_images[view_idx]).to(device)

            # Generate rays
            rays_o, rays_d = get_rays_medical(height, width, focal_length, pose)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)

            # Flatten for processing
            rays_o_flat = rays_o.reshape(-1, 3)
            rays_d_flat = rays_d.reshape(-1, 3)
            target_flat = target_img.reshape(-1, 3)

            # Process in chunks to avoid memory issues
            chunk_size = 1024
            for i in range(0, len(rays_o_flat), chunk_size):
                end_idx = min(i + chunk_size, len(rays_o_flat))

                result = trainer.train_step(
                    rays_o_flat[i:end_idx],
                    rays_d_flat[i:end_idx],
                    target_flat[i:end_idx]
                )

                epoch_loss += result['loss']
                num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.log_metrics(epoch + 1, avg_loss)

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Saving results at epoch {epoch + 1}...")

            model.eval()
            with torch.no_grad():
                # Render from novel viewpoint
                test_pose = poses[0]  # Use first pose for testing
                rays_o, rays_d = get_rays_medical(height, width, focal_length, test_pose)
                rays_o = rays_o.to(device)
                rays_d = rays_d.to(device)

                rendered_img = torch.zeros(height, width, 3)

                # Render in chunks
                rays_o_flat = rays_o.reshape(-1, 3)
                rays_d_flat = rays_d.reshape(-1, 3)

                for i in range(0, len(rays_o_flat), chunk_size):
                    end_idx = min(i + chunk_size, len(rays_o_flat))

                    # Sample points along rays
                    t_vals = torch.linspace(0.1, 5.0, 64, device=device)
                    t_vals = t_vals.expand(end_idx - i, 64)

                    pts = rays_o_flat[i:end_idx].unsqueeze(1) + \
                          t_vals.unsqueeze(-1) * rays_d_flat[i:end_idx].unsqueeze(1)

                    dirs = rays_d_flat[i:end_idx].unsqueeze(1).expand_as(pts)
                    colors, densities = model(pts.reshape(-1, 3), dirs.reshape(-1, 3))
                    colors = colors.reshape(end_idx - i, 64, 3)
                    densities = densities.reshape(end_idx - i, 64, 1)

                    # Simple volume rendering
                    dists = torch.full((end_idx - i, 64), 5.0/64, device=device)
                    rgb, _, _, _ = volume_render_medical(colors, densities, dists)

                    rendered_img.reshape(-1, 3)[i:end_idx] = rgb.cpu()

                # Save rendered image
                rendered_np = rendered_img.numpy()
                rendered_np = np.clip(rendered_np, 0, 1)

                logger.save_image(rendered_np, f"rendered_epoch_{epoch+1:03d}",
                                title=f"NeRF Rendering - Epoch {epoch+1}")

            model.train()

        # Save model checkpoint
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(model, f"nerf_checkpoint_epoch_{epoch+1:03d}",
                             optimizer=trainer.optimizer, epoch=epoch+1,
                             config=config)

    # Final results
    logger.log("Training completed! Generating final results...")

    model.eval()
    with torch.no_grad():
        # Generate multi-view renderings
        final_renderings = []

        for i, pose in enumerate(poses[:16]):  # Render first 16 views
            rays_o, rays_d = get_rays_medical(height, width, focal_length, pose)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)

            rendered_img = torch.zeros(height, width, 3)
            rays_o_flat = rays_o.reshape(-1, 3)
            rays_d_flat = rays_d.reshape(-1, 3)

            for j in range(0, len(rays_o_flat), chunk_size):
                end_idx = min(j + chunk_size, len(rays_o_flat))

                t_vals = torch.linspace(0.1, 5.0, 64, device=device)
                t_vals = t_vals.expand(end_idx - j, 64)

                pts = rays_o_flat[j:end_idx].unsqueeze(1) + \
                      t_vals.unsqueeze(-1) * rays_d_flat[j:end_idx].unsqueeze(1)

                dirs = rays_d_flat[j:end_idx].unsqueeze(1).expand_as(pts)
                colors, densities = model(pts.reshape(-1, 3), dirs.reshape(-1, 3))
                colors = colors.reshape(end_idx - j, 64, 3)
                densities = densities.reshape(end_idx - j, 64, 1)

                dists = torch.full((end_idx - j, 64), 5.0/64, device=device)
                rgb, _, _, _ = volume_render_medical(colors, densities, dists)

                rendered_img.reshape(-1, 3)[j:end_idx] = rgb.cpu()

            final_renderings.append(np.clip(rendered_img.numpy(), 0, 1))

        # Save final renderings
        logger.save_image_grid(
            final_renderings,
            "final_novel_view_synthesis",
            titles=[f"View {i+1}" for i in range(len(final_renderings))],
            rows=4, cols=4
        )

    # Plot training curves
    logger.plot_training_curves()

    # Save final model
    logger.save_model(model, "nerf_final_model", optimizer=trainer.optimizer,
                     epoch=num_epochs, config=config)

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return model, results_dir

if __name__ == "__main__":
    print("Medical NeRF with Result Logging")
    print("=================================")

    # Configuration
    volume_types = ['brain_mri', 'ct_scan']
    selected_volume = 'brain_mri'  # Change this to test different volume types

    print(f"Training Medical NeRF on {selected_volume} data...")
    print("Results will be automatically saved including:")
    print("- Original volume visualizations")
    print("- Training progress logs")
    print("- Novel view synthesis results")
    print("- Model checkpoints")
    print("- Training curves")

    try:
        model, results_dir = train_medical_nerf(
            volume_type=selected_volume,
            num_epochs=50,
            save_interval=10
        )

        print(f"\nTraining completed successfully!")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Volume visualizations and rendered views")
        print("- models/: NeRF model checkpoints")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
