#!/usr/bin/env python3
"""Unified Medical Image and Volume Synthesis Example."""

#!/usr/bin/env python3
"""Unified medical synthesis script supporting 2D and 3D modalities."""

import math
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Ensure the project root is importable
sys.path.append('/workspace/Vision-101')
try:
    from result_logger import create_logger_for_medical_synthesis
except ImportError:  # pragma: no cover - fallback for standalone usage
    from datetime import datetime

    class SimpleLogger:
        def __init__(self, name):
            self.name = name
            self.start_time = datetime.now()

        def log(self, message):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] {message}")

        def save_image_grid(self, images, filename, titles=None, nrow=4):
            # Minimal placeholder
            self.log(f"Image grid '{filename}' would be saved (placeholder).")

        def save_numpy_array(self, array, filename, description=None):
            self.log(f"NumPy array '{filename}' would be saved (placeholder).")

        def log_metrics(self, epoch, train_loss, val_loss=None, **kwargs):
            self.log(f"Epoch {epoch}: Train Loss = {train_loss:.6f}")

    def create_logger_for_medical_synthesis(algorithm, dataset):
        return SimpleLogger(f"medical_synthesis_{algorithm}_{dataset}")


# ---------------------------------------------------------------------------
# 2D medical synthesis datasets and models
# ---------------------------------------------------------------------------


class MedicalModality(Enum):
    """의료 영상 모달리티"""

    CHEST_XRAY = "chest_xray"
    CT_SCAN = "ct_scan"
    MRI = "mri"
    MAMMOGRAPHY = "mammography"
    ULTRASOUND = "ultrasound"
    RETINAL = "retinal"


@dataclass
class MedicalCondition:
    """의료 상태 정보"""

    pathology: str
    severity: float
    location: Tuple[float, float]
    size: float
    age: float
    gender: int


class Medical2DDataset(Dataset):
    """2D 의료 영상 데이터셋"""

    def __init__(
        self,
        modality: MedicalModality,
        num_samples: int = 1000,
        image_size: int = 256,
        transform=None,
    ):
        self.modality = modality
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.samples = self._generate_medical_samples()

    # The generation utilities below mirror the original 2D example.
    def _generate_medical_samples(self):
        samples = []
        for i in range(self.num_samples):
            np.random.seed(i)
            if self.modality == MedicalModality.CHEST_XRAY:
                image = self._generate_chest_xray(i)
            elif self.modality == MedicalModality.CT_SCAN:
                image = self._generate_ct_scan(i)
            elif self.modality == MedicalModality.MRI:
                image = self._generate_mri(i)
            elif self.modality == MedicalModality.MAMMOGRAPHY:
                image = self._generate_mammography(i)
            else:
                image = self._generate_generic_medical(i)

            condition = self._generate_medical_condition(i)
            samples.append(
                {
                    "image": image,
                    "condition": condition,
                    "modality": self.modality.value,
                }
            )
        return samples

    def _generate_chest_xray(self, seed: int) -> np.ndarray:
        np.random.seed(seed)
        image = np.ones((self.image_size, self.image_size)) * 0.8
        center_x, center_y = self.image_size // 2, self.image_size // 2

        for y in range(self.image_size // 6, 5 * self.image_size // 6):
            for x in range(self.image_size // 6, center_x - 10):
                dist = np.sqrt((x - self.image_size // 4) ** 2 + (y - center_y) ** 2)
                if dist < self.image_size // 3:
                    image[y, x] = 0.3 + 0.3 * np.random.random()

        for y in range(self.image_size // 6, 5 * self.image_size // 6):
            for x in range(center_x + 10, 5 * self.image_size // 6):
                dist = np.sqrt((x - 3 * self.image_size // 4) ** 2 + (y - center_y) ** 2)
                if dist < self.image_size // 3:
                    image[y, x] = 0.3 + 0.3 * np.random.random()

        for i in range(8):
            rib_y = self.image_size // 6 + i * self.image_size // 12
            for x in range(self.image_size // 6, 5 * self.image_size // 6):
                if rib_y < self.image_size:
                    image[rib_y : rib_y + 2, x] = 0.9

        if np.random.random() > 0.7:
            for y in range(self.image_size // 2, 5 * self.image_size // 6):
                for x in range(self.image_size // 3, 2 * self.image_size // 3):
                    image[y, x] = max(image[y, x], 0.9)

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _generate_ct_scan(self, seed: int) -> np.ndarray:
        np.random.seed(seed)
        image = np.random.normal(0.5, 0.1, (self.image_size, self.image_size))
        image = scipy.ndimage.gaussian_filter(image, sigma=1)

        for _ in range(np.random.randint(2, 5)):
            center_x = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
            center_y = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
            radius = np.random.randint(self.image_size // 10, self.image_size // 6)
            intensity = np.random.uniform(0.6, 0.9)

            y, x = np.ogrid[: self.image_size, : self.image_size]
            mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
            image[mask] = np.clip(image[mask] + intensity * np.random.uniform(0.2, 0.5), 0, 1)

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _generate_mri(self, seed: int) -> np.ndarray:
        np.random.seed(seed)
        image = np.random.normal(0.5, 0.05, (self.image_size, self.image_size))
        image = scipy.ndimage.gaussian_filter(image, sigma=1.5)

        ventricles_x, ventricles_y = self.image_size // 2, self.image_size // 2
        ventricles_radius = self.image_size // 8
        y, x = np.ogrid[: self.image_size, : self.image_size]
        ventricles_mask = (x - ventricles_x) ** 2 + (y - ventricles_y) ** 2 <= ventricles_radius ** 2
        image[ventricles_mask] = np.clip(image[ventricles_mask] + 0.2, 0, 1)

        if np.random.random() > 0.6:
            lesion_x = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
            lesion_y = np.random.randint(self.image_size // 4, 3 * self.image_size // 4)
            lesion_radius = np.random.randint(self.image_size // 12, self.image_size // 8)
            lesion_mask = (x - lesion_x) ** 2 + (y - lesion_y) ** 2 <= lesion_radius ** 2
            image[lesion_mask] = np.clip(image[lesion_mask] + 0.3, 0, 1)

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _generate_mammography(self, seed: int) -> np.ndarray:
        np.random.seed(seed)
        image = np.ones((self.image_size, self.image_size)) * 0.2

        for i in range(100):
            x = np.random.randint(0, self.image_size)
            y = np.random.randint(0, self.image_size)
            size = np.random.randint(5, 15)
            intensity = 0.3 + 0.4 * np.random.random()

            for dy in range(-size, size):
                for dx in range(-size, size):
                    if 0 <= y + dy < self.image_size and 0 <= x + dx < self.image_size:
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < size:
                            image[y + dy, x + dx] = max(
                                image[y + dy, x + dx], intensity * (1 - dist / size)
                            )

        for i in range(np.random.randint(0, 20)):
            calc_x = np.random.randint(0, self.image_size)
            calc_y = np.random.randint(0, self.image_size)
            calc_size = np.random.randint(1, 3)

            for dy in range(-calc_size, calc_size + 1):
                for dx in range(-calc_size, calc_size + 1):
                    if 0 <= calc_y + dy < self.image_size and 0 <= calc_x + dx < self.image_size:
                        image[calc_y + dy, calc_x + dx] = 0.9

        if np.random.random() < 0.2:
            mass_x = np.random.randint(50, self.image_size - 50)
            mass_y = np.random.randint(50, self.image_size - 50)
            mass_size = np.random.randint(20, 40)

            for y in range(max(0, mass_y - mass_size), min(self.image_size, mass_y + mass_size)):
                for x in range(max(0, mass_x - mass_size), min(self.image_size, mass_x + mass_size)):
                    dist = np.sqrt((x - mass_x) ** 2 + (y - mass_y) ** 2)
                    if dist < mass_size:
                        image[y, x] = min(1.0, image[y, x] + 0.4 * (1 - dist / mass_size))

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _generate_generic_medical(self, seed: int) -> np.ndarray:
        np.random.seed(seed)
        image = np.random.normal(0.5, 0.2, (self.image_size, self.image_size))
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _generate_medical_condition(self, seed: int) -> MedicalCondition:
        np.random.seed(seed)
        pathologies = [
            "normal",
            "pneumonia",
            "nodule",
            "mass",
            "fracture",
            "tumor",
            "lesion",
            "calcification",
            "atelectasis",
            "edema",
        ]
        return MedicalCondition(
            pathology=np.random.choice(pathologies),
            severity=np.random.random(),
            location=(np.random.random(), np.random.random()),
            size=np.random.random(),
            age=np.random.random(),
            gender=np.random.randint(0, 2),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.fromarray(sample["image"]).convert("L")
        if self.transform:
            image = self.transform(image)
        condition = sample["condition"]
        condition_tensor = torch.tensor(
            [
                condition.severity,
                condition.location[0],
                condition.location[1],
                condition.size,
                condition.age,
                condition.gender,
            ],
            dtype=torch.float32,
        )
        return {
            "image": image,
            "condition": condition_tensor,
            "modality": sample["modality"],
            "pathology": condition.pathology,
        }
class MedicalDiffusionUNet(nn.Module):
    """의료 영상 특화 Diffusion U-Net"""

    def __init__(self, in_channels=1, out_channels=1, condition_dim=6,
                 base_channels=64, time_embed_dim=256):
        super(MedicalDiffusionUNet, self).__init__()

        self.time_embed_dim = time_embed_dim
        self.condition_dim = condition_dim

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Condition embedding
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels)
        )

        # Down path
        self.down1 = self._make_conv_block(in_channels, base_channels, time_embed_dim)
        self.down2 = self._make_conv_block(base_channels, base_channels * 2, time_embed_dim)
        self.down3 = self._make_conv_block(base_channels * 2, base_channels * 4, time_embed_dim)
        self.down4 = self._make_conv_block(base_channels * 4, base_channels * 8, time_embed_dim)

        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 16, 3, padding=1),
            nn.GroupNorm(8, base_channels * 16),
            nn.SiLU(),
            nn.MultiheadAttention(base_channels * 16, 8, batch_first=True),
            nn.Conv2d(base_channels * 16, base_channels * 8, 3, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU()
        )

        # Up path
        self.up4 = self._make_conv_block(base_channels * 16, base_channels * 4, time_embed_dim)
        self.up3 = self._make_conv_block(base_channels * 8, base_channels * 2, time_embed_dim)
        self.up2 = self._make_conv_block(base_channels * 4, base_channels, time_embed_dim)
        self.up1 = self._make_conv_block(base_channels * 2, base_channels, time_embed_dim)

        # Output
        self.output = nn.Conv2d(base_channels, out_channels, 1)

        # Max pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _make_conv_block(self, in_ch, out_ch, time_embed_dim):
        return nn.ModuleDict({
            'conv1': nn.Conv2d(in_ch, out_ch, 3, padding=1),
            'norm1': nn.GroupNorm(min(out_ch // 4, 32), out_ch),
            'conv2': nn.Conv2d(out_ch, out_ch, 3, padding=1),
            'norm2': nn.GroupNorm(min(out_ch // 4, 32), out_ch),
            'time_proj': nn.Linear(time_embed_dim, out_ch),
            'act': nn.SiLU()
        })

    def _apply_conv_block(self, x, block, time_embed, condition_embed=None):
        # First conv
        h = block['conv1'](x)
        h = block['norm1'](h)

        # Add time embedding
        time_proj = block['time_proj'](time_embed)
        h = h + time_proj[:, :, None, None]

        # Add condition embedding if provided
        if condition_embed is not None:
            h = h + condition_embed[:, :, None, None]

        h = block['act'](h)

        # Second conv
        h = block['conv2'](h)
        h = block['norm2'](h)
        h = block['act'](h)

        return h

    def positional_encoding(self, timesteps, dim):
        """Sinusoidal positional encoding for time steps"""
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def forward(self, x, timesteps, condition):
        # Time embedding
        time_embed = self.positional_encoding(timesteps, self.time_embed_dim)
        time_embed = self.time_embedding(time_embed)

        # Condition embedding
        condition_embed = self.condition_embedding(condition)

        # Encoder
        x1 = self._apply_conv_block(x, self.down1, time_embed, condition_embed)
        x2 = self._apply_conv_block(self.pool(x1), self.down2, time_embed)
        x3 = self._apply_conv_block(self.pool(x2), self.down3, time_embed)
        x4 = self._apply_conv_block(self.pool(x3), self.down4, time_embed)

        # Bottleneck
        bottleneck = x4
        for layer in self.bottleneck:
            if isinstance(layer, nn.MultiheadAttention):
                # Reshape for attention
                b, c, h, w = bottleneck.shape
                bottleneck_flat = bottleneck.view(b, c, -1).permute(0, 2, 1)
                attn_out, _ = layer(bottleneck_flat, bottleneck_flat, bottleneck_flat)
                bottleneck = attn_out.permute(0, 2, 1).view(b, c, h, w)
            else:
                bottleneck = layer(bottleneck)

        # Decoder
        up4_upsampled = self.upsample(bottleneck)
        # Ensure spatial dimensions match
        if up4_upsampled.shape[-2:] != x4.shape[-2:]:
            up4_upsampled = F.interpolate(up4_upsampled, size=x4.shape[-2:], mode='bilinear', align_corners=True)
        up4 = self._apply_conv_block(
            torch.cat([up4_upsampled, x4], dim=1),
            self.up4, time_embed
        )

        up3_upsampled = self.upsample(up4)
        if up3_upsampled.shape[-2:] != x3.shape[-2:]:
            up3_upsampled = F.interpolate(up3_upsampled, size=x3.shape[-2:], mode='bilinear', align_corners=True)
        up3 = self._apply_conv_block(
            torch.cat([up3_upsampled, x3], dim=1),
            self.up3, time_embed
        )

        up2_upsampled = self.upsample(up3)
        if up2_upsampled.shape[-2:] != x2.shape[-2:]:
            up2_upsampled = F.interpolate(up2_upsampled, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        up2 = self._apply_conv_block(
            torch.cat([up2_upsampled, x2], dim=1),
            self.up2, time_embed
        )

        up1_upsampled = self.upsample(up2)
        if up1_upsampled.shape[-2:] != x1.shape[-2:]:
            up1_upsampled = F.interpolate(up1_upsampled, size=x1.shape[-2:], mode='bilinear', align_corners=True)
        up1 = self._apply_conv_block(
            torch.cat([up1_upsampled, x1], dim=1),
            self.up1, time_embed
        )

        return self.output(up1)

class MedicalDDPM(nn.Module):
    """Medical-specific DDPM (Denoising Diffusion Probabilistic Model)"""

    def __init__(self, unet, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super(MedicalDDPM, self).__init__()

        self.unet = unet
        self.num_timesteps = num_timesteps

        # Define beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def q_sample(self, x_start, t, noise=None):
        """Sample from q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, x_start, t, condition, noise=None):
        """Calculate training loss"""
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.unet(x_noisy, t, condition)

        # Medical-specific loss: MSE + Perceptual + Edge preservation
        mse_loss = F.mse_loss(predicted_noise, noise)

        # Edge preservation loss
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                              dtype=torch.float32, device=x_start.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                              dtype=torch.float32, device=x_start.device).view(1, 1, 3, 3)

        edges_real = torch.sqrt(
            F.conv2d(x_start, sobel_x, padding=1)**2 +
            F.conv2d(x_start, sobel_y, padding=1)**2
        )

        x_denoised = (x_noisy - self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1) * predicted_noise) / \
                     self.sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)

        edges_pred = torch.sqrt(
            F.conv2d(x_denoised, sobel_x, padding=1)**2 +
            F.conv2d(x_denoised, sobel_y, padding=1)**2
        )

        edge_loss = F.mse_loss(edges_pred, edges_real)

        return mse_loss + 0.1 * edge_loss

    @torch.no_grad()
    def p_sample(self, x, t, condition):
        """Sample from p(x_{t-1} | x_t)"""
        pred_noise = self.unet(x, t, condition)

        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cumprod_t = self.alpha_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1, 1)

        # Predict x_0
        pred_x0 = (x - sqrt_one_minus_alpha_cumprod_t * pred_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # Compute mean
        model_mean = (
            torch.sqrt(alpha_t) * (1 - self.alpha_cumprod_prev[t].view(-1, 1, 1, 1)) * x +
            torch.sqrt(self.alpha_cumprod_prev[t].view(-1, 1, 1, 1)) * beta_t * pred_x0
        ) / (1 - alpha_cumprod_t)

        if t[0] > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(self.posterior_variance[t].view(-1, 1, 1, 1))
            return model_mean + variance * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, shape, condition, device):
        """Generate samples"""
        x = torch.randn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, condition)

        return x

# Medical-specific StyleGAN Generator
class MedicalStyleGenerator(nn.Module):
    """의료 영상 특화 StyleGAN Generator"""

    def __init__(self, latent_dim=512, condition_dim=6, img_size=256, img_channels=1):
        super(MedicalStyleGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size
        self.condition_dim = condition_dim

        # Mapping network for style codes
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim)
        )

        # Synthesis network
        self.start_size = 4
        self.num_layers = int(math.log2(img_size) - 1)  # -1 because we start from 4x4

        # Constant input
        self.const_input = nn.Parameter(torch.randn(1, latent_dim, self.start_size, self.start_size))

        # Progressive layers
        self.layers = nn.ModuleList()
        in_channels = latent_dim

        for i in range(self.num_layers):
            out_channels = min(latent_dim, latent_dim // (2**i))
            layer = self._make_style_block(in_channels, out_channels, latent_dim)
            self.layers.append(layer)
            in_channels = out_channels

        # Final output layer
        self.to_rgb = nn.Conv2d(out_channels, img_channels, 1)

    def _make_style_block(self, in_channels, out_channels, style_dim):
        return nn.ModuleDict({
            'conv1': nn.Conv2d(in_channels, out_channels, 3, padding=1),
            'conv2': nn.Conv2d(out_channels, out_channels, 3, padding=1),
            'style1': nn.Linear(style_dim, in_channels),
            'style2': nn.Linear(style_dim, out_channels),
            'noise1': nn.Parameter(torch.zeros(1)),
            'noise2': nn.Parameter(torch.zeros(1)),
            'activation': nn.LeakyReLU(0.2)
        })

    def _apply_style_block(self, x, block, style):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        # First conv with style modulation
        style1 = block['style1'](style)
        x = x * (1 + style1.view(x.size(0), x.size(1), 1, 1))
class VolumeType(Enum):
    """3D 볼륨 유형"""
    CT_CHEST = "ct_chest"
    CT_ABDOMEN = "ct_abdomen"
    CT_HEAD = "ct_head"
    MRI_BRAIN = "mri_brain"
    MRI_CARDIAC = "mri_cardiac"
    MRI_SPINE = "mri_spine"

@dataclass
class Volume3DCondition:
    """3D 볼륨 생성 조건"""
    anatomy: str  # 해부학적 부위
    pathology: str  # 병리학적 상태
    age: float  # 환자 나이 (normalized)
    gender: int  # 성별
    contrast: bool  # 조영제 사용 여부
    slice_thickness: float  # 슬라이스 두께
    resolution: Tuple[float, float, float]  # 공간 해상도
    organ_mask: Optional[np.ndarray] = None  # 장기 마스크

class Medical3DDataset(Dataset):
    """3D 의료 볼륨 데이터셋"""

    def __init__(self, volume_type: VolumeType, num_samples: int = 200,
                 volume_size: Tuple[int, int, int] = (64, 64, 64)):
        self.volume_type = volume_type
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.depth, self.height, self.width = volume_size

        # Generate synthetic 3D medical volumes
        self.samples = self._generate_3d_volumes()

    def _generate_3d_volumes(self):
        """3D 의료 볼륨 생성"""
        samples = []

        for i in range(self.num_samples):
            np.random.seed(i)

            # Generate base 3D volume
            if self.volume_type == VolumeType.CT_CHEST:
                volume = self._generate_ct_chest(i)
            elif self.volume_type == VolumeType.CT_ABDOMEN:
                volume = self._generate_ct_abdomen(i)
            elif self.volume_type == VolumeType.MRI_BRAIN:
                volume = self._generate_mri_brain(i)
            elif self.volume_type == VolumeType.MRI_CARDIAC:
                volume = self._generate_mri_cardiac(i)
            else:
                volume = self._generate_generic_volume(i)

            # Generate condition
            condition = self._generate_volume_condition(i)

            samples.append({
                'volume': volume,
                'condition': condition,
                'volume_type': self.volume_type.value
            })

        return samples

    def _generate_ct_chest(self, seed: int) -> np.ndarray:
        """3D 흉부 CT 볼륨 생성"""
        np.random.seed(seed)

        volume = np.ones(self.volume_size) * 0.2  # Air background

        d, h, w = self.volume_size
        center_d, center_h, center_w = d // 2, h // 2, w // 2

        # Generate body outline
        for z in range(d):
            # Elliptical body cross-section that varies with z
            body_scale = 0.8 + 0.2 * np.sin(np.pi * z / d)
            a = int(w * 0.4 * body_scale)  # Semi-major axis
            b = int(h * 0.35 * body_scale)  # Semi-minor axis

            for y in range(h):
                for x in range(w):
                    # Ellipse equation
                    if ((x - center_w)**2 / a**2 + (y - center_h)**2 / b**2) <= 1:
                        volume[z, y, x] = 0.5  # Soft tissue

        # Add lungs (lower density regions)
        lung_depth_start = d // 4
        lung_depth_end = 3 * d // 4

        # Left lung
        left_lung_center_w = center_w - w // 6
        for z in range(lung_depth_start, lung_depth_end):
            lung_scale = 0.6 + 0.3 * np.sin(np.pi * (z - lung_depth_start) / (lung_depth_end - lung_depth_start))
            lung_a = int(w * 0.15 * lung_scale)
            lung_b = int(h * 0.25 * lung_scale)

            for y in range(center_h - lung_b, center_h + lung_b):
                for x in range(left_lung_center_w - lung_a, left_lung_center_w + lung_a):
                    if (0 <= y < h and 0 <= x < w and
                        ((x - left_lung_center_w)**2 / lung_a**2 + (y - center_h)**2 / lung_b**2) <= 1):
                        volume[z, y, x] = 0.1  # Lung tissue (air-filled)

        # Right lung
        right_lung_center_w = center_w + w // 6
        for z in range(lung_depth_start, lung_depth_end):
            lung_scale = 0.6 + 0.3 * np.sin(np.pi * (z - lung_depth_start) / (lung_depth_end - lung_depth_start))
            lung_a = int(w * 0.15 * lung_scale)
            lung_b = int(h * 0.25 * lung_scale)

            for y in range(center_h - lung_b, center_h + lung_b):
                for x in range(right_lung_center_w - lung_a, right_lung_center_w + lung_a):
                    if (0 <= y < h and 0 <= x < w and
                        ((x - right_lung_center_w)**2 / lung_a**2 + (y - center_h)**2 / lung_b**2) <= 1):
                        volume[z, y, x] = 0.1  # Lung tissue

        # Add ribs (high density structures)
        for rib_idx in range(8):
            rib_z = lung_depth_start + rib_idx * (lung_depth_end - lung_depth_start) // 8
            rib_y = center_h + int(h * 0.2 * np.sin(2 * np.pi * rib_idx / 8))

            # Left rib
            for x in range(center_w - w // 3, center_w - w // 6):
                if 0 <= rib_y < h and 0 <= x < w:
                    volume[rib_z:rib_z+2, rib_y:rib_y+3, x] = 0.9

            # Right rib
            for x in range(center_w + w // 6, center_w + w // 3):
                if 0 <= rib_y < h and 0 <= x < w:
                    volume[rib_z:rib_z+2, rib_y:rib_y+3, x] = 0.9

        # Add heart
        heart_z_start = center_d - d // 8
        heart_z_end = center_d + d // 8
        heart_center_w = center_w - w // 12
        heart_center_h = center_h + h // 8

        for z in range(heart_z_start, heart_z_end):
            heart_a = int(w * 0.08)
            heart_b = int(h * 0.12)

            for y in range(heart_center_h - heart_b, heart_center_h + heart_b):
                for x in range(heart_center_w - heart_a, heart_center_w + heart_a):
                    if (0 <= y < h and 0 <= x < w and
                        ((x - heart_center_w)**2 / heart_a**2 + (y - heart_center_h)**2 / heart_b**2) <= 1):
                        volume[z, y, x] = 0.7  # Heart muscle

        # Add pathology (nodules) occasionally
        if np.random.random() < 0.3:
            nodule_z = np.random.randint(lung_depth_start, lung_depth_end)
            nodule_y = np.random.randint(center_h - h // 4, center_h + h // 4)
            nodule_x = np.random.choice([left_lung_center_w, right_lung_center_w]) + np.random.randint(-w//12, w//12)
            nodule_size = np.random.randint(3, 8)

            for z in range(max(0, nodule_z - nodule_size), min(d, nodule_z + nodule_size)):
                for y in range(max(0, nodule_y - nodule_size), min(h, nodule_y + nodule_size)):
                    for x in range(max(0, nodule_x - nodule_size), min(w, nodule_x + nodule_size)):
                        dist = np.sqrt((x - nodule_x)**2 + (y - nodule_y)**2 + (z - nodule_z)**2)
                        if dist < nodule_size:
                            volume[z, y, x] = 0.6  # Nodule

        # Add noise and smooth
        volume += np.random.normal(0, 0.02, volume.shape)
        volume = scipy.ndimage.gaussian_filter(volume, sigma=0.5)
        volume = np.clip(volume, 0, 1)

        return volume

    def _generate_mri_brain(self, seed: int) -> np.ndarray:
        """3D 뇌 MRI 볼륨 생성"""
        np.random.seed(seed)

        volume = np.zeros(self.volume_size)
        d, h, w = self.volume_size
        center_d, center_h, center_w = d // 2, h // 2, w // 2

        # Brain outline (ellipsoid)
        brain_a = w * 0.4  # Anterior-posterior
        brain_b = h * 0.45  # Superior-inferior
        brain_c = d * 0.35  # Left-right

        for z in range(d):
            for y in range(h):
                for x in range(w):
                    # Ellipsoid equation
                    dist = ((x - center_w)**2 / brain_a**2 +
                           (y - center_h)**2 / brain_b**2 +
                           (z - center_d)**2 / brain_c**2)

                    if dist <= 1:
                        volume[z, y, x] = 0.4  # Brain tissue

        # Add gray matter (cortex)
        gray_thickness = 5
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    dist = ((x - center_w)**2 / brain_a**2 +
                           (y - center_h)**2 / brain_b**2 +
                           (z - center_d)**2 / brain_c**2)

                    inner_dist = ((x - center_w)**2 / (brain_a - gray_thickness)**2 +
                                 (y - center_h)**2 / (brain_b - gray_thickness)**2 +
                                 (z - center_d)**2 / (brain_c - gray_thickness)**2)

                    if dist <= 1 and inner_dist > 1:
                        volume[z, y, x] = 0.6  # Gray matter

        # Add white matter
        white_a = brain_a - gray_thickness
        white_b = brain_b - gray_thickness
        white_c = brain_c - gray_thickness

        for z in range(d):
            for y in range(h):
                for x in range(w):
                    dist = ((x - center_w)**2 / white_a**2 +
                           (y - center_h)**2 / white_b**2 +
                           (z - center_d)**2 / white_c**2)

                    if dist <= 1:
                        volume[z, y, x] = 0.8  # White matter

        # Add ventricles
        ventricle_size = 8
        # Lateral ventricles
        for side in [-1, 1]:
            vent_z = center_d + side * d // 6
            vent_y = center_h - h // 6
            vent_x = center_w

            for z in range(max(0, vent_z - ventricle_size), min(d, vent_z + ventricle_size)):
                for y in range(max(0, vent_y - ventricle_size), min(h, vent_y + ventricle_size)):
                    for x in range(max(0, vent_x - ventricle_size//2), min(w, vent_x + ventricle_size//2)):
                        dist = np.sqrt((x - vent_x)**2 + (y - vent_y)**2 + (z - vent_z)**2)
                        if dist < ventricle_size:
                            volume[z, y, x] = 0.1  # CSF

        # Third ventricle
        third_vent_size = 4
        for y in range(center_h - third_vent_size, center_h + third_vent_size):
            for x in range(center_w - third_vent_size//2, center_w + third_vent_size//2):
                for z in range(center_d - third_vent_size//2, center_d + third_vent_size//2):
                    if 0 <= y < h and 0 <= x < w and 0 <= z < d:
                        volume[z, y, x] = 0.1  # Third ventricle

        # Add pathology (lesions, tumors)
        if np.random.random() < 0.4:
            lesion_type = np.random.choice(['tumor', 'stroke', 'ms_lesion'])

            if lesion_type == 'tumor':
                tumor_z = np.random.randint(d//4, 3*d//4)
                tumor_y = np.random.randint(h//4, 3*h//4)
                tumor_x = np.random.randint(w//4, 3*w//4)
                tumor_size = np.random.randint(8, 15)

                for z in range(max(0, tumor_z - tumor_size), min(d, tumor_z + tumor_size)):
                    for y in range(max(0, tumor_y - tumor_size), min(h, tumor_y + tumor_size)):
                        for x in range(max(0, tumor_x - tumor_size), min(w, tumor_x + tumor_size)):
                            dist = np.sqrt((x - tumor_x)**2 + (y - tumor_y)**2 + (z - tumor_z)**2)
                            if dist < tumor_size:
                                volume[z, y, x] = 0.3  # Tumor tissue

            elif lesion_type == 'ms_lesion':
                # Multiple small lesions
                for _ in range(np.random.randint(3, 8)):
                    lesion_z = np.random.randint(d//4, 3*d//4)
                    lesion_y = np.random.randint(h//4, 3*h//4)
                    lesion_x = np.random.randint(w//4, 3*w//4)
                    lesion_size = np.random.randint(2, 5)

                    for z in range(max(0, lesion_z - lesion_size), min(d, lesion_z + lesion_size)):
                        for y in range(max(0, lesion_y - lesion_size), min(h, lesion_y + lesion_size)):
                            for x in range(max(0, lesion_x - lesion_size), min(w, lesion_x + lesion_size)):
                                dist = np.sqrt((x - lesion_x)**2 + (y - lesion_y)**2 + (z - lesion_z)**2)
                                if dist < lesion_size:
                                    volume[z, y, x] = 0.9  # MS lesion

        # Add noise and smooth
        volume += np.random.normal(0, 0.03, volume.shape)
        volume = scipy.ndimage.gaussian_filter(volume, sigma=0.8)
        volume = np.clip(volume, 0, 1)

        return volume

    def _generate_ct_abdomen(self, seed: int) -> np.ndarray:
        """3D 복부 CT 볼륨 생성"""
        np.random.seed(seed)

        volume = np.ones(self.volume_size) * 0.2  # Air background
        d, h, w = self.volume_size
        center_d, center_h, center_w = d // 2, h // 2, w // 2

        # Body outline
        for z in range(d):
            body_scale = 0.9 + 0.1 * np.sin(2 * np.pi * z / d)
            a = int(w * 0.45 * body_scale)
            b = int(h * 0.4 * body_scale)

            for y in range(h):
                for x in range(w):
                    if ((x - center_w)**2 / a**2 + (y - center_h)**2 / b**2) <= 1:
                        volume[z, y, x] = 0.5  # Soft tissue

        # Add liver
        liver_z_start = d // 6
        liver_z_end = 2 * d // 3
        liver_center_w = center_w + w // 6
        liver_center_h = center_h - h // 8

        for z in range(liver_z_start, liver_z_end):
            liver_scale = 0.7 + 0.2 * np.sin(np.pi * (z - liver_z_start) / (liver_z_end - liver_z_start))
            liver_a = int(w * 0.15 * liver_scale)
            liver_b = int(h * 0.2 * liver_scale)

            for y in range(liver_center_h - liver_b, liver_center_h + liver_b):
                for x in range(liver_center_w - liver_a, liver_center_w + liver_a):
                    if (0 <= y < h and 0 <= x < w and
                        ((x - liver_center_w)**2 / liver_a**2 + (y - liver_center_h)**2 / liver_b**2) <= 1):
                        volume[z, y, x] = 0.7  # Liver tissue

        # Add kidneys
        for side in [-1, 1]:
            kidney_z_start = d // 3
            kidney_z_end = 2 * d // 3
            kidney_center_w = center_w + side * w // 4
            kidney_center_h = center_h + h // 6

            for z in range(kidney_z_start, kidney_z_end):
                kidney_a = int(w * 0.06)
                kidney_b = int(h * 0.1)

                for y in range(kidney_center_h - kidney_b, kidney_center_h + kidney_b):
                    for x in range(kidney_center_w - kidney_a, kidney_center_w + kidney_a):
                        if (0 <= y < h and 0 <= x < w and
                            ((x - kidney_center_w)**2 / kidney_a**2 + (y - kidney_center_h)**2 / kidney_b**2) <= 1):
                            volume[z, y, x] = 0.6  # Kidney tissue

        # Add spine
        spine_center_w = center_w
        spine_center_h = center_h + h // 3

        for z in range(d):
            spine_size = 6
            for y in range(spine_center_h - spine_size, spine_center_h + spine_size):
                for x in range(spine_center_w - spine_size, spine_center_w + spine_size):
                    if (0 <= y < h and 0 <= x < w and
                        ((x - spine_center_w)**2 + (y - spine_center_h)**2) <= spine_size**2):
                        volume[z, y, x] = 0.9  # Bone

        # Add pathology
        if np.random.random() < 0.3:
            # Liver lesion
            lesion_z = np.random.randint(liver_z_start, liver_z_end)
            lesion_y = np.random.randint(liver_center_h - liver_b//2, liver_center_h + liver_b//2)
            lesion_x = np.random.randint(liver_center_w - liver_a//2, liver_center_w + liver_a//2)
            lesion_size = np.random.randint(4, 10)

            for z in range(max(0, lesion_z - lesion_size), min(d, lesion_z + lesion_size)):
                for y in range(max(0, lesion_y - lesion_size), min(h, lesion_y + lesion_size)):
                    for x in range(max(0, lesion_x - lesion_size), min(w, lesion_x + lesion_size)):
                        dist = np.sqrt((x - lesion_x)**2 + (y - lesion_y)**2 + (z - lesion_z)**2)
                        if dist < lesion_size:
                            volume[z, y, x] = 0.3  # Lesion

        # Add noise and smooth
        volume += np.random.normal(0, 0.02, volume.shape)
        volume = scipy.ndimage.gaussian_filter(volume, sigma=0.5)
        volume = np.clip(volume, 0, 1)

        return volume

    def _generate_mri_cardiac(self, seed: int) -> np.ndarray:
        """3D 심장 MRI 볼륨 생성"""
        np.random.seed(seed)

        volume = np.zeros(self.volume_size)
        d, h, w = self.volume_size
        center_d, center_h, center_w = d // 2, h // 2, w // 2

        # Heart outline (approximate shape)
        heart_a = w * 0.3
        heart_b = h * 0.35
        heart_c = d * 0.25

        for z in range(d):
            for y in range(h):
                for x in range(w):
                    # Heart-like shape (modified ellipsoid)
                    dist = ((x - center_w)**2 / heart_a**2 +
                           (y - center_h + h*0.1)**2 / heart_b**2 +
                           (z - center_d)**2 / heart_c**2)

                    if dist <= 1:
                        volume[z, y, x] = 0.6  # Myocardium

        # Add ventricles (blood pools)
        # Left ventricle
        lv_a = heart_a * 0.4
        lv_b = heart_b * 0.4
        lv_c = heart_c * 0.6

        for z in range(d):
            for y in range(h):
                for x in range(w):
                    dist = ((x - center_w + w*0.05)**2 / lv_a**2 +
                           (y - center_h)**2 / lv_b**2 +
                           (z - center_d)**2 / lv_c**2)

                    if dist <= 1:
                        volume[z, y, x] = 0.1  # Blood

        # Right ventricle
        rv_a = heart_a * 0.35
        rv_b = heart_b * 0.3
        rv_c = heart_c * 0.5

        for z in range(d):
            for y in range(h):
                for x in range(w):
                    dist = ((x - center_w - w*0.08)**2 / rv_a**2 +
                           (y - center_h + h*0.05)**2 / rv_b**2 +
                           (z - center_d)**2 / rv_c**2)

                    if dist <= 1:
                        volume[z, y, x] = 0.1  # Blood

        # Add pathology (infarct, scar)
        if np.random.random() < 0.4:
            # Myocardial infarct
            infarct_z = np.random.randint(center_d - d//6, center_d + d//6)
            infarct_y = np.random.randint(center_h - h//6, center_h + h//6)
            infarct_x = np.random.randint(center_w - w//6, center_w + w//6)
            infarct_size = np.random.randint(3, 8)

            for z in range(max(0, infarct_z - infarct_size), min(d, infarct_z + infarct_size)):
                for y in range(max(0, infarct_y - infarct_size), min(h, infarct_y + infarct_size)):
                    for x in range(max(0, infarct_x - infarct_size), min(w, infarct_x + infarct_size)):
                        dist = np.sqrt((x - infarct_x)**2 + (y - infarct_y)**2 + (z - infarct_z)**2)
                        if dist < infarct_size and volume[z, y, x] > 0.5:  # Only in myocardium
                            volume[z, y, x] = 0.9  # Scar tissue

        # Add noise and smooth
        volume += np.random.normal(0, 0.02, volume.shape)
        volume = scipy.ndimage.gaussian_filter(volume, sigma=0.7)
        volume = np.clip(volume, 0, 1)

        return volume

    def _generate_generic_volume(self, seed: int) -> np.ndarray:
        """일반적인 3D 의료 볼륨 생성"""
        np.random.seed(seed)
        volume = np.random.normal(0.5, 0.2, self.volume_size)
        volume = scipy.ndimage.gaussian_filter(volume, sigma=1.0)
        volume = np.clip(volume, 0, 1)
        return volume

    def _generate_volume_condition(self, seed: int) -> Volume3DCondition:
        """볼륨 생성 조건 생성"""
        np.random.seed(seed)

        anatomies = ["chest", "abdomen", "head", "pelvis", "spine"]
        pathologies = ["normal", "tumor", "lesion", "fracture", "inflammation", "ischemia"]

        return Volume3DCondition(
            anatomy=np.random.choice(anatomies),
            pathology=np.random.choice(pathologies),
            age=np.random.random(),
            gender=np.random.randint(0, 2),
            contrast=np.random.choice([True, False]),
            slice_thickness=np.random.uniform(1.0, 5.0),
            resolution=(np.random.uniform(0.5, 2.0),
                       np.random.uniform(0.5, 2.0),
                       np.random.uniform(0.5, 2.0))
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert volume to tensor
        volume = torch.from_numpy(sample['volume']).float().unsqueeze(0)  # Add channel dimension

        # Normalize to [-1, 1]
        volume = volume * 2.0 - 1.0

        # Convert condition to tensor
        condition = sample['condition']
        condition_tensor = torch.tensor([
            condition.age,
            condition.gender,
            float(condition.contrast),
            condition.slice_thickness / 5.0,  # Normalize
            condition.resolution[0] / 2.0,     # Normalize
            condition.resolution[1] / 2.0,
            condition.resolution[2] / 2.0
        ], dtype=torch.float32)

        return {
            'volume': volume,
            'condition': condition_tensor,
            'volume_type': sample['volume_type'],
            'pathology': condition.pathology
        }

# 3D Diffusion U-Net for Volume Generation
class Medical3DDiffusionUNet(nn.Module):
    """3D 의료 볼륨을 위한 3D Diffusion U-Net"""

    def __init__(self, in_channels=1, out_channels=1, condition_dim=7,
                 base_channels=32, time_embed_dim=128):
        super(Medical3DDiffusionUNet, self).__init__()

        self.time_embed_dim = time_embed_dim
        self.condition_dim = condition_dim

        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Condition embedding
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, base_channels),
            nn.SiLU(),
            nn.Linear(base_channels, base_channels)
        )

        # 3D U-Net architecture
        self.down1 = self._make_3d_conv_block(in_channels, base_channels, time_embed_dim)
        self.down2 = self._make_3d_conv_block(base_channels, base_channels * 2, time_embed_dim)
        self.down3 = self._make_3d_conv_block(base_channels * 2, base_channels * 4, time_embed_dim)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU(),
            nn.Conv3d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )

        # Up path
        self.up3 = self._make_3d_conv_block(base_channels * 8, base_channels * 2, time_embed_dim)
        self.up2 = self._make_3d_conv_block(base_channels * 4, base_channels, time_embed_dim)
        self.up1 = self._make_3d_conv_block(base_channels * 2, base_channels, time_embed_dim)

        # Output
        self.output = nn.Conv3d(base_channels, out_channels, 1)

        # 3D pooling and upsampling
        self.pool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def _make_3d_conv_block(self, in_ch, out_ch, time_embed_dim):
        return nn.ModuleDict({
            'conv1': nn.Conv3d(in_ch, out_ch, 3, padding=1),
            'norm1': nn.GroupNorm(min(out_ch // 4, 8), out_ch),
            'conv2': nn.Conv3d(out_ch, out_ch, 3, padding=1),
            'norm2': nn.GroupNorm(min(out_ch // 4, 8), out_ch),
            'time_proj': nn.Linear(time_embed_dim, out_ch),
            'act': nn.SiLU()
        })

    def _apply_3d_conv_block(self, x, block, time_embed, condition_embed=None):
        # First conv
        h = block['conv1'](x)
        h = block['norm1'](h)

        # Add time embedding
        time_proj = block['time_proj'](time_embed)
        h = h + time_proj[:, :, None, None, None]

        # Add condition embedding if provided
        if condition_embed is not None:
            h = h + condition_embed[:, :, None, None, None]

        h = block['act'](h)

        # Second conv
        h = block['conv2'](h)
        h = block['norm2'](h)
        h = block['act'](h)

        return h

    def positional_encoding(self, timesteps, dim):
        """Sinusoidal positional encoding for time steps"""
        half_dim = dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def forward(self, x, timesteps, condition):
        # Time embedding
        time_embed = self.positional_encoding(timesteps, self.time_embed_dim)
        time_embed = self.time_embedding(time_embed)

        # Condition embedding
        condition_embed = self.condition_embedding(condition)

        # Encoder
        x1 = self._apply_3d_conv_block(x, self.down1, time_embed, condition_embed)
        x2 = self._apply_3d_conv_block(self.pool(x1), self.down2, time_embed)
        x3 = self._apply_3d_conv_block(self.pool(x2), self.down3, time_embed)

        # Bottleneck
        bottleneck = self.pool(x3)
        for layer in self.bottleneck:
            bottleneck = layer(bottleneck)

        # Decoder
        up3_upsampled = self.upsample(bottleneck)
        # Ensure spatial dimensions match
        if up3_upsampled.shape[-3:] != x3.shape[-3:]:
            up3_upsampled = F.interpolate(up3_upsampled, size=x3.shape[-3:], mode='trilinear', align_corners=True)
        up3 = self._apply_3d_conv_block(
            torch.cat([up3_upsampled, x3], dim=1),
            self.up3, time_embed
        )

        up2_upsampled = self.upsample(up3)
        if up2_upsampled.shape[-3:] != x2.shape[-3:]:
            up2_upsampled = F.interpolate(up2_upsampled, size=x2.shape[-3:], mode='trilinear', align_corners=True)
        up2 = self._apply_3d_conv_block(
            torch.cat([up2_upsampled, x2], dim=1),
            self.up2, time_embed
        )

        up1_upsampled = self.upsample(up2)
        if up1_upsampled.shape[-3:] != x1.shape[-3:]:
            up1_upsampled = F.interpolate(up1_upsampled, size=x1.shape[-3:], mode='trilinear', align_corners=True)
        up1 = self._apply_3d_conv_block(
            torch.cat([up1_upsampled, x1], dim=1),
            self.up1, time_embed
        )

        return self.output(up1)

class Medical3DDVAE(nn.Module):
    """3D Variational Autoencoder for Medical Volumes"""

    def __init__(self, in_channels=1, latent_dim=128, condition_dim=7):
        super(Medical3DDVAE, self).__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, 4, 2, 1),  # 32x32x32
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2),

            nn.Conv3d(32, 64, 4, 2, 1),  # 16x16x16
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),

            nn.Conv3d(64, 128, 4, 2, 1),  # 8x8x8
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),

            nn.Conv3d(128, 256, 4, 2, 1),  # 4x4x4
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool3d(1)  # 1x1x1
        )

        # Latent space
        self.fc_mu = nn.Linear(256 + condition_dim, latent_dim)
        self.fc_logvar = nn.Linear(256 + condition_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim + condition_dim, 256 * 4 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 4, 2, 1),  # 8x8x8
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 64, 4, 2, 1),  # 16x16x16
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, 4, 2, 1),  # 32x32x32
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, in_channels, 4, 2, 1),  # 64x64x64
            nn.Tanh()
        )

    def encode(self, x, condition):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        h = torch.cat([h, condition], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        h = torch.cat([z, condition], dim=1)
        h = self.fc_decode(h)
        h = h.view(h.size(0), 256, 4, 4, 4)
        return self.decoder(h)

    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar

    def sample(self, num_samples, condition, device):
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z, condition)


def vae_loss_function(recon_x, x, mu, logvar, beta: float = 1.0) -> torch.Tensor:
    """VAE loss combining reconstruction and KL divergence."""

    recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss


# ---------------------------------------------------------------------------
# Training strategies
# ---------------------------------------------------------------------------


class BaseSynthesisStrategy:
    """Common utilities for medical synthesis training strategies."""

    def __init__(
        self,
        batch_size: int,
        learning_rate: float,
        sample_interval: int = 10,
        dataset_kwargs: Optional[Dict] = None,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sample_interval = sample_interval
        self.dataset_kwargs = dataset_kwargs or {}
        self.device: Optional[torch.device] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.dataset_label: str = "unknown"

    # --- hooks to be implemented by subclasses ---------------------------------
    def create_dataset(self) -> Tuple[Dataset, str]:  # pragma: no cover - abstract
        raise NotImplementedError

    def initialize_models(self, device: torch.device) -> None:  # pragma: no cover
        raise NotImplementedError

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        raise NotImplementedError

    def get_primary_model(self):  # pragma: no cover - abstract
        raise NotImplementedError

    # --- lifecycle helpers ------------------------------------------------------
    def setup(self, device: torch.device) -> None:
        """Prepare dataset, dataloaders and models."""

        self.device = device
        dataset, dataset_label = self.create_dataset()
        self.dataset_label = dataset_label

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        self.initialize_models(device)

    def on_epoch_start(self, epoch: int) -> None:
        """Optional hook executed at the beginning of each epoch."""
        # Subclasses may override to reset buffers

    def summarize_epoch(self, avg_loss: float, extra_metrics: Dict[str, float]) -> Dict:
        """Return metrics for logging."""
        return {"train_loss": avg_loss, "val_loss": None, "extra": {}}

    def generate_samples(self) -> Optional[Dict]:
        """Return information about samples to log."""
        return None

    def on_epoch_end(self, epoch: int, logger) -> None:
        if self.sample_interval and (epoch + 1) % self.sample_interval == 0:
            sample_info = self.generate_samples()
            if not sample_info:
                return

            filename = sample_info.get("filename") or f"generated_samples_epoch_{epoch + 1:03d}"

            if sample_info.get("type") == "images":
                logger.save_image_grid(
                    sample_info["data"],
                    filename,
                    titles=sample_info.get("titles"),
                    nrow=sample_info.get("nrow", 4),
                    cmap=sample_info.get("cmap", "gray"),
                )
            elif sample_info.get("type") == "volume":
                logger.save_numpy_array(
                    sample_info["data"],
                    filename,
                    description=sample_info.get("description"),
                )


class Diffusion2DStrategy(BaseSynthesisStrategy):
    """Diffusion-based strategy for 2D medical images."""

    def __init__(self, **kwargs):
        dataset_kwargs = kwargs.pop("dataset_kwargs", {})
        super().__init__(
            batch_size=kwargs.get("batch_size", 8),
            learning_rate=kwargs.get("learning_rate", 0.0002),
            sample_interval=kwargs.get("sample_interval", 10),
            dataset_kwargs=dataset_kwargs,
        )
        self.modality: MedicalModality = dataset_kwargs.get(
            "modality", MedicalModality.CHEST_XRAY
        )
        self.image_size: int = dataset_kwargs.get("image_size", 256)
        self.num_samples: int = dataset_kwargs.get("num_samples", 800)
        self.model: Optional[MedicalDDPM] = None
        self.optimizer: Optional[optim.Optimizer] = None

    def create_dataset(self) -> Tuple[Dataset, str]:
        dataset = Medical2DDataset(
            modality=self.modality,
            num_samples=self.num_samples,
            image_size=self.image_size,
        )
        return dataset, self.modality.value

    def initialize_models(self, device: torch.device) -> None:
        unet = MedicalDiffusionUNet(in_channels=1, out_channels=1, condition_dim=6)
        self.model = MedicalDDPM(unet, num_timesteps=1000).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def on_epoch_start(self, epoch: int) -> None:
        if self.model:
            self.model.train()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        images = batch["image"].to(self.device)
        conditions = batch["condition"].to(self.device)
        t = torch.randint(0, self.model.num_timesteps, (images.shape[0],), device=self.device)

        loss = self.model.p_losses(images, t, conditions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def generate_samples(self) -> Optional[Dict]:
        if not self.model or not self.val_loader:
            return None
        self.model.eval()
        with torch.no_grad():
            val_batch = next(iter(self.val_loader))
            val_conditions = val_batch["condition"][:4].to(self.device)
            samples = self.model.sample(
                (val_conditions.size(0), 1, self.image_size, self.image_size),
                val_conditions,
                self.device,
            )
            samples = (samples + 1) / 2
            sample_images = [samples[i].cpu().squeeze().numpy() for i in range(samples.size(0))]
        return {
            "type": "images",
            "data": sample_images,
            "titles": [f"Sample {i + 1}" for i in range(len(sample_images))],
            "nrow": 2,
        }

    def get_primary_model(self):
        return self.model


class StyleGAN2DStrategy(BaseSynthesisStrategy):
    """StyleGAN-based strategy for 2D medical images."""

    def __init__(self, **kwargs):
        dataset_kwargs = kwargs.pop("dataset_kwargs", {})
        super().__init__(
            batch_size=kwargs.get("batch_size", 8),
            learning_rate=kwargs.get("learning_rate", 0.0002),
            sample_interval=kwargs.get("sample_interval", 10),
            dataset_kwargs=dataset_kwargs,
        )
        self.modality: MedicalModality = dataset_kwargs.get(
            "modality", MedicalModality.CHEST_XRAY
        )
        self.image_size: int = dataset_kwargs.get("image_size", 256)
        self.num_samples: int = dataset_kwargs.get("num_samples", 800)
        self.latent_dim: int = kwargs.get("latent_dim", 512)
        self.generator: Optional[MedicalStyleGenerator] = None
        self.discriminator: Optional[nn.Module] = None
        self.g_optimizer: Optional[optim.Optimizer] = None
        self.d_optimizer: Optional[optim.Optimizer] = None
        self.criterion = nn.BCELoss()

    def create_dataset(self) -> Tuple[Dataset, str]:
        dataset = Medical2DDataset(
            modality=self.modality,
            num_samples=self.num_samples,
            image_size=self.image_size,
        )
        return dataset, self.modality.value

    def initialize_models(self, device: torch.device) -> None:
        self.generator = MedicalStyleGenerator(
            latent_dim=self.latent_dim,
            condition_dim=6,
            img_size=self.image_size,
            img_channels=1,
        ).to(device)
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid(),
        ).to(device)

        self.g_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999)
        )

    def on_epoch_start(self, epoch: int) -> None:
        if self.generator:
            self.generator.train()
        if self.discriminator:
            self.discriminator.train()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        real_images = batch["image"].to(self.device)
        conditions = batch["condition"].to(self.device)
        batch_size = real_images.size(0)

        # Train discriminator
        self.d_optimizer.zero_grad()
        real_labels = torch.ones(batch_size, 1, 1, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, 1, 1, device=self.device)

        real_output = self.discriminator(real_images)
        d_real_loss = self.criterion(real_output, real_labels)

        latent = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(latent, conditions)
        fake_output = self.discriminator(fake_images.detach())
        d_fake_loss = self.criterion(fake_output, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        self.d_optimizer.step()

        # Train generator
        self.g_optimizer.zero_grad()
        fake_output = self.discriminator(fake_images)
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()

        return {
            "loss": g_loss.item(),
            "metrics": {
                "generator_loss": g_loss.item(),
                "discriminator_loss": d_loss.item(),
            },
        }

    def summarize_epoch(self, avg_loss: float, extra_metrics: Dict[str, float]) -> Dict:
        train_loss = extra_metrics.get("generator_loss", avg_loss)
        val_loss = extra_metrics.get("discriminator_loss")
        return {"train_loss": train_loss, "val_loss": val_loss, "extra": {}}

    def generate_samples(self) -> Optional[Dict]:
        if not self.generator or not self.val_loader:
            return None
        self.generator.eval()
        with torch.no_grad():
            val_batch = next(iter(self.val_loader))
            val_conditions = val_batch["condition"][:4].to(self.device)
            latent = torch.randn(val_conditions.size(0), self.latent_dim, device=self.device)
            samples = self.generator(latent, val_conditions)
            samples = (samples + 1) / 2
            sample_images = [samples[i].cpu().squeeze().numpy() for i in range(samples.size(0))]
        return {
            "type": "images",
            "data": sample_images,
            "titles": [f"Sample {i + 1}" for i in range(len(sample_images))],
            "nrow": 2,
        }

    def get_primary_model(self):
        return self.generator


class VAE3DStrategy(BaseSynthesisStrategy):
    """Variational autoencoder strategy for 3D medical volumes."""

    def __init__(self, **kwargs):
        dataset_kwargs = kwargs.pop("dataset_kwargs", {})
        super().__init__(
            batch_size=kwargs.get("batch_size", 2),
            learning_rate=kwargs.get("learning_rate", 0.001),
            sample_interval=kwargs.get("sample_interval", 10),
            dataset_kwargs=dataset_kwargs,
        )
        self.volume_type: VolumeType = dataset_kwargs.get("volume_type", VolumeType.CT_CHEST)
        self.volume_size: Tuple[int, int, int] = dataset_kwargs.get("volume_size", (32, 32, 32))
        self.num_samples: int = dataset_kwargs.get("num_samples", 100)
        self.model: Optional[Medical3DDVAE] = None
        self.optimizer: Optional[optim.Optimizer] = None

    def create_dataset(self) -> Tuple[Dataset, str]:
        dataset = Medical3DDataset(
            volume_type=self.volume_type,
            num_samples=self.num_samples,
            volume_size=self.volume_size,
        )
        return dataset, self.volume_type.value

    def initialize_models(self, device: torch.device) -> None:
        self.model = Medical3DDVAE(in_channels=1, latent_dim=64, condition_dim=7).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def on_epoch_start(self, epoch: int) -> None:
        if self.model:
            self.model.train()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        volumes = batch["volume"].to(self.device)
        conditions = batch["condition"].to(self.device)

        self.optimizer.zero_grad()
        recon_volumes, mu, logvar = self.model(volumes, conditions)
        loss = vae_loss_function(recon_volumes, volumes, mu, logvar, beta=0.5)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def generate_samples(self) -> Optional[Dict]:
        if not self.model or not self.val_loader:
            return None
        self.model.eval()
        with torch.no_grad():
            val_batch = next(iter(self.val_loader))
            val_conditions = val_batch["condition"][:1].to(self.device)
            sample_volume = self.model.sample(1, val_conditions, self.device)
            sample_np = sample_volume[0, 0].cpu().numpy()
        return {
            "type": "volume",
            "data": sample_np,
            "description": f"Sample volume for {self.volume_type.value}",
        }

    def get_primary_model(self):
        return self.model


class Diffusion3DStrategy(BaseSynthesisStrategy):
    """Simplified diffusion strategy for 3D medical volumes."""

    def __init__(self, **kwargs):
        dataset_kwargs = kwargs.pop("dataset_kwargs", {})
        super().__init__(
            batch_size=kwargs.get("batch_size", 2),
            learning_rate=kwargs.get("learning_rate", 0.001),
            sample_interval=kwargs.get("sample_interval", 10),
            dataset_kwargs=dataset_kwargs,
        )
        self.volume_type: VolumeType = dataset_kwargs.get("volume_type", VolumeType.CT_CHEST)
        self.volume_size: Tuple[int, int, int] = dataset_kwargs.get("volume_size", (32, 32, 32))
        self.num_samples: int = dataset_kwargs.get("num_samples", 100)
        self.model: Optional[Medical3DDiffusionUNet] = None
        self.optimizer: Optional[optim.Optimizer] = None

    def create_dataset(self) -> Tuple[Dataset, str]:
        dataset = Medical3DDataset(
            volume_type=self.volume_type,
            num_samples=self.num_samples,
            volume_size=self.volume_size,
        )
        return dataset, self.volume_type.value

    def initialize_models(self, device: torch.device) -> None:
        self.model = Medical3DDiffusionUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=7,
            base_channels=16,
            time_embed_dim=64,
        ).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

    def on_epoch_start(self, epoch: int) -> None:
        if self.model:
            self.model.train()

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict:
        volumes = batch["volume"].to(self.device)
        conditions = batch["condition"].to(self.device)

        noise = torch.randn_like(volumes)
        timesteps = torch.randint(0, 1000, (volumes.shape[0],), device=self.device)
        noisy_volumes = volumes + noise * 0.1

        self.optimizer.zero_grad()
        predicted_noise = self.model(noisy_volumes, timesteps, conditions)
        loss = F.mse_loss(predicted_noise, noise)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def generate_samples(self) -> Optional[Dict]:
        if not self.model or not self.val_loader:
            return None
        self.model.eval()
        with torch.no_grad():
            sample_volume = torch.randn((1, 1) + tuple(self.volume_size), device=self.device)
            val_batch = next(iter(self.val_loader))
            val_conditions = val_batch["condition"][:1].to(self.device)
            timesteps = torch.zeros(1, device=self.device, dtype=torch.long)
            generated = self.model(sample_volume, timesteps, val_conditions)
            sample_np = generated[0, 0].cpu().numpy()
        return {
            "type": "volume",
            "data": sample_np,
            "description": f"Diffusion sample for {self.volume_type.value}",
        }

    def get_primary_model(self):
        return self.model

# ---------------------------------------------------------------------------
# Strategy registry and training pipeline
# ---------------------------------------------------------------------------


STRATEGY_REGISTRY: Dict[Tuple[str, str], type] = {
    ("2d", "diffusion"): Diffusion2DStrategy,
    ("2d", "stylegan"): StyleGAN2DStrategy,
    ("3d", "vae"): VAE3DStrategy,
    ("3d", "diffusion"): Diffusion3DStrategy,
}


class MedicalSynthesisPipeline:
    """High-level orchestration for medical synthesis experiments."""

    def __init__(
        self,
        data_dim: str,
        model_type: str,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        sample_interval: int = 10,
        dataset_kwargs: Optional[Dict] = None,
        strategy_kwargs: Optional[Dict] = None,
        log_interval: int = 10,
    ):
        key = (data_dim.lower(), model_type.lower())
        if key not in STRATEGY_REGISTRY:
            valid = ", ".join(sorted({f"{k[0]}:{k[1]}" for k in STRATEGY_REGISTRY}))
            raise ValueError(f"Unsupported configuration {key}. Available: {valid}")

        strategy_kwargs = dict(strategy_kwargs or {})
        strategy_kwargs.update(
            {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "sample_interval": sample_interval,
                "dataset_kwargs": dataset_kwargs or {},
            }
        )

        self.data_dim = data_dim.lower()
        self.model_type = model_type.lower()
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.strategy: BaseSynthesisStrategy = STRATEGY_REGISTRY[key](**strategy_kwargs)
        self.logger = None
        self.device: Optional[torch.device] = None

    def train(self) -> nn.Module:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.strategy.setup(self.device)

        experiment_name = f"{self.data_dim}_{self.model_type}"
        self.logger = create_logger_for_medical_synthesis(
            experiment_name, self.strategy.dataset_label
        )
        self.logger.log(f"Using device: {self.device}")
        self.logger.log(
            f"Training {self.model_type} for {self.strategy.dataset_label}"
        )

        train_size = len(self.strategy.train_loader.dataset)
        val_size = len(self.strategy.val_loader.dataset)
        self.logger.log(f"Training samples: {train_size}, Validation samples: {val_size}")

        for epoch in range(self.num_epochs):
            self.strategy.on_epoch_start(epoch)
            epoch_losses: List[float] = []
            extra_metrics: Dict[str, List[float]] = {}

            for batch_idx, batch in enumerate(self.strategy.train_loader):
                step_result = self.strategy.training_step(batch, batch_idx)
                epoch_losses.append(step_result["loss"])

                for key, value in step_result.get("metrics", {}).items():
                    extra_metrics.setdefault(key, []).append(value)

                if batch_idx % self.log_interval == 0:
                    details = step_result.get("metrics") or {"loss": step_result["loss"]}
                    detail_str = ", ".join(f"{k}: {v:.6f}" for k, v in details.items())
                    self.logger.log(
                        f"Epoch {epoch + 1}/{self.num_epochs}, Batch {batch_idx}, {detail_str}"
                    )

            avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            averaged_metrics = {
                key: float(np.mean(values)) for key, values in extra_metrics.items() if values
            }
            summary = self.strategy.summarize_epoch(avg_loss, averaged_metrics)
            self.logger.log(f"Epoch {epoch + 1}/{self.num_epochs} completed")

            log_kwargs = dict(averaged_metrics)
            log_kwargs.update(summary.get("extra", {}))
            self.logger.log_metrics(
                epoch + 1,
                summary.get("train_loss", avg_loss),
                summary.get("val_loss"),
                **log_kwargs,
            )

            self.strategy.on_epoch_end(epoch, self.logger)

        self.logger.log("Medical synthesis training completed!")
        return self.strategy.get_primary_model()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_medical_2d_synthesis(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_type: str = "diffusion",
) -> Dict[str, float]:
    """Evaluate 2D medical synthesis model."""

    def calculate_medical_metrics(real_images, generated_images):
        mse = torch.mean((real_images - generated_images) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))

        def ssim(img1, img2, window_size=11):
            mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
            mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2

            sigma1_sq = (
                F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size // 2) - mu1_sq
            )
            sigma2_sq = (
                F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size // 2) - mu2_sq
            )
            sigma12 = (
                F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2)
                - mu1_mu2
            )

            c1 = 0.01 ** 2
            c2 = 0.03 ** 2

            ssim_map = (
                (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
                / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
            )
            return ssim_map.mean()

        ssim_score = ssim(real_images, generated_images)
        return psnr.item(), ssim_score.item()

    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            real_images = batch["image"].to(device)
            conditions = batch["condition"].to(device)

            if model_type == "diffusion":
                generated = model.sample(real_images.shape, conditions, device)
            else:
                latent = torch.randn(real_images.size(0), 512, device=device)
                generated = model(latent, conditions)

            real_norm = (real_images + 1) / 2
            gen_norm = (generated + 1) / 2

            psnr, ssim = calculate_medical_metrics(real_norm, gen_norm)
            total_psnr += psnr
            total_ssim += ssim
            num_batches += 1

    return {
        "psnr": total_psnr / max(num_batches, 1),
        "ssim": total_ssim / max(num_batches, 1),
    }


def evaluate_medical_3d_synthesis(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    model_type: str = "vae",
) -> Dict[str, float]:
    """Evaluate 3D medical synthesis models."""

    def calculate_3d_metrics(real_volumes, generated_volumes):
        mse = torch.mean((real_volumes - generated_volumes) ** 2)
        psnr = 20 * torch.log10(2.0 / torch.sqrt(mse))

        real_flat = real_volumes.view(real_volumes.size(0), -1)
        gen_flat = generated_volumes.view(generated_volumes.size(0), -1)
        cos_sim = F.cosine_similarity(real_flat, gen_flat, dim=1).mean()
        return psnr.item(), cos_sim.item()

    model.eval()
    total_psnr = 0.0
    total_similarity = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            real_volumes = batch["volume"].to(device)
            conditions = batch["condition"].to(device)

            if model_type == "vae":
                generated, _, _ = model(real_volumes, conditions)
            else:
                # Simplified for diffusion
                generated = torch.randn_like(real_volumes)

            psnr, similarity = calculate_3d_metrics(real_volumes, generated)
            total_psnr += psnr
            total_similarity += similarity
            num_batches += 1

    return {
        "psnr_3d": total_psnr / max(num_batches, 1),
        "volume_similarity": total_similarity / max(num_batches, 1),
    }


def evaluate_medical_synthesis(
    model: nn.Module,
    data_dim: str,
    test_loader: DataLoader,
    device: torch.device,
    model_type: str,
) -> Dict[str, float]:
    """Unified evaluation entry point."""

    if data_dim.lower() == "2d":
        return evaluate_medical_2d_synthesis(model, test_loader, device, model_type)
    return evaluate_medical_3d_synthesis(model, test_loader, device, model_type)


# ---------------------------------------------------------------------------
# Utility visualisations
# ---------------------------------------------------------------------------


def visualize_3d_volume(volume: np.ndarray, title: str = "3D Medical Volume", num_slices: int = 9):
    """Visualize a 3D volume by plotting multiple slices."""

    depth = volume.shape[0]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)

    slice_indices = np.linspace(0, depth - 1, num_slices, dtype=int)

    for i, slice_idx in enumerate(slice_indices):
        row, col = divmod(i, 3)
        axes[row, col].imshow(volume[slice_idx], cmap="gray")
        axes[row, col].set_title(f"Slice {slice_idx}")
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# High level API
# ---------------------------------------------------------------------------


def train_medical_synthesis(
    data_dim: str = "2d",
    model_type: str = "diffusion",
    *,
    num_epochs: int = 20,
    batch_size: int = 4,
    learning_rate: float = 0.0002,
    sample_interval: int = 10,
    image_size: int = 128,
    volume_size: Tuple[int, int, int] = (32, 32, 32),
    modality: str = MedicalModality.CHEST_XRAY.value,
    volume_type: str = VolumeType.CT_CHEST.value,
    dataset_kwargs: Optional[Dict] = None,
    strategy_kwargs: Optional[Dict] = None,
    log_interval: int = 10,
) -> Tuple[nn.Module, MedicalSynthesisPipeline]:
    """Configure and run a medical synthesis experiment."""

    dataset_kwargs = dict(dataset_kwargs or {})
    if data_dim.lower() == "2d":
        modality_enum = modality if isinstance(modality, MedicalModality) else MedicalModality(modality)
        dataset_kwargs.setdefault("modality", modality_enum)
        dataset_kwargs.setdefault("image_size", image_size)
        dataset_kwargs.setdefault("num_samples", dataset_kwargs.get("num_samples", 800))
    else:
        volume_type_enum = volume_type if isinstance(volume_type, VolumeType) else VolumeType(volume_type)
        dataset_kwargs.setdefault("volume_type", volume_type_enum)
        dataset_kwargs.setdefault("volume_size", volume_size)
        dataset_kwargs.setdefault("num_samples", dataset_kwargs.get("num_samples", 100))

    pipeline = MedicalSynthesisPipeline(
        data_dim=data_dim,
        model_type=model_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        sample_interval=sample_interval,
        dataset_kwargs=dataset_kwargs,
        strategy_kwargs=strategy_kwargs,
        log_interval=log_interval,
    )

    model = pipeline.train()
    return model, pipeline


def build_test_loader(
    data_dim: str,
    batch_size: int,
    image_size: int,
    volume_size: Tuple[int, int, int],
    modality: MedicalModality,
    volume_type: VolumeType,
) -> DataLoader:
    if data_dim.lower() == "2d":
        dataset = Medical2DDataset(modality=modality, num_samples=128, image_size=image_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataset = Medical3DDataset(volume_type=volume_type, num_samples=32, volume_size=volume_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Unified medical synthesis runner")
    parser.add_argument("--data-dim", choices=["2d", "3d"], default="2d")
    parser.add_argument(
        "--model-type",
        choices=["diffusion", "stylegan", "vae"],
        default="diffusion",
        help="Model family to use",
    )
    parser.add_argument("--modality", default=MedicalModality.CHEST_XRAY.value)
    parser.add_argument("--volume-type", default=VolumeType.CT_CHEST.value)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--sample-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument(
        "--volume-size",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        default=(32, 32, 32),
    )
    parser.add_argument("--latent-dim", type=int, default=512, help="Latent size for StyleGAN")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dim = args.data_dim.lower()
    model_type = args.model_type.lower()

    if data_dim == "2d" and model_type not in {"diffusion", "stylegan"}:
        raise ValueError("For 2D synthesis choose 'diffusion' or 'stylegan'.")
    if data_dim == "3d" and model_type not in {"diffusion", "vae"}:
        raise ValueError("For 3D synthesis choose 'diffusion' or 'vae'.")

    modality = MedicalModality(args.modality)
    volume_type = VolumeType(args.volume_type)
    volume_size = tuple(args.volume_size)

    strategy_kwargs = {}
    if model_type == "stylegan":
        strategy_kwargs["latent_dim"] = args.latent_dim

    model, pipeline = train_medical_synthesis(
        data_dim=data_dim,
        model_type=model_type,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sample_interval=args.sample_interval,
        image_size=args.image_size,
        volume_size=volume_size,
        modality=modality,
        volume_type=volume_type,
        strategy_kwargs=strategy_kwargs,
        log_interval=args.log_interval,
    )

    test_loader = build_test_loader(
        data_dim,
        args.batch_size,
        args.image_size,
        volume_size,
        modality,
        volume_type,
    )
    metrics = evaluate_medical_synthesis(
        model,
        data_dim,
        test_loader,
        pipeline.device,
        model_type,
    )

    print("\n📊 Evaluation metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")

    if data_dim == "3d":
        pipeline.strategy.get_primary_model().eval()
        with torch.no_grad():
            condition = torch.tensor([[0.5, 1, 1, 0.2, 0.5, 0.5, 0.5]], device=pipeline.device)
            if model_type == "vae":
                sample = pipeline.strategy.get_primary_model().sample(1, condition, pipeline.device)
            else:
                sample = torch.randn((1, 1) + volume_size, device=pipeline.device)
                sample = pipeline.strategy.get_primary_model()(sample, torch.zeros(1, device=pipeline.device, dtype=torch.long), condition)
            volume_np = sample[0, 0].cpu().numpy()
            visualize_3d_volume(volume_np, f"Sample {volume_type.value}")
            plt.savefig("generated_3d_volume_sample.png", dpi=150, bbox_inches="tight")
            plt.close()


if __name__ == "__main__":
    main()
