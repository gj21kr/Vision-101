"""
Diffusion Model (Simplified DDPM) Implementation

Diffusion Model은 점진적 노이즈 추가/제거 과정을 통해 데이터를 생성하는 모델입니다.
Ho et al. (2020)의 DDPM을 기반으로 한 간단한 구현을 제공합니다.

핵심 아이디어:
1. **Forward Process (Diffusion)**:
   - 데이터에 점진적으로 가우시안 노이즈를 추가
   - T 스텝에 걸쳐 데이터를 순수 노이즈로 변환
   - q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)

2. **Reverse Process (Denoising)**:
   - 신경망이 노이즈 제거 과정을 학습
   - 순수 노이즈에서 시작해 점진적으로 데이터 복원
   - p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t² I)

3. **Training Objective**:
   - 각 타임스텝에서 추가된 노이즈를 예측하도록 학습
   - L_simple = E_t,x_0,ε [||ε - ε_θ(√(ᾱ_t)x_0 + √(1-ᾱ_t)ε, t)||²]

주요 특징:
- 고품질 이미지 생성 (FID, IS 등에서 우수한 성능)
- 안정적인 훈련 과정 (GAN 대비)
- 다양한 응용 가능 (inpainting, super-resolution 등)
- 계산 비용이 높음 (1000 스텝 샘플링 필요)

이 구현은 교육 목적으로 단순화된 버전입니다.
실제 DDPM의 완전한 구현은 diffusion_variants/ddpm_example.py를 참조하세요.

Reference:
- Ho, J., Jain, A., & Abbeel, P. (2020).
  "Denoising diffusion probabilistic models."
  Neural Information Processing Systems (NeurIPS).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, time_emb_dim=32):
        super(SimpleUNet, self).__init__()
        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

        # Down path
        self.down1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down2 = nn.Conv2d(64, 128, 3, padding=1)
        self.down3 = nn.Conv2d(128, 256, 3, padding=1)

        # Bottleneck
        self.bottleneck = nn.Conv2d(256, 512, 3, padding=1)

        # Up path
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(256 + 256, 128, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(128 + 128, 64, 2, stride=2)

        # Output
        self.out = nn.Conv2d(64 + 64, out_channels, 1)

        self.pool = nn.MaxPool2d(2)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_emb_dim)
        t = self.time_embed(t)

        # Encoder
        skip1 = F.relu(self.down1(x))
        x1 = self.pool(skip1)

        skip2 = F.relu(self.down2(x1))
        x2 = self.pool(skip2)

        skip3 = F.relu(self.down3(x2))
        x3 = self.pool(skip3)

        # Bottleneck
        x = F.relu(self.bottleneck(x3))

        # Decoder
        x = self.up1(x)
        x = torch.cat([x, skip3], dim=1)

        x = self.up2(x)
        x = torch.cat([x, skip2], dim=1)

        x = self.up3(x)
        x = torch.cat([x, skip1], dim=1)

        return self.out(x)

class DiffusionModel:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Precompute noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, model, x, t):
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

def train_diffusion():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 32
    epochs = 5
    lr = 2e-4

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and diffusion
    model = SimpleUNet().to(device)
    diffusion = DiffusionModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)

            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (data.shape[0],)).long().to(device)

            # Sample noise
            noise = torch.randn_like(data)

            # Forward diffusion (add noise)
            x_noisy = diffusion.q_sample(data, t, noise=noise)

            # Predict noise
            predicted_noise = model(x_noisy, t)

            # Compute loss
            loss = F.mse_loss(predicted_noise, noise)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

    return model, diffusion

def sample_images(model, diffusion, num_samples=16):
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Start from random noise
        img = torch.randn(num_samples, 1, 28, 28, device=device)

        # Reverse diffusion process
        for i in reversed(range(diffusion.timesteps)):
            t = torch.full((num_samples,), i, dtype=torch.long, device=device)
            img = diffusion.p_sample(model, img, t)

            if i % 100 == 0:
                print(f'Sampling step {i}')

        # Denormalize
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)

        # Visualize
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(img[i].squeeze().cpu(), cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Diffusion Model Example - DDPM (Denoising Diffusion Probabilistic Models)")
    print("This example demonstrates a simple diffusion model for generating MNIST-like images.")
    print("Features:")
    print("- Forward diffusion process (adding noise)")
    print("- Reverse diffusion process (denoising)")
    print("- U-Net architecture for noise prediction")
    print("- Timestep conditioning")

    # Uncomment to train and generate samples
    # model, diffusion = train_diffusion()
    # sample_images(model, diffusion)