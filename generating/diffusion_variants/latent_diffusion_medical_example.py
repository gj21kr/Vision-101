"""
Medical Latent Diffusion Model (LDM) Implementation

의료 이미지 생성을 위한 Latent Diffusion Model 구현으로, VAE의 잠재 공간에서
diffusion을 수행하여 계산 효율성을 크게 향상시킵니다.

의료 특화 기능:
- 의료 이미지 특화 데이터 로더 (chest X-ray, brain MRI, skin lesion)
- 자동 결과 저장 및 로깅 시스템
- Two-stage training (VAE + Diffusion)
- 고해상도 의료 이미지 생성 가능
- 메모리 효율적인 훈련 및 추론

LDM의 핵심 아이디어:
1. Two-Stage Training:
   - Stage 1: VAE 훈련 (이미지 ↔ 잠재 표현)
   - Stage 2: 잠재 공간에서 Diffusion 모델 훈련
2. Perceptually Equivalent Latent Space:
   - 높은 압축률로 메모리 효율성
   - 의미적으로 중요한 정보만 보존
3. Computational Efficiency:
   - 16-64배 적은 메모리 사용
   - 고해상도 의료 이미지 생성 가능

의료 분야에서의 활용:
- 고해상도 의료 이미지 생성 (512×512 이상)
- 메모리 제약이 있는 환경에서의 훈련
- 실시간 의료 이미지 생성
- 의료 이미지 편집 및 인페인팅

Reference:
- Rombach, R., et al. (2022).
  "High-resolution image synthesis with latent diffusion models."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import math
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from medical_data_utils import MedicalImageLoader, load_chest_xray_data, load_brain_mri_data
from result_logger import create_logger_for_generating

# Import basic components
from ddpm_medical_example import TimeEmbedding, ResidualBlock, AttentionBlock

class MedicalVAEEncoder(nn.Module):
    """의료 이미지를 위한 VAE-style encoder"""
    def __init__(self, in_channels=1, latent_channels=4, base_channels=128):
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Downsampling path - 의료 이미지의 계층적 특징 추출
        self.down_blocks = nn.ModuleList([
            # 64x64 -> 32x32
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 2),
                nn.SiLU(),
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
                nn.GroupNorm(8, base_channels * 2),
                nn.SiLU(),
            ),
            # 32x32 -> 16x16
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 4),
                nn.SiLU(),
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                nn.GroupNorm(8, base_channels * 4),
                nn.SiLU(),
            ),
            # 16x16 -> 8x8
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 8),
                nn.SiLU(),
                nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
                nn.GroupNorm(8, base_channels * 8),
                nn.SiLU(),
            ),
        ])

        # Middle block - 의료 이미지의 글로벌 특징 처리
        self.mid_block = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU(),
            AttentionBlock(base_channels * 8),  # Self-attention for better feature representation
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU(),
        )

        # Output layers for mean and logvar
        self.conv_out = nn.Conv2d(base_channels * 8, latent_channels * 2, 3, padding=1)

    def forward(self, x):
        """
        의료 이미지를 잠재 공간으로 인코딩

        Args:
            x: 입력 의료 이미지 [B, C, H, W]

        Returns:
            mean, logvar: VAE의 평균과 log 분산
        """
        x = self.conv_in(x)

        # Downsampling
        for down_block in self.down_blocks:
            x = down_block(x)

        # Middle
        x = self.mid_block(x)

        # Output - 평균과 분산 추출
        moments = self.conv_out(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        return mean, logvar

class MedicalVAEDecoder(nn.Module):
    """의료 이미지를 위한 VAE-style decoder"""
    def __init__(self, latent_channels=4, out_channels=1, base_channels=128):
        super().__init__()

        # Input layer
        self.conv_in = nn.Conv2d(latent_channels, base_channels * 8, 3, padding=1)

        # Middle block
        self.mid_block = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU(),
            AttentionBlock(base_channels * 8),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
            nn.GroupNorm(8, base_channels * 8),
            nn.SiLU(),
        )

        # Upsampling blocks - 의료 이미지 복원
        self.up_blocks = nn.ModuleList([
            # 8x8 -> 16x16
            nn.Sequential(
                nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
                nn.GroupNorm(8, base_channels * 8),
                nn.SiLU(),
                nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 4),
                nn.SiLU(),
            ),
            # 16x16 -> 32x32
            nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                nn.GroupNorm(8, base_channels * 4),
                nn.SiLU(),
                nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 2),
                nn.SiLU(),
            ),
            # 32x32 -> 64x64
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
                nn.GroupNorm(8, base_channels * 2),
                nn.SiLU(),
                nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
                nn.GroupNorm(8, base_channels),
                nn.SiLU(),
            ),
        ])

        # Output layer
        self.conv_out = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh()  # 의료 이미지를 [-1, 1] 범위로 정규화
        )

    def forward(self, z):
        """
        잠재 벡터를 의료 이미지로 디코딩

        Args:
            z: 잠재 벡터 [B, latent_channels, H', W']

        Returns:
            재구성된 의료 이미지 [B, out_channels, H, W]
        """
        x = self.conv_in(z)
        x = self.mid_block(x)

        # Upsampling
        for up_block in self.up_blocks:
            x = up_block(x)

        # Output
        return self.conv_out(x)

class MedicalVAE(nn.Module):
    """의료 이미지를 위한 완전한 VAE"""
    def __init__(self, in_channels=1, latent_channels=4, base_channels=128):
        super().__init__()
        self.encoder = MedicalVAEEncoder(in_channels, latent_channels, base_channels)
        self.decoder = MedicalVAEDecoder(latent_channels, in_channels, base_channels)

    def encode(self, x):
        """Encode and sample from posterior"""
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode(self, z):
        """Decode latent to image"""
        return self.decoder(z)

    def reparameterize(self, mean, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        """Complete VAE forward pass"""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

class LatentUNet(nn.Module):
    """잠재 공간에서 작동하는 U-Net"""
    def __init__(self, in_channels=4, model_channels=64, out_channels=4,
                 num_res_blocks=2, attention_resolutions=[8],
                 channel_mult=[1, 2, 4], time_emb_dim=256):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels

        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down sampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [model_channels]

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_emb_dim)]
                ch = mult * model_channels

                if 8 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)

        # Middle
        self.middle_block = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_emb_dim),
        ])

        # Up sampling
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResidualBlock(ch + ich, mult * model_channels, time_emb_dim)]
                ch = mult * model_channels

                if 8 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                if level != 0 and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

                self.up_blocks.append(nn.ModuleList(layers))

        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        """잠재 공간에서 노이즈 예측"""
        time_emb = self.time_embed(timesteps)

        x = self.input_conv(x)
        skip_connections = [x]

        # Down
        for layers in self.down_blocks:
            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    x = layer(x, time_emb)
                else:
                    x = layer(x)
            skip_connections.append(x)

        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ResidualBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)

        # Up
        for layers in self.up_blocks:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)

            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    x = layer(x, time_emb)
                else:
                    x = layer(x)

        return self.output_conv(x)

class MedicalLatentDiffusion:
    """의료 이미지를 위한 Latent Diffusion Model"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # Noise schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_latents(self, z, t):
        """잠재 공간에서 노이즈 추가"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(z)
        return sqrt_alpha_hat * z + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_latents(self, model, n, latent_channels=4, latent_size=8):
        """잠재 공간에서 샘플링"""
        model.eval()
        with torch.no_grad():
            z = torch.randn((n, latent_channels, latent_size, latent_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), desc="Sampling latents"):
                t = (torch.ones(n) * i).long().to(self.device)

                predicted_noise = model(z, t)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(z)
                else:
                    noise = torch.zeros_like(z)

                z = 1 / torch.sqrt(alpha) * (z - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        return z

class MedicalImageDataset(Dataset):
    def __init__(self, images, image_size=64, channels=1):
        self.images = images
        self.image_size = image_size
        self.channels = channels

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and self.channels == 1:
            if image.shape[-1] == 3:
                image = np.mean(image, axis=-1, keepdims=True)

        if self.transform:
            image = self.transform(image.astype(np.uint8))

        return image, 0

def train_medical_latent_diffusion(dataset_type='chest_xray', data_path=None,
                                  vae_epochs=100, diffusion_epochs=200, save_interval=20):
    """
    Medical Latent Diffusion 2-stage 훈련

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        vae_epochs: VAE 훈련 에포크
        diffusion_epochs: Diffusion 훈련 에포크
        save_interval: 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("latent_diffusion", dataset_type)

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        vae_epochs = int(os.getenv('TEST_EPOCHS', 5))
        diffusion_epochs = int(os.getenv('TEST_EPOCHS', 5))
        logger.log(f"Quick test mode: VAE {vae_epochs}, Diffusion {diffusion_epochs} epochs")

    # Hyperparameters
    batch_size = 16
    img_size = 64
    latent_size = 8  # 8x compression
    latent_channels = 4
    lr = 1e-4

    config = {
        'algorithm': 'Latent_Diffusion',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'img_size': img_size,
        'latent_size': latent_size,
        'latent_channels': latent_channels,
        'vae_epochs': vae_epochs,
        'diffusion_epochs': diffusion_epochs,
        'lr': lr,
        'data_path': data_path
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Load medical data
    logger.log(f"Loading {dataset_type} data...")
    if dataset_type == 'chest_xray':
        images = load_chest_xray_data(data_path, num_samples=1000, image_size=img_size)
        input_channels = 1
    elif dataset_type == 'brain_mri':
        images = load_brain_mri_data(data_path, num_samples=1000, image_size=img_size)
        input_channels = 1
    else:
        loader = MedicalImageLoader(dataset_type, img_size)
        if data_path and os.path.exists(data_path):
            images = loader.load_real_dataset(data_path, 1000)
        else:
            images = loader.create_synthetic_medical_data(1000)
        input_channels = 3 if len(images[0].shape) == 3 else 1

    logger.log(f"Loaded {len(images)} {dataset_type} images")

    # Create dataset
    dataset = MedicalImageDataset(images, img_size, input_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ============================================
    # STAGE 1: VAE Training
    # ============================================
    logger.log("Starting Stage 1: VAE Training...")

    vae = MedicalVAE(input_channels, latent_channels, base_channels=64).to(device)
    vae_optimizer = optim.Adam(vae.parameters(), lr=lr)

    vae.train()
    for epoch in range(vae_epochs):
        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch+1}/{vae_epochs}")
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)

            vae_optimizer.zero_grad()

            # VAE forward
            recon, mean, logvar = vae(images)

            # Losses
            recon_loss = F.mse_loss(recon, images, reduction='sum') / images.shape[0]
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / images.shape[0]

            total_loss = recon_loss + kl_loss * 0.1  # Beta-VAE with beta=0.1

            total_loss.backward()
            vae_optimizer.step()

            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            pbar.set_postfix(recon=recon_loss.item(), kl=kl_loss.item())

        logger.log_metrics(epoch + 1, epoch_recon_loss / len(dataloader),
                          kl_loss=epoch_kl_loss / len(dataloader))

        # Save VAE results
        if (epoch + 1) % save_interval == 0:
            with torch.no_grad():
                # Reconstruction
                sample_images = images[:8]
                recon_images, _, _ = vae(sample_images)

                # Convert for saving
                orig_imgs = ((sample_images + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
                recon_imgs = ((recon_images + 1) / 2 * 255).cpu().numpy().astype(np.uint8)

                if input_channels == 1:
                    orig_imgs = orig_imgs.squeeze(1)
                    recon_imgs = recon_imgs.squeeze(1)
                else:
                    orig_imgs = orig_imgs.transpose(0, 2, 3, 1)
                    recon_imgs = recon_imgs.transpose(0, 2, 3, 1)

                # Save comparison
                comparison = []
                titles = []
                for i in range(8):
                    comparison.extend([orig_imgs[i], recon_imgs[i]])
                    titles.extend([f"Orig {i+1}", f"Recon {i+1}"])

                logger.save_image_grid(comparison, f"vae_reconstruction_epoch_{epoch+1:03d}",
                                      titles=titles, rows=8, cols=2,
                                      cmap='gray' if input_channels == 1 else None)

    # Save trained VAE
    logger.save_model(vae, "vae_final", optimizer=vae_optimizer,
                     epoch=vae_epochs, config=config)

    # ============================================
    # STAGE 2: Diffusion Training in Latent Space
    # ============================================
    logger.log("Starting Stage 2: Diffusion Training in Latent Space...")

    diffusion_model = LatentUNet(latent_channels, model_channels=32, out_channels=latent_channels).to(device)
    diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
    latent_diffusion = MedicalLatentDiffusion(device=device)

    # Freeze VAE
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    diffusion_model.train()
    for epoch in range(diffusion_epochs):
        pbar = tqdm(dataloader, desc=f"Diffusion Epoch {epoch+1}/{diffusion_epochs}")
        epoch_loss = 0

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)

            diffusion_optimizer.zero_grad()

            # Encode to latent space
            with torch.no_grad():
                mean, logvar = vae.encode(images)
                z = vae.reparameterize(mean, logvar)

            # Sample timesteps
            t = latent_diffusion.sample_timesteps(z.shape[0]).to(device)

            # Add noise in latent space
            z_t, noise = latent_diffusion.noise_latents(z, t)

            # Predict noise
            predicted_noise = diffusion_model(z_t, t)

            # Calculate loss
            loss = F.mse_loss(noise, predicted_noise)

            loss.backward()
            diffusion_optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        logger.log_metrics(vae_epochs + epoch + 1, epoch_loss / len(dataloader))

        # Generate samples
        if (epoch + 1) % save_interval == 0:
            logger.log(f"Generating latent diffusion samples at epoch {epoch + 1}...")

            with torch.no_grad():
                # Sample in latent space
                sampled_latents = latent_diffusion.sample_latents(
                    diffusion_model, n=16, latent_channels=latent_channels, latent_size=latent_size
                )

                # Decode to images
                generated_images = vae.decode(sampled_latents)

                # Convert for saving
                gen_imgs = ((generated_images + 1) / 2 * 255).cpu().numpy().astype(np.uint8)

                if input_channels == 1:
                    gen_imgs = gen_imgs.squeeze(1)
                else:
                    gen_imgs = gen_imgs.transpose(0, 2, 3, 1)

                logger.save_image_grid(
                    [gen_imgs[i] for i in range(16)],
                    f"generated_samples_epoch_{epoch+1:03d}",
                    titles=[f"Gen {i+1}" for i in range(16)],
                    rows=4, cols=4,
                    cmap='gray' if input_channels == 1 else None
                )

    # Final results
    logger.log("Training completed! Saving final results...")

    # Save final models
    logger.save_model(diffusion_model, "diffusion_model_final",
                     optimizer=diffusion_optimizer, epoch=diffusion_epochs, config=config)

    # Final generation
    logger.log("Generating final samples...")
    with torch.no_grad():
        final_latents = latent_diffusion.sample_latents(
            diffusion_model, n=64, latent_channels=latent_channels, latent_size=latent_size
        )
        final_images = vae.decode(final_latents)

        final_imgs = ((final_images + 1) / 2 * 255).cpu().numpy().astype(np.uint8)

        if input_channels == 1:
            final_imgs = final_imgs.squeeze(1)
        else:
            final_imgs = final_imgs.transpose(0, 2, 3, 1)

        logger.save_image_grid(
            [final_imgs[i] for i in range(64)],
            "final_generated_samples",
            titles=[f"Final {i+1}" for i in range(64)],
            rows=8, cols=8,
            cmap='gray' if input_channels == 1 else None
        )

    # Plot training curves
    logger.plot_training_curves()

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return vae, diffusion_model, latent_diffusion, results_dir

if __name__ == "__main__":
    print("Medical Latent Diffusion Model Implementation")
    print("=============================================")

    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'

    print(f"Training Latent Diffusion on {selected_dataset} images...")
    print("LDM Key Features:")
    print("- Two-stage training: VAE + Diffusion in latent space")
    print("- Memory efficient: 16-64x less VRAM usage")
    print("- High resolution capable with compact latent space")
    print("- Perceptually equivalent latent representation")
    print("- Fast training and inference compared to pixel-space diffusion")

    try:
        vae, diffusion_model, latent_diffusion, results_dir = train_medical_latent_diffusion(
            dataset_type=selected_dataset,
            data_path=None,
            vae_epochs=50,
            diffusion_epochs=100,
            save_interval=10
        )

        print(f"\nTraining completed successfully!")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: VAE reconstructions and generated samples")
        print("- models/: Separate VAE and diffusion model checkpoints")
        print("- logs/: Two-stage training logs")
        print("- plots/: Training curves for both stages")
        print("- metrics/: Training metrics for reconstruction and generation")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise