"""
Medical DDPM (Denoising Diffusion Probabilistic Models) Implementation

의료 이미지 생성을 위한 DDPM 구현으로, forward diffusion 과정에서 점진적으로 노이즈를 추가하고,
reverse diffusion 과정에서 신경망을 통해 노이즈를 제거하여 고품질 의료 이미지를 생성합니다.

의료 특화 기능:
- 의료 이미지 특화 데이터 로더 (chest X-ray, brain MRI, skin lesion)
- 자동 결과 저장 및 로깅 시스템
- 의료 이미지에 최적화된 노이즈 스케줄
- 의료 이미지 품질 평가 메트릭
- 점진적 노이즈 제거 과정 시각화

DDPM의 핵심 개념:
1. Forward Process: q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
2. Reverse Process: p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t² I)
3. Training Objective: E_t,x_0,ε [||ε - ε_θ(x_t, t)||²]

의료 분야에서의 활용:
- 의료 이미지 데이터 증강
- 희귀 질병 이미지 생성
- 다양한 촬영 조건 시뮬레이션
- 의료진 교육용 합성 이미지

Reference:
- Ho, J., Jain, A., & Abbeel, P. (2020).
  "Denoising diffusion probabilistic models."
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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT.name not in {"medical", "non_medical"} and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if PROJECT_ROOT.name in {"medical", "non_medical"}:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import math
from tqdm import tqdm

# Ensure project root is on sys.path
from medical.medical_data_utils import MedicalImageLoader, load_chest_xray_data, load_brain_mri_data
from non_medical.result_logger import create_logger_for_generating

class TimeEmbedding(nn.Module):
    """시간 정보를 위한 Sinusoidal Position Embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2

        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """시간 임베딩을 포함한 Residual block"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.activation = nn.SiLU()

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Add time embedding
        time_emb = self.activation(self.time_mlp(time_emb))
        x = x + time_emb[:, :, None, None]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        return x + residual

class AttentionBlock(nn.Module):
    """Self-attention block for better global coherence"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        B, C, H, W = x.shape

        q = self.q(x).view(B, C, H * W).transpose(1, 2)
        k = self.k(x).view(B, C, H * W)
        v = self.v(x).view(B, C, H * W).transpose(1, 2)

        # Scaled dot-product attention
        attn = torch.bmm(q, k) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, v).transpose(1, 2).view(B, C, H, W)
        out = self.proj_out(out)

        return out + residual

class MedicalUNet(nn.Module):
    """의료 이미지를 위한 U-Net architecture"""
    def __init__(self, in_channels=1, model_channels=128, out_channels=1,
                 num_res_blocks=2, attention_resolutions=[16],
                 channel_mult=[1, 2, 2, 2], time_emb_dim=512):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult

        # Time embedding - 의료 이미지의 diffusion timestep 정보
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down sampling - 의료 이미지의 계층적 특징 추출
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [model_channels]

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_emb_dim)]
                ch = mult * model_channels

                # Add attention at specified resolutions for better medical structure preservation
                if 64 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)

        # Middle - 최저 해상도에서 글로벌 의료 특징 처리
        self.middle_block = nn.ModuleList([
            ResidualBlock(ch, ch, time_emb_dim),
            AttentionBlock(ch),
            ResidualBlock(ch, ch, time_emb_dim),
        ])

        # Up sampling - 고해상도 의료 이미지 디테일 복원
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ResidualBlock(ch + ich, mult * model_channels, time_emb_dim)]
                ch = mult * model_channels

                # Add attention for fine medical detail preservation
                if 64 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                # Upsample (except last iteration of last level)
                if level != 0 and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

                self.up_blocks.append(nn.ModuleList(layers))

        # Output - 최종 의료 이미지 생성
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        """
        의료 이미지 노이즈 예측

        Args:
            x: 노이즈가 추가된 의료 이미지 [B, C, H, W]
            timesteps: Diffusion timesteps [B]

        Returns:
            예측된 노이즈 [B, C, H, W]
        """
        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Initial conv
        x = self.input_conv(x)

        # Store skip connections for U-Net
        skip_connections = [x]

        # Down sampling
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

        # Up sampling
        for layers in self.up_blocks:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)

            for layer in layers:
                if isinstance(layer, ResidualBlock):
                    x = layer(x, time_emb)
                else:
                    x = layer(x)

        # Output
        return self.output_conv(x)

class MedicalDDPM:
    """의료 이미지를 위한 DDPM"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Precompute noise schedule - 의료 이미지에 최적화
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # For efficient sampling
        self.alpha_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.beta * (1. - self.alpha_hat_prev) / (1. - self.alpha_hat)

    def prepare_noise_schedule(self):
        """의료 이미지에 적합한 선형 노이즈 스케줄"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """Forward diffusion: 의료 이미지에 노이즈 추가"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        Ɛ = torch.randn_like(x)
        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ

        return x_noisy, Ɛ

    def sample_timesteps(self, n):
        """랜덤 timesteps 샘플링"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, channels=1):
        """의료 이미지 생성 샘플링"""
        print(f"Generating {n} medical images...")
        model.eval()
        with torch.no_grad():
            # 랜덤 노이즈에서 시작
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)

            # Reverse diffusion process
            for i in tqdm(reversed(range(1, self.noise_steps)), desc="Sampling"):
                t = (torch.ones(n) * i).long().to(self.device)

                # 노이즈 예측
                predicted_noise = model(x, t)

                # Denoising step 계산
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                # DDPM sampling equation
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        model.train()
        # Denormalize
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

class MedicalImageDataset(Dataset):
    def __init__(self, images, image_size=64, channels=1):
        self.images = images
        self.image_size = image_size
        self.channels = channels

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if len(image.shape) == 2:  # Grayscale
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and self.channels == 1:
            if image.shape[-1] == 3:
                image = np.mean(image, axis=-1, keepdims=True)

        if self.transform:
            image = self.transform(image.astype(np.uint8))

        return image, 0

def calculate_quality_score(images):
    """의료 이미지 품질 평가"""
    if len(images) == 0:
        return 0.0
    images_np = images.detach().cpu().numpy()
    variance = np.var(images_np)
    return float(variance)

def visualize_diffusion_process(ddpm, model, original_image, timesteps=[0, 250, 500, 750, 999]):
    """의료 이미지의 diffusion 과정 시각화"""
    model.eval()
    with torch.no_grad():
        images = []
        titles = []

        for t in timesteps:
            if t == 0:
                img = original_image
                title = "Original"
            else:
                t_tensor = torch.tensor([t]).to(ddpm.device)
                noisy_img, _ = ddpm.noise_images(original_image.unsqueeze(0).to(ddpm.device), t_tensor)
                img = noisy_img.squeeze(0)
                title = f"t={t}"

            # Denormalize for visualization
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)

            if img.dim() == 3 and img.shape[0] == 1:
                img = img.squeeze(0)

            images.append((img.cpu().numpy() * 255).astype(np.uint8))
            titles.append(title)

        return images, titles

def train_medical_ddpm(dataset_type='chest_xray', data_path=None, num_epochs=500, save_interval=50):
    """
    Medical DDPM 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수 (DDPM은 오랜 훈련 필요)
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("ddpm", dataset_type)

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        num_epochs = int(os.getenv('TEST_EPOCHS', 5))
        save_interval = min(save_interval, num_epochs)
        logger.log(f"Quick test mode: training for {num_epochs} epochs only")

    # DDPM hyperparameters
    batch_size = 16  # DDPM은 메모리를 많이 사용
    img_size = 64
    noise_steps = 1000
    lr = 3e-4

    # Save configuration
    config = {
        'algorithm': 'DDPM',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'img_size': img_size,
        'noise_steps': noise_steps,
        'lr': lr,
        'num_epochs': num_epochs,
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
    logger.log(f"Image shape: {images[0].shape}")

    # Save sample original images
    sample_images = [images[i] for i in range(min(9, len(images)))]
    logger.save_image_grid(sample_images, "original_samples",
                          titles=[f"Original {i+1}" for i in range(len(sample_images))],
                          cmap='gray' if input_channels == 1 else None)

    # Create dataset and dataloader
    dataset = MedicalImageDataset(images, img_size, input_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model and diffusion
    model = MedicalUNet(in_channels=input_channels, out_channels=input_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = MedicalDDPM(noise_steps=noise_steps, img_size=img_size, device=device)

    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    logger.log("Starting DDPM training...")
    logger.log("Note: DDPM requires long training for high quality results")

    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0

        for i, (images_batch, _) in enumerate(pbar):
            images_batch = images_batch.to(device)

            # Sample random timesteps
            t = diffusion.sample_timesteps(images_batch.shape[0]).to(device)

            # Add noise to images
            x_t, noise = diffusion.noise_images(images_batch, t)

            # Predict noise
            predicted_noise = model(x_t, t)

            # Calculate loss
            loss = mse(noise, predicted_noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        logger.log_metrics(epoch + 1, avg_loss)

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Saving intermediate results at epoch {epoch + 1}...")

            # Generate samples
            sampled_images = diffusion.sample(model, n=16, channels=input_channels)

            # Convert for saving
            if input_channels == 1:
                sampled_imgs = sampled_images.squeeze(1).cpu().numpy()
            else:
                sampled_imgs = sampled_images.permute(0, 2, 3, 1).cpu().numpy()

            logger.save_image_grid(
                [sampled_imgs[j] for j in range(16)],
                f"generated_samples_epoch_{epoch+1:03d}",
                titles=[f"Generated {j+1}" for j in range(16)],
                rows=4, cols=4,
                cmap='gray' if input_channels == 1 else None
            )

            # Visualize diffusion process on a sample image
            try:
                sample_original = dataset[0][0]  # Get first image
                diff_images, diff_titles = visualize_diffusion_process(
                    diffusion, model, sample_original
                )
                logger.save_image_grid(
                    diff_images,
                    f"diffusion_process_epoch_{epoch+1:03d}",
                    titles=diff_titles,
                    rows=1, cols=len(diff_images),
                    cmap='gray' if input_channels == 1 else None
                )
            except Exception as e:
                logger.log(f"Could not visualize diffusion process: {e}")

            # Quality metrics
            quality_score = calculate_quality_score(model(x_t, t))
            logger.log(f"Epoch {epoch+1} - Quality Score: {quality_score:.4f}")

        # Save model checkpoint
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(model, f"ddpm_model_epoch_{epoch+1:03d}",
                             optimizer=optimizer, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Saving final results...")

    # Plot training curves
    logger.plot_training_curves()

    # Save final model
    logger.save_model(model, "ddpm_model_final", optimizer=optimizer,
                     epoch=num_epochs, config=config)

    # Final comprehensive generation
    logger.log("Generating final samples...")
    final_samples = diffusion.sample(model, n=64, channels=input_channels)

    if input_channels == 1:
        final_imgs = final_samples.squeeze(1).cpu().numpy()
    else:
        final_imgs = final_samples.permute(0, 2, 3, 1).cpu().numpy()

    logger.save_image_grid(
        [final_imgs[j] for j in range(64)],
        "final_generated_samples",
        titles=[f"Final {j+1}" for j in range(64)],
        rows=8, cols=8,
        cmap='gray' if input_channels == 1 else None
    )

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return model, diffusion, results_dir

if __name__ == "__main__":
    print("Medical DDPM Implementation")
    print("===========================")

    # Configuration
    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'

    print(f"Training DDPM on {selected_dataset} images...")
    print("DDPM Key Features:")
    print("- Forward diffusion process adds Gaussian noise progressively")
    print("- Reverse diffusion process learned by U-Net with time embedding")
    print("- Sinusoidal time embeddings for timestep information")
    print("- Attention blocks for better global medical structure preservation")
    print("- High quality medical image generation (requires long training)")

    # Train the model
    try:
        model, diffusion, results_dir = train_medical_ddpm(
            dataset_type=selected_dataset,
            data_path=None,
            num_epochs=100,  # DDPM usually needs many more epochs
            save_interval=20
        )

        print(f"\nTraining completed successfully!")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Generated medical image samples and diffusion process visualization")
        print("- models/: Model checkpoints with U-Net architecture")
        print("- logs/: Training logs with diffusion-specific metrics")
        print("- plots/: Training loss curves")
        print("- metrics/: Training metrics in JSON format")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
