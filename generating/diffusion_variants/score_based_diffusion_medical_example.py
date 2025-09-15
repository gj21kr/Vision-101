"""
Medical Score-Based Diffusion Model Implementation

의료 이미지 생성을 위한 Score-based Diffusion Model 구현으로,
연속 시간 확률적 미분 방정식(SDE)을 사용한 고품질 의료 이미지 생성을 제공합니다.

의료 특화 기능:
- 의료 이미지 특화 데이터 로더 (chest X-ray, brain MRI, skin lesion)
- 자동 결과 저장 및 로깅 시스템
- 연속 시간 SDE 기반 생성
- 다양한 SDE 샘플링 방법 (Euler-Maruyama, PC sampler)
- 의료 이미지 품질 평가 메트릭

Score-based Diffusion의 핵심 개념:
1. Forward SDE: dx = f(x,t)dt + g(t)dw
2. Reverse SDE: dx = [f(x,t) - g(t)²∇_x log p_t(x)]dt + g(t)dw̄
3. Score Function: s_θ(x,t) ≈ ∇_x log p_t(x)
4. 연속 시간에서의 부드러운 생성 과정

의료 분야에서의 활용:
- 고품질 의료 이미지 생성
- 연속적인 노이즈 레벨에서의 생성
- 다양한 확률적 샘플링 방법
- Inpainting 및 conditional generation

Reference:
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).
  "Score-based generative modeling through stochastic differential equations."
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

# Import components from DDPM
from ddpm_medical_example import TimeEmbedding, ResidualBlock, AttentionBlock

class ScoreNet(nn.Module):
    """Score function s_θ(x,t) 예측을 위한 신경망"""
    def __init__(self, in_channels=1, model_channels=128, out_channels=1,
                 num_res_blocks=2, attention_resolutions=[16],
                 channel_mult=[1, 2, 2, 2], time_emb_dim=512):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks

        # Continuous time embedding for SDE
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down sampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [model_channels]

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_emb_dim)]
                ch = mult * model_channels

                if 64 // (2 ** level) in attention_resolutions:
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

                if 64 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                if level != 0 and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

                self.up_blocks.append(nn.ModuleList(layers))

        # Output - score function
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, t):
        """
        Score function 예측

        Args:
            x: 노이즈가 추가된 의료 이미지 [B, C, H, W]
            t: 연속 시간 [B] (0 to T)

        Returns:
            예측된 score function ∇_x log p_t(x)
        """
        # Time embedding
        time_emb = self.time_embed(t)

        # Input
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

class MedicalScoreBasedSDE:
    """의료 이미지를 위한 Score-based SDE 모델"""
    def __init__(self, T=1.0, sigma_min=0.01, sigma_max=50.0, device="cuda"):
        self.T = T  # 총 diffusion 시간
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.device = device

    def marginal_prob(self, x, t):
        """
        Marginal probability p_t(x|x_0)의 평균과 표준편차

        VP SDE: dx = -1/2 β(t) x dt + √β(t) dw
        여기서 β(t) = β_min + t(β_max - β_min)
        """
        # β(t) 함수 정의
        beta_min, beta_max = 0.1, 20.0
        beta_t = beta_min + t * (beta_max - beta_min)

        # 적분 ∫₀ᵗ β(s) ds
        log_mean_coeff = -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x

        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std[:, None, None, None]

    def prior_sampling(self, shape):
        """Prior distribution에서 샘플링 (Gaussian)"""
        return torch.randn(*shape, device=self.device)

    def sde(self, x, t):
        """
        Forward SDE: dx = f(x,t)dt + g(t)dw

        VP SDE의 경우:
        f(x,t) = -1/2 β(t) x
        g(t) = √β(t)
        """
        beta_min, beta_max = 0.1, 20.0
        beta_t = beta_min + t * (beta_max - beta_min)

        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def reverse_sde(self, x, t, score):
        """
        Reverse SDE: dx = [f(x,t) - g(t)² ∇_x log p_t(x)]dt + g(t)dw̄

        Args:
            x: 현재 상태
            t: 현재 시간
            score: Score function s_θ(x,t) ≈ ∇_x log p_t(x)
        """
        drift, diffusion = self.sde(x, t)

        # Reverse SDE drift
        drift = drift - (diffusion[:, None, None, None] ** 2) * score

        return drift, diffusion

    def euler_maruyama_sampler(self, score_model, shape, num_steps=1000):
        """
        Euler-Maruyama method로 reverse SDE 해결

        의료 이미지 생성을 위한 확률적 샘플링
        """
        score_model.eval()
        dt = -self.T / num_steps

        # Prior에서 시작
        x = self.prior_sampling(shape)

        with torch.no_grad():
            for i in tqdm(range(num_steps), desc="SDE Sampling"):
                t = torch.ones(shape[0], device=self.device) * (self.T - i * (-dt))

                # Score function 계산
                score = score_model(x, t)

                # Reverse SDE
                drift, diffusion = self.reverse_sde(x, t, score)

                # Euler-Maruyama 업데이트
                x_mean = x + drift * dt
                noise = torch.randn_like(x) if i < num_steps - 1 else 0
                x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * noise

        score_model.train()
        return x

    def pc_sampler(self, score_model, shape, num_steps=1000, snr=0.16):
        """
        Predictor-Corrector sampler

        더 고품질의 의료 이미지 생성을 위한 개선된 샘플링
        """
        score_model.eval()
        dt = -self.T / num_steps

        # Prior에서 시작
        x = self.prior_sampling(shape)

        with torch.no_grad():
            for i in tqdm(range(num_steps), desc="PC Sampling"):
                t = torch.ones(shape[0], device=self.device) * (self.T - i * (-dt))

                # Predictor step (Euler-Maruyama)
                score = score_model(x, t)
                drift, diffusion = self.reverse_sde(x, t, score)

                x_mean = x + drift * dt
                noise = torch.randn_like(x) if i < num_steps - 1 else 0
                x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * noise

                # Corrector step (Langevin MCMC)
                if i < num_steps - 1:
                    # 현재 시간에서의 표준편차
                    _, std = self.marginal_prob(torch.zeros_like(x), t)

                    for _ in range(1):  # 한 번의 corrector step
                        score = score_model(x, t)
                        noise = torch.randn_like(x)

                        # Langevin dynamics
                        step_size = (snr * std) ** 2 * 2
                        x = x + step_size[:, None, None, None] * score + torch.sqrt(2 * step_size[:, None, None, None]) * noise

        score_model.train()
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
            transforms.Lambda(lambda t: t * 2 - 1)  # [0,1] -> [-1,1]
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

def calculate_quality_score(images):
    """의료 이미지 품질 평가"""
    if len(images) == 0:
        return 0.0
    images_np = images.detach().cpu().numpy()
    variance = np.var(images_np)
    return float(variance)

def train_medical_score_based_sde(dataset_type='chest_xray', data_path=None,
                                 num_epochs=300, save_interval=30):
    """
    Medical Score-based SDE 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("score_based_sde", dataset_type)

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        num_epochs = int(os.getenv('TEST_EPOCHS', 5))
        save_interval = min(save_interval, num_epochs)
        logger.log(f"Quick test mode: training for {num_epochs} epochs only")

    # Hyperparameters
    batch_size = 16
    img_size = 64
    lr = 1e-4
    T = 1.0

    # Save configuration
    config = {
        'algorithm': 'Score_Based_SDE',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'img_size': img_size,
        'lr': lr,
        'T': T,
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

    # Save sample original images
    sample_images = [images[i] for i in range(min(9, len(images)))]
    logger.save_image_grid(sample_images, "original_samples",
                          titles=[f"Original {i+1}" for i in range(len(sample_images))],
                          cmap='gray' if input_channels == 1 else None)

    # Create dataset and dataloader
    dataset = MedicalImageDataset(images, img_size, input_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model and SDE
    score_model = ScoreNet(in_channels=input_channels, out_channels=input_channels).to(device)
    optimizer = optim.Adam(score_model.parameters(), lr=lr)
    sde = MedicalScoreBasedSDE(T=T, device=device)

    logger.log(f"ScoreNet parameters: {sum(p.numel() for p in score_model.parameters()):,}")

    # Training loop
    logger.log("Starting Score-based SDE training...")
    logger.log("Training objective: Denoising Score Matching")

    score_model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0

        for batch_idx, (images_batch, _) in enumerate(pbar):
            images_batch = images_batch.to(device)

            # Random time sampling [0, T]
            t = torch.rand(images_batch.shape[0], device=device) * T

            # Add noise according to SDE
            mean, std = sde.marginal_prob(images_batch, t)
            z = torch.randn_like(images_batch)
            perturbed_x = mean + std * z

            # Train score model
            optimizer.zero_grad()

            # Predict score
            score = score_model(perturbed_x, t)

            # Target: -z / std (정확한 score)
            target = -z / std

            # Denoising score matching loss
            loss = F.mse_loss(score, target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        logger.log_metrics(epoch + 1, avg_loss)

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Generating samples at epoch {epoch + 1}...")

            # Generate samples with different sampling methods
            sampling_methods = [
                {'method': 'euler_maruyama', 'steps': 1000, 'name': 'EM'},
                {'method': 'pc_sampler', 'steps': 500, 'name': 'PC'},
            ]

            for method_config in sampling_methods:
                if method_config['method'] == 'euler_maruyama':
                    generated = sde.euler_maruyama_sampler(
                        score_model, (16, input_channels, img_size, img_size),
                        num_steps=method_config['steps']
                    )
                else:  # pc_sampler
                    generated = sde.pc_sampler(
                        score_model, (16, input_channels, img_size, img_size),
                        num_steps=method_config['steps']
                    )

                # Convert for saving
                gen_imgs = ((generated + 1) / 2 * 255).cpu().numpy().astype(np.uint8)

                if input_channels == 1:
                    gen_imgs = gen_imgs.squeeze(1)
                else:
                    gen_imgs = gen_imgs.transpose(0, 2, 3, 1)

                logger.save_image_grid(
                    [gen_imgs[i] for i in range(16)],
                    f"generated_samples_{method_config['name']}_epoch_{epoch+1:03d}",
                    titles=[f"{method_config['name']} {i+1}" for i in range(16)],
                    rows=4, cols=4,
                    cmap='gray' if input_channels == 1 else None
                )

            # Quality metrics
            quality_score = calculate_quality_score(generated)
            logger.log(f"Epoch {epoch+1} - Quality Score: {quality_score:.4f}")

        # Save model checkpoint
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(score_model, f"score_model_epoch_{epoch+1:03d}",
                             optimizer=optimizer, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Saving final results...")

    # Plot training curves
    logger.plot_training_curves()

    # Save final model
    logger.save_model(score_model, "score_model_final", optimizer=optimizer,
                     epoch=num_epochs, config=config)

    # Final comprehensive generation
    logger.log("Generating final samples with both sampling methods...")

    # Euler-Maruyama sampling
    final_em = sde.euler_maruyama_sampler(
        score_model, (32, input_channels, img_size, img_size), num_steps=1000
    )

    em_imgs = ((final_em + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
    if input_channels == 1:
        em_imgs = em_imgs.squeeze(1)
    else:
        em_imgs = em_imgs.transpose(0, 2, 3, 1)

    logger.save_image_grid(
        [em_imgs[i] for i in range(32)],
        "final_generated_euler_maruyama",
        titles=[f"EM {i+1}" for i in range(32)],
        rows=4, cols=8,
        cmap='gray' if input_channels == 1 else None
    )

    # PC sampling
    final_pc = sde.pc_sampler(
        score_model, (32, input_channels, img_size, img_size), num_steps=500
    )

    pc_imgs = ((final_pc + 1) / 2 * 255).cpu().numpy().astype(np.uint8)
    if input_channels == 1:
        pc_imgs = pc_imgs.squeeze(1)
    else:
        pc_imgs = pc_imgs.transpose(0, 2, 3, 1)

    logger.save_image_grid(
        [pc_imgs[i] for i in range(32)],
        "final_generated_predictor_corrector",
        titles=[f"PC {i+1}" for i in range(32)],
        rows=4, cols=8,
        cmap='gray' if input_channels == 1 else None
    )

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return score_model, sde, results_dir

if __name__ == "__main__":
    print("Medical Score-Based SDE Implementation")
    print("======================================")

    # Configuration
    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'

    print(f"Training Score-based SDE on {selected_dataset} images...")
    print("Score-based SDE Key Features:")
    print("- Continuous-time stochastic differential equations")
    print("- Score function learning: ∇_x log p_t(x)")
    print("- Multiple sampling methods: Euler-Maruyama, Predictor-Corrector")
    print("- High-quality medical image generation")
    print("- Flexible reverse-time SDE solving")

    # Train the model
    try:
        score_model, sde, results_dir = train_medical_score_based_sde(
            dataset_type=selected_dataset,
            data_path=None,
            num_epochs=100,
            save_interval=20
        )

        print(f"\nTraining completed successfully!")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Generated samples using different SDE solvers")
        print("- models/: Score function model checkpoints")
        print("- logs/: Training logs with score matching loss")
        print("- plots/: Training loss curves")
        print("- metrics/: Training metrics and quality scores")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise