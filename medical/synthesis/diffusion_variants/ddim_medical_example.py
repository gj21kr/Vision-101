"""
Medical DDIM (Denoising Diffusion Implicit Models) Implementation

의료 이미지 생성을 위한 DDIM 구현으로, DDPM보다 빠르고 결정론적인 샘플링을 제공합니다.
의료 분야에서 특히 유용한 빠른 고품질 이미지 생성이 가능합니다.

의료 특화 기능:
- 의료 이미지 특화 데이터 로더 (chest X-ray, brain MRI, skin lesion)
- 자동 결과 저장 및 로깅 시스템
- 빠른 의료 이미지 생성 (DDPM 대비 10-50배 빠름)
- 결정론적 생성으로 재현 가능한 결과
- 의료 이미지 보간 및 편집 기능

DDIM의 핵심 개선사항:
1. 결정론적 샘플링: η=0일 때 완전히 결정론적
2. 빠른 샘플링: DDPM의 1000 스텝 → 20-250 스텝
3. 잠재 공간 보간: smooth interpolation 가능
4. 이미지 편집: 의미적 편집 응용

의료 분야에서의 활용:
- 실시간 의료 이미지 생성
- 의료 이미지 augmentation
- 의료진 교육용 빠른 시뮬레이션
- 의료 이미지 편집 및 보간

수학적 공식:
DDIM: x_{t-1} = √(ᾱ_{t-1}) · pred_x0 + √(1-ᾱ_{t-1}-σ²) · ε_θ(x_t,t) + σ_t ε
where pred_x0 = (x_t - √(1-ᾱ_t) · ε_θ(x_t,t)) / √(ᾱ_t)

Reference:
- Song, J., Meng, C., & Ermon, S. (2021).
  "Denoising diffusion implicit models."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import math
from tqdm import tqdm

from ...medical_data_utils import (
    MedicalImageLoader,
    load_chest_xray_data,
    load_brain_mri_data,
)
from ...result_logger import create_logger_for_generating

# Import UNet from DDPM medical example
from .ddpm_medical_example import MedicalUNet, MedicalImageDataset

class MedicalDDIM:
    """의료 이미지를 위한 DDIM"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Precompute noise schedule (same as DDPM)
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """의료 이미지에 적합한 선형 노이즈 스케줄"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """Add noise to medical images (same as DDPM)"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        """Sample random timesteps"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def ddim_sample(self, model, n, channels=1, eta=0.0, ddim_steps=50):
        """
        DDIM sampling - 의료 이미지를 위한 빠르고 결정론적 샘플링

        Args:
            model: 훈련된 노이즈 예측 모델
            n: 생성할 이미지 수
            channels: 이미지 채널 수
            eta: 확률성 조절 (0=완전 결정론적, 1=DDPM과 동일)
            ddim_steps: 샘플링 스텝 수 (적을수록 빠름)

        Returns:
            생성된 의료 이미지
        """
        print(f"DDIM sampling {n} medical images with {ddim_steps} steps (eta={eta})")
        model.eval()

        # Create subset of timesteps for faster sampling
        skip = self.noise_steps // ddim_steps
        seq = range(0, self.noise_steps, skip)
        seq = list(reversed(seq))

        with torch.no_grad():
            # Start from random noise
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)

            for i, j in enumerate(tqdm(seq[:-1], desc="DDIM Sampling")):
                t = torch.full((n,), j, dtype=torch.long).to(self.device)
                next_t = torch.full((n,), seq[i + 1], dtype=torch.long).to(self.device)

                # Predict noise
                predicted_noise = model(x, t)

                # Get alpha values
                alpha_t = self.alpha_hat[t][:, None, None, None]
                alpha_t_next = self.alpha_hat[next_t][:, None, None, None]

                # Predict x_0 (original medical image)
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

                # Clamp to valid range for medical images
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                # Compute variance (eta controls stochasticity)
                sigma_t = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)

                # Compute direction pointing to x_t
                direction = torch.sqrt(1 - alpha_t_next - sigma_t**2) * predicted_noise

                # Add noise if eta > 0 (for stochastic sampling)
                noise = torch.randn_like(x) if eta > 0 else 0

                # DDIM update rule
                x = torch.sqrt(alpha_t_next) * pred_x0 + direction + sigma_t * noise

        model.train()
        # Denormalize
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def interpolate_latents(self, model, start_noise, end_noise, steps=10, channels=1, eta=0.0, ddim_steps=50):
        """
        의료 이미지 잠재 공간 보간

        DDIM의 결정론적 특성을 활용하여 두 노이즈 벡터 사이의
        부드러운 보간을 통해 의료 이미지 간 전환을 생성

        Args:
            model: 훈련된 모델
            start_noise: 시작 노이즈 벡터
            end_noise: 끝 노이즈 벡터
            steps: 보간 스텝 수
            channels: 이미지 채널 수
            eta: 확률성 조절
            ddim_steps: DDIM 샘플링 스텝

        Returns:
            보간된 의료 이미지 시퀀스
        """
        print(f"Interpolating between medical images with {steps} steps")
        model.eval()

        interpolated_images = []

        with torch.no_grad():
            for i in range(steps):
                # Linear interpolation in noise space
                alpha = i / (steps - 1)
                interpolated_noise = (1 - alpha) * start_noise + alpha * end_noise

                # Generate image from interpolated noise
                image = self._sample_from_noise(model, interpolated_noise, channels, eta, ddim_steps)
                interpolated_images.append(image)

        return interpolated_images

    def _sample_from_noise(self, model, noise, channels=1, eta=0.0, ddim_steps=50):
        """특정 노이즈에서 이미지 생성 (내부 함수)"""
        skip = self.noise_steps // ddim_steps
        seq = range(0, self.noise_steps, skip)
        seq = list(reversed(seq))

        x = noise

        for i, j in enumerate(seq[:-1]):
            t = torch.full((noise.shape[0],), j, dtype=torch.long).to(self.device)
            next_t = torch.full((noise.shape[0],), seq[i + 1], dtype=torch.long).to(self.device)

            predicted_noise = model(x, t)

            alpha_t = self.alpha_hat[t][:, None, None, None]
            alpha_t_next = self.alpha_hat[next_t][:, None, None, None]

            pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

            sigma_t = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)
            direction = torch.sqrt(1 - alpha_t_next - sigma_t**2) * predicted_noise

            noise_term = torch.randn_like(x) if eta > 0 else 0
            x = torch.sqrt(alpha_t_next) * pred_x0 + direction + sigma_t * noise_term

        # Denormalize
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def calculate_quality_score(images):
    """의료 이미지 품질 평가"""
    if len(images) == 0:
        return 0.0
    images_np = images.detach().cpu().numpy()
    variance = np.var(images_np)
    return float(variance)

def compare_sampling_speeds(ddim, model, channels=1):
    """DDIM의 다양한 샘플링 속도 비교"""
    step_counts = [1000, 500, 100, 50, 20]
    results = []

    for steps in step_counts:
        print(f"\nTesting DDIM with {steps} steps...")
        import time
        start_time = time.time()

        # Generate single image for timing
        _ = ddim.ddim_sample(model, n=1, channels=channels, ddim_steps=steps)

        end_time = time.time()
        elapsed = end_time - start_time

        results.append({
            'steps': steps,
            'time': elapsed,
            'speedup': results[0]['time'] / elapsed if results else 1.0
        })

    return results

def train_medical_ddim(dataset_type='chest_xray', data_path=None, num_epochs=500, save_interval=50):
    """
    Medical DDIM 훈련 함수 (DDPM과 동일한 훈련, DDIM은 샘플링에서만 차이)

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("ddim", dataset_type)

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        num_epochs = int(os.getenv('TEST_EPOCHS', 5))
        save_interval = min(save_interval, num_epochs)
        logger.log(f"Quick test mode: training for {num_epochs} epochs only")

    # DDIM hyperparameters (same as DDPM for training)
    batch_size = 16
    img_size = 64
    noise_steps = 1000
    lr = 3e-4

    # Save configuration
    config = {
        'algorithm': 'DDIM',
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
    ddim = MedicalDDIM(noise_steps=noise_steps, img_size=img_size, device=device)

    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop (same as DDPM - DDIM only differs in sampling)
    logger.log("Starting DDIM training (same as DDPM, difference is in sampling)...")

    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0

        for i, (images_batch, _) in enumerate(pbar):
            images_batch = images_batch.to(device)

            # Sample random timesteps
            t = ddim.sample_timesteps(images_batch.shape[0]).to(device)

            # Add noise to images
            x_t, noise = ddim.noise_images(images_batch, t)

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

            # Generate samples with different DDIM configurations
            sampling_configs = [
                {'eta': 0.0, 'steps': 50, 'name': 'deterministic_50'},
                {'eta': 0.0, 'steps': 20, 'name': 'deterministic_20'},
                {'eta': 0.5, 'steps': 50, 'name': 'stochastic_50'},
            ]

            for config_sample in sampling_configs:
                sampled_images = ddim.ddim_sample(
                    model, n=16, channels=input_channels,
                    eta=config_sample['eta'], ddim_steps=config_sample['steps']
                )

                if input_channels == 1:
                    sampled_imgs = sampled_images.squeeze(1).cpu().numpy()
                else:
                    sampled_imgs = sampled_images.permute(0, 2, 3, 1).cpu().numpy()

                logger.save_image_grid(
                    [sampled_imgs[j] for j in range(16)],
                    f"generated_samples_{config_sample['name']}_epoch_{epoch+1:03d}",
                    titles=[f"Gen {j+1}" for j in range(16)],
                    rows=4, cols=4,
                    cmap='gray' if input_channels == 1 else None
                )

            # Medical image interpolation demonstration
            if epoch > 0:  # Skip first epoch for speed
                logger.log("Generating medical image interpolation...")

                # Create two different noise vectors
                noise1 = torch.randn(1, input_channels, img_size, img_size).to(device)
                noise2 = torch.randn(1, input_channels, img_size, img_size).to(device)

                # Generate interpolation
                interpolated = ddim.interpolate_latents(
                    model, noise1, noise2, steps=8, channels=input_channels,
                    eta=0.0, ddim_steps=50
                )

                if input_channels == 1:
                    interp_imgs = [img.squeeze(1).cpu().numpy() for img in interpolated]
                else:
                    interp_imgs = [img.permute(0, 2, 3, 1).cpu().numpy() for img in interpolated]

                # Flatten for grid display
                interp_imgs_flat = [img[0] for img in interp_imgs]

                logger.save_image_grid(
                    interp_imgs_flat,
                    f"medical_interpolation_epoch_{epoch+1:03d}",
                    titles=[f"Step {j+1}" for j in range(len(interp_imgs_flat))],
                    rows=1, cols=len(interp_imgs_flat),
                    cmap='gray' if input_channels == 1 else None
                )

            # Quality metrics
            try:
                quality_score = calculate_quality_score(model(x_t, t))
                logger.log(f"Epoch {epoch+1} - Quality Score: {quality_score:.4f}")
            except:
                pass

        # Save model checkpoint
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(model, f"ddim_model_epoch_{epoch+1:03d}",
                             optimizer=optimizer, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Saving final results...")

    # Plot training curves
    logger.plot_training_curves()

    # Save final model
    logger.save_model(model, "ddim_model_final", optimizer=optimizer,
                     epoch=num_epochs, config=config)

    # Final comprehensive generation and speed comparison
    logger.log("Generating final samples and conducting speed comparison...")

    # Speed comparison
    try:
        speed_results = compare_sampling_speeds(ddim, model, input_channels)
        logger.log("DDIM Sampling Speed Comparison:")
        for result in speed_results:
            logger.log(f"Steps: {result['steps']}, Time: {result['time']:.2f}s, Speedup: {result['speedup']:.1f}x")
    except Exception as e:
        logger.log(f"Could not conduct speed comparison: {e}")

    # Generate final samples with different configurations
    final_configs = [
        {'eta': 0.0, 'steps': 50, 'name': 'final_deterministic'},
        {'eta': 0.0, 'steps': 20, 'name': 'final_fast'},
    ]

    for config_final in final_configs:
        final_samples = ddim.ddim_sample(
            model, n=64, channels=input_channels,
            eta=config_final['eta'], ddim_steps=config_final['steps']
        )

        if input_channels == 1:
            final_imgs = final_samples.squeeze(1).cpu().numpy()
        else:
            final_imgs = final_samples.permute(0, 2, 3, 1).cpu().numpy()

        logger.save_image_grid(
            [final_imgs[j] for j in range(64)],
            f"{config_final['name']}_samples",
            titles=[f"Final {j+1}" for j in range(64)],
            rows=8, cols=8,
            cmap='gray' if input_channels == 1 else None
        )

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return model, ddim, results_dir

if __name__ == "__main__":
    print("Medical DDIM Implementation")
    print("============================")

    # Configuration
    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'

    print(f"Training DDIM on {selected_dataset} images...")
    print("DDIM Key Features:")
    print("- Deterministic sampling (eta=0) for reproducible results")
    print("- Fast sampling: 10-50x faster than DDPM")
    print("- Same training as DDPM, only sampling differs")
    print("- Medical image interpolation capabilities")
    print("- Quality preservation with fewer sampling steps")

    # Train the model
    try:
        model, ddim, results_dir = train_medical_ddim(
            dataset_type=selected_dataset,
            data_path=None,
            num_epochs=100,
            save_interval=20
        )

        print(f"\nTraining completed successfully!")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Generated samples with different DDIM configurations")
        print("- images/: Medical image interpolation sequences")
        print("- models/: Model checkpoints (compatible with DDPM)")
        print("- logs/: Training logs with DDIM sampling metrics")
        print("- plots/: Training loss curves")
        print("- metrics/: Training metrics and speed comparisons")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise