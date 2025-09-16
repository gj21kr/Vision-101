"""
DDIM (Denoising Diffusion Implicit Models) Implementation

DDIM은 Song et al. (2021)에서 제안된 DDPM의 개선 버전으로, 빠르고 결정론적인
샘플링을 가능하게 하는 혁신적인 방법입니다.

핵심 개선사항:

1. **결정론적 샘플링**:
   - η=0일 때 완전히 결정론적
   - 같은 초기 노이즈 → 같은 출력 이미지
   - 재현 가능한 결과

2. **빠른 샘플링**:
   - DDPM의 1000 스텝 → 20-250 스텝으로 단축
   - 타임스텝을 건너뛰는 non-Markovian 과정
   - 품질 저하 없이 10-50배 빠름

3. **잠재 공간 보간**:
   - 결정론적 특성으로 인한 smooth interpolation
   - 의미있는 잠재 공간 산술 연산 가능
   - 이미지 편집 응용에 유용

수학적 공식:
DDPM: x_{t-1} = μ_θ(x_t, t) + σ_t ε
DDIM: x_{t-1} = √(ᾱ_{t-1}) · pred_x0 + √(1-ᾱ_{t-1}) · ε_θ(x_t,t)

여기서 pred_x0 = (x_t - √(1-ᾱ_t) · ε_θ(x_t,t)) / √(ᾱ_t)

핵심 아이디어:
- Forward process를 non-Markovian으로 일반화
- 동일한 marginal distribution 유지
- Reverse process에서 deterministic mapping 사용

η (eta) 파라미터:
- η = 0: 완전 결정론적 (ODE)
- η = 1: DDPM과 동일 (SDE)
- 0 < η < 1: 부분적으로 확률적

응용:
- 실시간 이미지 생성
- 이미지 편집 (semantic editing)
- 잠재 공간 탐색
- 이미지 보간

장점:
- 매우 빠른 샘플링 (10-50배)
- 결정론적 결과
- DDPM과 동일한 훈련 과정
- 고품질 유지

Reference:
- Song, J., Meng, C., & Ermon, S. (2021).
  "Denoising diffusion implicit models."
  International Conference on Learning Representations (ICLR).
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
from tqdm import tqdm

# Import UNet from DDPM (assuming it's in the same directory)
from ddpm_example import UNet, TimeEmbedding, ResidualBlock, AttentionBlock

class DDIM:
    """Denoising Diffusion Implicit Models"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
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
        """Linear noise schedule"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """Add noise to images according to diffusion process (same as DDPM)"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        """Sample random timesteps"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def ddim_sample(self, model, n, channels=1, eta=0.0, ddim_steps=50):
        """
        DDIM sampling - deterministic and faster than DDPM
        eta=0: deterministic, eta=1: equivalent to DDPM
        """
        print(f"DDIM sampling {n} images with {ddim_steps} steps (eta={eta})")
        model.eval()

        # Create subset of timesteps for faster sampling
        skip = self.noise_steps // ddim_steps
        seq = range(0, self.noise_steps, skip)
        seq = list(reversed(seq))

        with torch.no_grad():
            # Start from random noise
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)

            for i, j in enumerate(tqdm(seq[:-1])):
                t = torch.full((n,), j, dtype=torch.long).to(self.device)
                next_t = torch.full((n,), seq[i + 1], dtype=torch.long).to(self.device)

                # Predict noise
                predicted_noise = model(x, t)

                # Get alpha values
                alpha_t = self.alpha_hat[t][:, None, None, None]
                alpha_t_next = self.alpha_hat[next_t][:, None, None, None]

                # Predict x_0
                pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

                # Compute variance
                sigma_t = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)

                # Compute direction pointing to x_t
                direction = torch.sqrt(1 - alpha_t_next - sigma_t**2) * predicted_noise

                # Add noise if eta > 0
                noise = torch.randn_like(x) if eta > 0 else 0

                # DDIM update
                x = torch.sqrt(alpha_t_next) * pred_x0 + direction + sigma_t * noise

        model.train()
        # Denormalize
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def ddim_sample_interpolation(self, model, x1, x2, n_interpolations=10, ddim_steps=50, eta=0.0):
        """
        Interpolate between two images using DDIM
        """
        model.eval()

        # Create subset of timesteps
        skip = self.noise_steps // ddim_steps
        seq = range(0, self.noise_steps, skip)
        seq = list(reversed(seq))

        interpolated_images = []

        with torch.no_grad():
            for alpha in torch.linspace(0, 1, n_interpolations):
                # Interpolate in latent space
                x_interpolated = (1 - alpha) * x1 + alpha * x2

                # DDIM reverse process
                x = x_interpolated.clone()

                for i, j in enumerate(seq[:-1]):
                    t = torch.full((1,), j, dtype=torch.long).to(self.device)
                    next_t = torch.full((1,), seq[i + 1], dtype=torch.long).to(self.device)

                    # Predict noise
                    predicted_noise = model(x, t)

                    # Get alpha values
                    alpha_t = self.alpha_hat[t][:, None, None, None]
                    alpha_t_next = self.alpha_hat[next_t][:, None, None, None]

                    # Predict x_0
                    pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

                    # Compute variance
                    sigma_t = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_next)

                    # Compute direction
                    direction = torch.sqrt(1 - alpha_t_next - sigma_t**2) * predicted_noise

                    # Add noise if eta > 0
                    noise = torch.randn_like(x) if eta > 0 else 0

                    # DDIM update
                    x = torch.sqrt(alpha_t_next) * pred_x0 + direction + sigma_t * noise

                # Denormalize and store
                x_denorm = (x.clamp(-1, 1) + 1) / 2
                interpolated_images.append(x_denorm)

        return torch.cat(interpolated_images, dim=0)

    def encode_to_latent(self, model, x0, t):
        """
        Encode image to latent representation at timestep t
        Useful for image editing applications
        """
        model.eval()
        with torch.no_grad():
            # Add noise to get x_t
            sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

            # We need to invert this: x_t = sqrt(alpha_hat) * x_0 + sqrt(1-alpha_hat) * noise
            # So: noise = (x_t - sqrt(alpha_hat) * x_0) / sqrt(1-alpha_hat)
            noise = torch.randn_like(x0)
            x_t = sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise

            return x_t, noise

def train_ddim():
    """Train model for DDIM (same training as DDPM)"""
    # Hyperparameters
    batch_size = 16
    epochs = 200
    lr = 3e-4
    img_size = 32
    noise_steps = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    transforms_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize to [-1, 1]
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms_pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and diffusion
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = DDIM(noise_steps=noise_steps, img_size=img_size, device=device)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training loop (identical to DDPM)
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)

            # Sample random timesteps
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            # Add noise to images
            x_t, noise = diffusion.noise_images(images, t)

            # Predict noise
            predicted_noise = model(x_t, t)

            # Calculate loss
            loss = mse(noise, predicted_noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())

        print(f"Epoch {epoch} | Average Loss: {epoch_loss/len(dataloader):.4f}")

        # Sample images every 25 epochs
        if epoch % 25 == 0 and epoch > 0:
            # Compare DDIM with different eta values
            sampled_images_det = diffusion.ddim_sample(model, n=8, eta=0.0, ddim_steps=50)
            sampled_images_stoch = diffusion.ddim_sample(model, n=8, eta=1.0, ddim_steps=50)

            save_comparison_images(sampled_images_det, sampled_images_stoch,
                                 f"ddim_comparison_epoch_{epoch}.png")

    return model, diffusion

def save_comparison_images(det_images, stoch_images, path):
    """Save comparison of deterministic vs stochastic sampling"""
    # Combine images
    combined = torch.cat([det_images, stoch_images], dim=0)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    for i in range(8):
        # Deterministic (eta=0)
        axes[0, i].imshow(det_images[i].squeeze().cpu().numpy(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Deterministic\n(η=0)', rotation=0, ha='right')

        # Stochastic (eta=1)
        axes[1, i].imshow(stoch_images[i].squeeze().cpu().numpy(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Stochastic\n(η=1)', rotation=0, ha='right')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def compare_ddpm_vs_ddim_sampling(model, diffusion):
    """Compare sampling speed and quality between DDPM and DDIM"""
    import time

    print("Comparing DDPM vs DDIM sampling:")
    print("-" * 50)

    step_counts = [1000, 250, 100, 50, 20]

    for steps in step_counts:
        # Simulate DDIM sampling time
        start_time = time.time()

        # DDIM sampling (deterministic)
        if steps <= 100:  # Only sample for reasonable step counts
            sampled_images = diffusion.ddim_sample(model, n=4, eta=0.0, ddim_steps=steps)
            end_time = time.time()

            print(f"DDIM {steps} steps: {end_time - start_time:.2f}s")
        else:
            print(f"DDIM {steps} steps: ~{steps * 0.005:.2f}s (estimated)")

def demonstrate_interpolation(model, diffusion):
    """Demonstrate DDIM interpolation capabilities"""
    print("\nDDIM Interpolation Demo:")
    print("DDIM enables smooth interpolation in latent space")
    print("This is useful for:")
    print("- Morphing between images")
    print("- Exploring latent space")
    print("- Image editing applications")

    # Generate two random noise vectors
    device = next(model.parameters()).device
    x1 = torch.randn(1, 1, 32, 32).to(device)
    x2 = torch.randn(1, 1, 32, 32).to(device)

    # Interpolate (would need trained model)
    print("\nTo see interpolation, run:")
    print("interpolated = diffusion.ddim_sample_interpolation(model, x1, x2)")

def demonstrate_ddim_properties():
    """Explain key properties of DDIM"""
    print("DDIM Key Properties:")
    print("=" * 50)
    print("1. Deterministic sampling when η=0")
    print("   - Same input noise → same output image")
    print("   - Enables reproducible generation")
    print()
    print("2. Faster sampling")
    print("   - Can skip timesteps without quality loss")
    print("   - 50 steps often sufficient vs 1000 for DDPM")
    print()
    print("3. Interpolation capability")
    print("   - Smooth interpolation in latent space")
    print("   - Useful for image editing and morphing")
    print()
    print("4. Same training as DDPM")
    print("   - No additional training required")
    print("   - Can use any DDPM-trained model")
    print()
    print("5. Trade-off parameter η:")
    print("   - η=0: Fully deterministic (fast, high quality)")
    print("   - η=1: Equivalent to DDPM (slow, high diversity)")
    print("   - 0<η<1: Balance between speed and stochasticity")

def visualize_eta_effect():
    """Visualize the effect of different eta values"""
    eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"\nEffect of η (eta) parameter:")
    print("-" * 30)
    for eta in eta_values:
        if eta == 0.0:
            description = "Fully deterministic"
        elif eta == 1.0:
            description = "Equivalent to DDPM"
        else:
            description = f"Partially stochastic ({eta*100}%)"

        print(f"η = {eta}: {description}")

if __name__ == "__main__":
    print("DDIM Example - Denoising Diffusion Implicit Models")
    print("Key improvements over DDPM:")
    print("- Faster sampling (20-250 steps vs 1000)")
    print("- Deterministic generation when η=0")
    print("- Smooth interpolation in latent space")
    print("- Same training procedure as DDPM")
    print("- Quality-speed trade-off via η parameter")

    # Demonstrate key properties
    demonstrate_ddim_properties()
    visualize_eta_effect()

    # Initialize DDIM
    ddim = DDIM()
    print(f"\nDDIM initialized with {ddim.noise_steps} timesteps")

    # Uncomment to train and sample
    # model, diffusion = train_ddim()
    #
    # # Compare different sampling strategies
    # compare_ddpm_vs_ddim_sampling(model, diffusion)
    #
    # # Demonstrate interpolation
    # demonstrate_interpolation(model, diffusion)

    print("\nDDIM enables fast, high-quality sampling with controllable stochasticity")