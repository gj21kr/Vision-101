"""
DDPM (Denoising Diffusion Probabilistic Models) Implementation

이 파일은 Ho et al. (2020)에서 제안한 DDPM의 완전한 구현을 제공합니다.
DDPM은 forward diffusion 과정에서 점진적으로 노이즈를 추가하고,
reverse diffusion 과정에서 신경망을 통해 노이즈를 제거하여 데이터를 생성합니다.

핵심 개념:
1. Forward Process: q(x_t | x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
2. Reverse Process: p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t² I)
3. Training Objective: E_t,x_0,ε [||ε - ε_θ(x_t, t)||²]

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
from tqdm import tqdm

class TimeEmbedding(nn.Module):
    """
    시간 정보를 위한 Sinusoidal Position Embedding

    DDPM에서는 각 타임스텝 t에 대한 정보를 네트워크에 제공해야 합니다.
    이 클래스는 Transformer에서 사용되는 sinusoidal position encoding과
    유사한 방식으로 시간 정보를 인코딩합니다.

    수식:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    Args:
        dim (int): 임베딩 차원

    Returns:
        torch.Tensor: 시간 임베딩 벡터 [batch_size, dim]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        시간 정보를 sinusoidal 임베딩으로 변환

        Args:
            time (torch.Tensor): 타임스텝 [batch_size,]

        Returns:
            torch.Tensor: 시간 임베딩 [batch_size, dim]
        """
        device = time.device
        half_dim = self.dim // 2

        # 주파수 계산: 1/10000^(2i/dim) for i in [0, half_dim)
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # 각 타임스텝에 대해 주파수 적용
        embeddings = time[:, None] * embeddings[None, :]

        # sin과 cos을 concat하여 최종 임베딩 생성
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
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
    """Self-attention block"""
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

class UNet(nn.Module):
    """U-Net architecture for DDPM"""
    def __init__(self, in_channels=1, model_channels=128, out_channels=1,
                 num_res_blocks=2, attention_resolutions=[16],
                 channel_mult=[1, 2, 2, 2], num_heads=8, time_emb_dim=512):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult

        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down sampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [model_channels]

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResidualBlock(ch, mult * model_channels, time_emb_dim)]
                ch = mult * model_channels

                # Add attention at specified resolutions
                if 32 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            # Downsample (except last level)
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

                # Add attention at specified resolutions
                if 32 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                # Upsample (except last iteration of last level)
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
        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Initial conv
        x = self.input_conv(x)

        # Store skip connections
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

class DDPM:
    """Denoising Diffusion Probabilistic Model"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Precompute noise schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # For efficient sampling
        self.alpha_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.beta * (1. - self.alpha_hat_prev) / (1. - self.alpha_hat)

    def prepare_noise_schedule(self):
        """Linear noise schedule"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        Forward diffusion process: 깨끗한 이미지에 노이즈 추가

        이 함수는 DDPM의 핵심인 forward diffusion process를 구현합니다.
        임의의 타임스텝 t에서 원본 이미지 x_0에 직접 노이즈를 추가할 수 있습니다.

        수학적 정의:
            q(x_t | x_0) = N(x_t; √(ᾱ_t) * x_0, (1 - ᾱ_t) * I)

        실제 구현:
            x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
            where ε ~ N(0, I)

        Args:
            x (torch.Tensor): 원본 이미지 [batch_size, channels, height, width]
            t (torch.Tensor): 타임스텝 [batch_size,]

        Returns:
            tuple: (노이즈가 추가된 이미지, 추가된 노이즈)
                - x_t: 타임스텝 t에서의 노이즈 이미지
                - ε: 추가된 가우시안 노이즈 (신경망이 예측해야 할 목표)
        """
        # ᾱ_t (alpha hat): 누적곱으로 계산된 노이즈 스케줄
        # [batch_size, 1, 1, 1] 형태로 reshape하여 broadcasting 가능하게 함
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        # 가우시안 노이즈 생성 (평균=0, 분산=1)
        Ɛ = torch.randn_like(x)

        # Forward diffusion 공식 적용
        # x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ

        return x_noisy, Ɛ

    def sample_timesteps(self, n):
        """Sample random timesteps"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, channels=1):
        """Sample new images using trained model"""
        print(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            # Start from random noise
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)

            # Reverse diffusion process
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)

                # Predict noise
                predicted_noise = model(x, t)

                # Compute denoising step
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

def train_ddpm():
    """Train DDPM model"""
    # Hyperparameters
    batch_size = 16
    epochs = 500
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
    diffusion = DDPM(noise_steps=noise_steps, img_size=img_size, device=device)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training loop
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

        # Sample images every 50 epochs
        if epoch % 50 == 0 and epoch > 0:
            sampled_images = diffusion.sample(model, n=16)
            save_images(sampled_images, f"ddpm_samples_epoch_{epoch}.png")

    return model, diffusion

def save_images(images, path):
    """Save sampled images"""
    grid = torchvision.utils.make_grid(images, nrow=4, normalize=False)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def demonstrate_noise_schedule():
    """Visualize the noise schedule and forward process"""
    ddpm = DDPM()

    # Plot beta schedule
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(ddpm.beta.cpu())
    plt.title('Beta Schedule')
    plt.xlabel('Timestep')
    plt.ylabel('Beta')

    plt.subplot(1, 3, 2)
    plt.plot(ddpm.alpha.cpu())
    plt.title('Alpha = 1 - Beta')
    plt.xlabel('Timestep')
    plt.ylabel('Alpha')

    plt.subplot(1, 3, 3)
    plt.plot(ddpm.alpha_hat.cpu())
    plt.title('Alpha Hat (Cumulative Product)')
    plt.xlabel('Timestep')
    plt.ylabel('Alpha Hat')

    plt.tight_layout()
    plt.show()

def visualize_forward_process(ddpm, image, timesteps=[0, 250, 500, 750, 999]):
    """Visualize how image gets corrupted over time"""
    fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 3))

    for i, t in enumerate(timesteps):
        if t == 0:
            noisy_image = image
        else:
            t_tensor = torch.tensor([t])
            noisy_image, _ = ddpm.noise_images(image.unsqueeze(0), t_tensor)
            noisy_image = noisy_image.squeeze(0)

        # Denormalize for visualization
        noisy_image = (noisy_image + 1) / 2

        axes[i].imshow(noisy_image.squeeze().cpu().numpy(), cmap='gray')
        axes[i].set_title(f't = {t}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def compare_sampling_strategies(model, diffusion):
    """Compare different sampling strategies"""
    print("DDPM uses the full reverse process with 1000 steps")
    print("Each step requires a neural network forward pass")
    print("This makes sampling slow but high quality")

    # Sample with different number of steps (simplified)
    step_counts = [1000, 500, 100, 50]

    for steps in step_counts:
        print(f"Sampling with {steps} steps would take ~{steps * 0.01:.2f} seconds")

if __name__ == "__main__":
    print("DDPM Example - Denoising Diffusion Probabilistic Models")
    print("Key features:")
    print("- Forward diffusion process adds Gaussian noise")
    print("- Reverse diffusion process learned by U-Net")
    print("- Sinusoidal time embeddings")
    print("- Attention blocks at specific resolutions")
    print("- Residual connections with time conditioning")

    # Demonstrate concepts
    demonstrate_noise_schedule()

    # Initialize DDPM
    ddpm = DDPM()
    print(f"\nDDPM initialized with {ddpm.noise_steps} timesteps")
    print(f"Beta range: {ddpm.beta_start} to {ddpm.beta_end}")

    # Uncomment to train (very computationally intensive!)
    # model, diffusion = train_ddpm()
    # sampled_images = diffusion.sample(model, n=16)

    print("\nDDPM produces high-quality samples but requires many sampling steps")