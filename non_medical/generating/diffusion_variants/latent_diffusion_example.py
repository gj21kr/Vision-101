"""
Latent Diffusion Model (LDM) Implementation

Latent Diffusion Model은 Rombach et al. (2022)에서 제안된 혁신적인 방법으로,
픽셀 공간 대신 VAE의 잠재 공간에서 diffusion을 수행하여 계산 효율성을 크게 향상시켰습니다.

핵심 아이디어:

1. **Two-Stage Training**:
   - Stage 1: VAE 훈련 (이미지 ↔ 잠재 표현)
   - Stage 2: 잠재 공간에서 Diffusion 모델 훈련

2. **Perceptually Equivalent Latent Space**:
   - 높은 압축률 (8×8 latent for 64×64 image)
   - 인간의 지각과 무관한 디테일 제거
   - 의미적으로 중요한 정보만 보존

3. **Computational Efficiency**:
   - 512×512 → 64×64 잠재 공간 (64배 압축)
   - 메모리 및 계산량 대폭 감소
   - 고해상도 이미지 생성 가능

구조:
┌─────────┐    ┌──────────┐    ┌─────────┐
│ Encoder │───▶│ Diffusion │───▶│ Decoder │
│  E(x)   │    │    U-Net  │    │  D(z)   │
│ x → z   │    │  ε_θ(z_t) │    │ z → x   │
└─────────┘    └──────────┘    └─────────┘

수학적 정의:
1. VAE 손실: L_VE = ||x - D(E(x))||² + KL(q(z|x)||p(z))
2. LDM 손실: L_LDM = E_t,z,ε [||ε - ε_θ(z_t, t)||²]
   where z = E(x), z_t = √(ᾱ_t)z + √(1-ᾱ_t)ε

장점:
- **메모리 효율성**: 16-64배 적은 VRAM 사용
- **속도**: 빠른 훈련 및 추론
- **확장성**: 고해상도 (1024×1024+) 이미지 생성 가능
- **의미적 압축**: 중요한 정보만 유지

단점:
- **Two-stage 훈련**: 복잡한 훈련 과정
- **VAE 의존성**: VAE 품질이 최종 결과에 영향
- **잠재 공간 제약**: VAE의 표현 능력에 제한

응용:
- Stable Diffusion (Text-to-Image)
- 고해상도 이미지 생성
- 이미지 편집 (inpainting, outpainting)
- 실시간 생성 (상대적으로)

Reference:
- Rombach, R., et al. (2022).
  "High-resolution image synthesis with latent diffusion models."
  Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
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

# Import components from previous examples
from ddpm_example import UNet, TimeEmbedding

class Encoder(nn.Module):
    """VAE-style encoder for Latent Diffusion"""
    def __init__(self, in_channels=1, latent_channels=4, base_channels=128):
        super().__init__()

        # Downsampling path
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Down blocks
        self.down_blocks = nn.ModuleList([
            # 32x32 -> 16x16
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 2),
                nn.SiLU(),
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
                nn.GroupNorm(8, base_channels * 2),
                nn.SiLU(),
            ),
            # 16x16 -> 8x8
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 4),
                nn.SiLU(),
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                nn.GroupNorm(8, base_channels * 4),
                nn.SiLU(),
            ),
        ])

        # Middle block
        self.mid_block = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )

        # Output layers for mean and logvar
        self.conv_out = nn.Conv2d(base_channels * 4, latent_channels * 2, 3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)

        # Downsampling
        for down_block in self.down_blocks:
            x = down_block(x)

        # Middle
        x = self.mid_block(x)

        # Output
        moments = self.conv_out(x)
        mean, logvar = torch.chunk(moments, 2, dim=1)

        return mean, logvar

class Decoder(nn.Module):
    """VAE-style decoder for Latent Diffusion"""
    def __init__(self, latent_channels=4, out_channels=1, base_channels=128):
        super().__init__()

        # Input layer
        self.conv_in = nn.Conv2d(latent_channels, base_channels * 4, 3, padding=1)

        # Middle block
        self.mid_block = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
        )

        # Up blocks
        self.up_blocks = nn.ModuleList([
            # 8x8 -> 16x16
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
                nn.GroupNorm(8, base_channels * 2),
                nn.SiLU(),
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
                nn.GroupNorm(8, base_channels * 2),
                nn.SiLU(),
            ),
            # 16x16 -> 32x32
            nn.Sequential(
                nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
                nn.GroupNorm(8, base_channels),
                nn.SiLU(),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.GroupNorm(8, base_channels),
                nn.SiLU(),
            ),
        ])

        # Output layer
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, z):
        x = self.conv_in(z)

        # Middle
        x = self.mid_block(x)

        # Upsampling
        for up_block in self.up_blocks:
            x = up_block(x)

        # Output
        x = self.conv_out(x)
        return torch.tanh(x)

class LatentUNet(nn.Module):
    """U-Net that operates in latent space"""
    def __init__(self, in_channels=4, model_channels=320, out_channels=4,
                 num_res_blocks=2, attention_resolutions=[4, 2],
                 channel_mult=[1, 2, 4], time_emb_dim=1280):
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
                layers = []

                # Residual block
                layers.append(nn.Sequential(
                    nn.GroupNorm(32, ch),
                    nn.SiLU(),
                    nn.Conv2d(ch, mult * model_channels, 3, padding=1),
                    nn.GroupNorm(32, mult * model_channels),
                    nn.SiLU(),
                    nn.Conv2d(mult * model_channels, mult * model_channels, 3, padding=1),
                ))

                ch = mult * model_channels

                # Add attention at specified resolutions
                if 8 // (2 ** level) in attention_resolutions:
                    layers.append(SelfAttention(ch))

                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)

        # Middle
        self.middle_block = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                nn.Conv2d(ch, ch, 3, padding=1),
            ),
            SelfAttention(ch),
            nn.Sequential(
                nn.GroupNorm(32, ch),
                nn.SiLU(),
                nn.Conv2d(ch, ch, 3, padding=1),
            ),
        ])

        # Up sampling
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = []

                # Residual block
                layers.append(nn.Sequential(
                    nn.GroupNorm(32, ch + ich),
                    nn.SiLU(),
                    nn.Conv2d(ch + ich, mult * model_channels, 3, padding=1),
                    nn.GroupNorm(32, mult * model_channels),
                    nn.SiLU(),
                    nn.Conv2d(mult * model_channels, mult * model_channels, 3, padding=1),
                ))

                ch = mult * model_channels

                # Add attention at specified resolutions
                if 8 // (2 ** level) in attention_resolutions:
                    layers.append(SelfAttention(ch))

                # Upsample (except last iteration of last level)
                if level != 0 and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

                self.up_blocks.append(nn.ModuleList(layers))

        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, model_channels),
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
                if isinstance(layer[0], nn.GroupNorm):  # Residual block
                    residual = x
                    x = layer(x)
                    # Add residual connection
                    if x.shape[1] != residual.shape[1]:
                        residual = F.conv2d(residual, torch.eye(x.shape[1], residual.shape[1]).to(x.device)[:, :, None, None])
                    x = x + residual
                else:
                    x = layer(x)
            skip_connections.append(x)

        # Middle
        for layer in self.middle_block:
            if isinstance(layer, SelfAttention):
                x = layer(x)
            else:
                residual = x
                x = layer(x)
                x = x + residual

        # Up sampling
        for layers in self.up_blocks:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)

            for layer in layers:
                if isinstance(layer[0], nn.GroupNorm):  # Residual block
                    residual = x
                    x = layer(x)
                    # Handle channel mismatch for residual connection
                    if x.shape[1] != residual.shape[1]:
                        residual = F.conv2d(residual, torch.eye(x.shape[1], residual.shape[1]).to(x.device)[:, :, None, None])
                    x = x + residual
                else:
                    x = layer(x)

        # Output
        return self.output_conv(x)

class SelfAttention(nn.Module):
    """Self-attention for latent space"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        residual = x
        B, C, H, W = x.shape

        x = self.norm(x)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        # Attention
        attn = torch.matmul(q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.proj_out(out)

        return out + residual

class LatentDiffusion:
    """Latent Diffusion Model"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 img_size=32, latent_size=8, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.latent_size = latent_size
        self.device = device

        # Precompute noise schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """Linear noise schedule"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def encode_to_latent(self, encoder, x):
        """Encode images to latent space"""
        with torch.no_grad():
            mean, logvar = encoder(x)
            # Sample from posterior
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            return z

    def decode_from_latent(self, decoder, z):
        """Decode from latent space to images"""
        with torch.no_grad():
            x = decoder(z)
            return x

    def noise_latents(self, z, t):
        """Add noise to latents"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(z)
        return sqrt_alpha_hat * z + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        """Sample random timesteps"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, unet, decoder, n, latent_channels=4, ddim_steps=50):
        """Sample new images using latent diffusion"""
        print(f"Sampling {n} images in latent space...")
        unet.eval()
        decoder.eval()

        # Sample in latent space
        with torch.no_grad():
            # Start from random noise in latent space
            z = torch.randn((n, latent_channels, self.latent_size, self.latent_size)).to(self.device)

            # DDIM sampling in latent space
            skip = self.noise_steps // ddim_steps
            seq = range(0, self.noise_steps, skip)
            seq = list(reversed(seq))

            for i, j in enumerate(tqdm(seq[:-1])):
                t = torch.full((n,), j, dtype=torch.long).to(self.device)
                next_t = torch.full((n,), seq[i + 1], dtype=torch.long).to(self.device)

                # Predict noise
                predicted_noise = unet(z, t)

                # DDIM step
                alpha_t = self.alpha_hat[t][:, None, None, None]
                alpha_t_next = self.alpha_hat[next_t][:, None, None, None]

                pred_z0 = (z - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                direction = torch.sqrt(1 - alpha_t_next) * predicted_noise
                z = torch.sqrt(alpha_t_next) * pred_z0 + direction

            # Decode to image space
            images = decoder(z)

        unet.train()
        decoder.train()

        # Denormalize
        images = (images.clamp(-1, 1) + 1) / 2
        images = (images * 255).type(torch.uint8)
        return images

def train_latent_diffusion():
    """Train Latent Diffusion Model"""
    # Hyperparameters
    batch_size = 8
    epochs = 100
    lr = 1e-4
    img_size = 32
    latent_size = 8
    latent_channels = 4
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

    # Models
    encoder = Encoder(in_channels=1, latent_channels=latent_channels).to(device)
    decoder = Decoder(latent_channels=latent_channels, out_channels=1).to(device)
    unet = LatentUNet(in_channels=latent_channels, out_channels=latent_channels).to(device)

    # Optimizers
    optimizer_ae = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    optimizer_unet = optim.AdamW(unet.parameters(), lr=lr)

    # Loss functions
    mse = nn.MSELoss()
    diffusion = LatentDiffusion(noise_steps=noise_steps, img_size=img_size,
                               latent_size=latent_size, device=device)

    print(f"Encoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # Training loop
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        ae_loss_epoch = 0
        diffusion_loss_epoch = 0

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)

            # Phase 1: Train autoencoder
            if epoch < 10:  # Pre-train autoencoder for first few epochs
                optimizer_ae.zero_grad()

                # Encode to latent
                mean, logvar = encoder(images)
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mean + eps * std

                # Decode back to image
                reconstructed = decoder(z)

                # Reconstruction loss
                recon_loss = mse(reconstructed, images)

                # KL divergence loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
                kl_loss /= images.numel()

                # Total autoencoder loss
                ae_loss = recon_loss + 0.1 * kl_loss
                ae_loss.backward()
                optimizer_ae.step()

                ae_loss_epoch += ae_loss.item()
                pbar.set_postfix(AE_Loss=ae_loss.item())

            else:
                # Phase 2: Train diffusion model in latent space
                optimizer_unet.zero_grad()

                # Encode to latent space (no gradient)
                with torch.no_grad():
                    mean, logvar = encoder(images)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mean + eps * std

                # Sample timesteps
                t = diffusion.sample_timesteps(z.shape[0]).to(device)

                # Add noise to latents
                z_noisy, noise = diffusion.noise_latents(z, t)

                # Predict noise
                predicted_noise = unet(z_noisy, t)

                # Diffusion loss
                diff_loss = mse(noise, predicted_noise)
                diff_loss.backward()
                optimizer_unet.step()

                diffusion_loss_epoch += diff_loss.item()
                pbar.set_postfix(Diff_Loss=diff_loss.item())

        if epoch < 10:
            print(f"Epoch {epoch} | AE Loss: {ae_loss_epoch/len(dataloader):.4f}")
        else:
            print(f"Epoch {epoch} | Diffusion Loss: {diffusion_loss_epoch/len(dataloader):.4f}")

        # Sample images every 10 epochs
        if epoch % 10 == 0 and epoch >= 10:
            sampled_images = diffusion.sample(unet, decoder, n=16, ddim_steps=20)
            save_images(sampled_images, f"latent_diffusion_samples_epoch_{epoch}.png")

    return encoder, decoder, unet, diffusion

def save_images(images, path):
    """Save sampled images"""
    import torchvision.utils as utils
    grid = utils.make_grid(images.float() / 255.0, nrow=4, normalize=False)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def demonstrate_latent_advantages():
    """Explain advantages of latent diffusion"""
    print("Latent Diffusion Model Advantages:")
    print("=" * 50)
    print("1. Computational Efficiency")
    print("   - Works in lower-dimensional latent space")
    print("   - ~8x8 latents vs 32x32 images = 16x fewer dimensions")
    print("   - Faster training and inference")
    print()
    print("2. Memory Efficiency")
    print("   - Smaller U-Net for latent space")
    print("   - Can handle larger images by working in compressed space")
    print("   - Enables high-resolution generation")
    print()
    print("3. Semantic Richness")
    print("   - VAE latent space captures meaningful features")
    print("   - Better disentanglement than pixel space")
    print("   - Enables better interpolation and editing")
    print()
    print("4. Stable Training")
    print("   - Pre-trained autoencoder provides stable latent space")
    print("   - Diffusion model focuses on semantic generation")
    print("   - Less mode collapse than GANs")

def compare_pixel_vs_latent():
    """Compare pixel-space vs latent-space diffusion"""
    print("\nPixel-space vs Latent-space Diffusion:")
    print("-" * 50)

    print("Pixel-space (DDPM):")
    print("- Direct generation in image space")
    print("- High computational cost")
    print("- Works well for smaller images")
    print("- Memory: O(H × W × C)")

    print("\nLatent-space (LDM):")
    print("- Generation in compressed latent space")
    print("- Lower computational cost")
    print("- Scales to high-resolution images")
    print("- Memory: O(h × w × c) where h,w << H,W")

    print("\nTypical compression ratios:")
    print("- 512×512 image → 64×64 latent (8× downsampling)")
    print("- 1024×1024 image → 128×128 latent (8× downsampling)")

if __name__ == "__main__":
    print("Latent Diffusion Model Example")
    print("Key innovations:")
    print("- Diffusion in VAE latent space instead of pixel space")
    print("- Encoder-Decoder architecture for compression")
    print("- U-Net operates on latent representations")
    print("- Significant computational savings")
    print("- Enables high-resolution image generation")

    # Demonstrate advantages
    demonstrate_latent_advantages()
    compare_pixel_vs_latent()

    # Initialize model
    print(f"\nLatent Diffusion reduces computation by ~16x")
    print("(32×32 image → 8×8 latent = 16× fewer dimensions)")

    # Uncomment to train (computationally intensive!)
    # encoder, decoder, unet, diffusion = train_latent_diffusion()
    # sampled_images = diffusion.sample(unet, decoder, n=16)

    print("\nLatent Diffusion enables efficient high-quality generation")