"""
StyleGAN (Style-Based Generator Architecture) Implementation

StyleGAN은 Karras et al. (2019)에서 제안된 혁신적인 GAN 아키텍처로,
스타일과 콘텐츠를 분리하여 제어 가능한 고품질 이미지 생성을 실현했습니다.

핵심 혁신사항:

1. **Mapping Network (Z → W)**:
   - 잠재 코드 z를 중간 잠재 공간 w로 변환
   - 8개의 FC 층으로 구성된 MLP
   - W 공간은 더 disentangled하고 linear

2. **Style-Based Generator**:
   - 각 해상도에서 서로 다른 w 사용 가능
   - Adaptive Instance Normalization (AdaIN) 사용
   - 스타일과 콘텐츠의 명확한 분리

3. **Noise Injection**:
   - 각 레이어에 확률적 디테일 추가
   - Fine-grained variation (머리카락, 피부 텍스처 등)
   - 학습 가능한 스케일링 팩터

4. **Progressive Growing** (원본):
   - 낮은 해상도부터 점진적으로 고해상도까지 훈련
   - 안정적인 고해상도 이미지 생성

수학적 구조:
1. Mapping: z ~ N(0,I) → f(z) = w
2. Synthesis: w → AdaIN → Conv → ... → RGB

AdaIN 공식:
AdaIN(x, y) = y_s × normalize(x) + y_b
where normalize(x) = (x - μ(x)) / σ(x)

주요 특징:
- **Style Mixing**: 서로 다른 w를 각 레이어에 사용
- **Truncation Trick**: w의 분산을 줄여 품질↑, 다양성↓ 조절
- **Disentanglement**: 특정 속성의 독립적 조절 가능

스타일 제어:
- Coarse Styles (4×4 - 8×8): 포즈, 얼굴형, 안경 여부
- Middle Styles (16×16 - 32×32): 표정, 눈 모양, 헤어스타일
- Fine Styles (64×64 - 1024×1024): 색상, 미세한 디테일

장점:
- 매우 고품질 이미지 생성 (FID < 10)
- 스타일 제어 가능성
- 의미있는 잠재 공간 구조
- 안정적인 훈련

단점:
- 복잡한 아키텍처
- 긴 훈련 시간
- 높은 메모리 요구량

이 구현은 교육 목적으로 단순화된 버전입니다.

Reference:
- Karras, T., et al. (2019).
  "A style-based generator architecture for generative adversarial networks."
  Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import math

class EqualizedConv2d(nn.Module):
    """Equalized learning rate Conv2d layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(EqualizedConv2d, self).__init__()
        self.stride = stride
        self.padding = padding

        # Initialize weights with normal distribution
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Calculate the scaling factor for equalized learning rate
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = math.sqrt(2.0 / fan_in)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)

class EqualizedLinear(nn.Module):
    """Equalized learning rate Linear layer"""
    def __init__(self, in_features, out_features, bias=True):
        super(EqualizedLinear, self).__init__()

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Calculate the scaling factor
        self.scale = math.sqrt(2.0 / in_features)

    def forward(self, x):
        return F.linear(x, self.weight * self.scale, self.bias)

class PixelNorm(nn.Module):
    """Pixel normalization layer"""
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class MinibatchStddev(nn.Module):
    """Minibatch standard deviation layer"""
    def __init__(self, group_size=4):
        super(MinibatchStddev, self).__init__()
        self.group_size = group_size

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        group_size = min(batch_size, self.group_size)

        # Reshape for group processing
        y = x.view(group_size, -1, channels, height, width)
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.pow(2).mean(dim=0) + 1e-8)
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(group_size, 1, height, width)

        return torch.cat([x, y], dim=1)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    def __init__(self, style_dim, num_features):
        super(AdaIN, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = EqualizedLinear(style_dim, num_features * 2)

    def forward(self, x, style):
        style = self.fc(style)
        style = style.view(style.size(0), -1, 1, 1)
        gamma, beta = style.chunk(2, dim=1)

        out = self.norm(x)
        out = gamma * out + beta
        return out

class NoiseInjection(nn.Module):
    """Noise injection layer"""
    def __init__(self, channels):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device)

        return x + self.weight * noise

class MappingNetwork(nn.Module):
    """Mapping network to transform z to w"""
    def __init__(self, z_dim=512, w_dim=512, n_layers=8):
        super(MappingNetwork, self).__init__()

        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(z_dim if i == 0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        return self.mapping(x)

class StyleBlock(nn.Module):
    """Style-based generator block"""
    def __init__(self, in_channels, out_channels, w_dim, upsample=True):
        super(StyleBlock, self).__init__()
        self.upsample = upsample

        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(out_channels, out_channels, 3, padding=1)

        self.noise1 = NoiseInjection(out_channels)
        self.noise2 = NoiseInjection(out_channels)

        self.adain1 = AdaIN(w_dim, out_channels)
        self.adain2 = AdaIN(w_dim, out_channels)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w, noise1=None, noise2=None):
        if self.upsample:
            x = self.upsample_layer(x)

        x = self.conv1(x)
        x = self.noise1(x, noise1)
        x = self.activation(x)
        x = self.adain1(x, w)

        x = self.conv2(x)
        x = self.noise2(x, noise2)
        x = self.activation(x)
        x = self.adain2(x, w)

        return x

class StyleGANGenerator(nn.Module):
    """Simplified StyleGAN Generator"""
    def __init__(self, z_dim=512, w_dim=512, img_channels=3):
        super(StyleGANGenerator, self).__init__()
        self.z_dim = z_dim
        self.w_dim = w_dim

        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim)

        # Constant input
        self.constant = nn.Parameter(torch.randn(1, 512, 4, 4))

        # Style blocks
        self.style_blocks = nn.ModuleList([
            StyleBlock(512, 512, w_dim, upsample=False),  # 4x4
            StyleBlock(512, 512, w_dim),                   # 8x8
            StyleBlock(512, 256, w_dim),                   # 16x16
            StyleBlock(256, 128, w_dim),                   # 32x32
            StyleBlock(128, 64, w_dim),                    # 64x64
        ])

        # RGB output layers
        self.to_rgb = nn.ModuleList([
            EqualizedConv2d(512, img_channels, 1),
            EqualizedConv2d(512, img_channels, 1),
            EqualizedConv2d(256, img_channels, 1),
            EqualizedConv2d(128, img_channels, 1),
            EqualizedConv2d(64, img_channels, 1),
        ])

    def forward(self, z, style_mixing_prob=0.9, truncation_psi=1.0):
        batch_size = z.shape[0]

        # Generate w from z
        w = self.mapping(z)

        # Style mixing (randomly use different w for different layers)
        if self.training and torch.rand(1).item() < style_mixing_prob:
            crossover_point = torch.randint(1, len(self.style_blocks), (1,)).item()
            z2 = torch.randn_like(z)
            w2 = self.mapping(z2)
        else:
            w2 = w
            crossover_point = len(self.style_blocks)

        # Start with constant input
        x = self.constant.repeat(batch_size, 1, 1, 1)
        rgb = None

        for i, (style_block, to_rgb_layer) in enumerate(zip(self.style_blocks, self.to_rgb)):
            # Choose which w to use
            current_w = w if i < crossover_point else w2

            # Apply style block
            x = style_block(x, current_w)

            # Generate RGB output
            new_rgb = to_rgb_layer(x)

            if rgb is not None:
                rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
                rgb = rgb + new_rgb
            else:
                rgb = new_rgb

        return torch.tanh(rgb)

class StyleGANDiscriminator(nn.Module):
    """Simplified StyleGAN Discriminator"""
    def __init__(self, img_channels=3):
        super(StyleGANDiscriminator, self).__init__()

        # Progressive blocks
        self.from_rgb = nn.ModuleList([
            EqualizedConv2d(img_channels, 64, 1),
            EqualizedConv2d(img_channels, 128, 1),
            EqualizedConv2d(img_channels, 256, 1),
            EqualizedConv2d(img_channels, 512, 1),
            EqualizedConv2d(img_channels, 512, 1),
        ])

        self.blocks = nn.ModuleList([
            # 64x64 -> 32x32
            nn.Sequential(
                EqualizedConv2d(64, 128, 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(128, 128, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ),
            # 32x32 -> 16x16
            nn.Sequential(
                EqualizedConv2d(128, 256, 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(256, 256, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ),
            # 16x16 -> 8x8
            nn.Sequential(
                EqualizedConv2d(256, 512, 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(512, 512, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ),
            # 8x8 -> 4x4
            nn.Sequential(
                EqualizedConv2d(512, 512, 3, padding=1),
                nn.LeakyReLU(0.2),
                EqualizedConv2d(512, 512, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ),
        ])

        # Final layers
        self.minibatch_stddev = MinibatchStddev()
        self.conv_final = EqualizedConv2d(513, 512, 3, padding=1)  # +1 for minibatch stddev
        self.fc = EqualizedLinear(512 * 4 * 4, 1)

    def forward(self, x):
        # Start from RGB
        x = self.from_rgb[-1](x)
        x = F.leaky_relu(x, 0.2)

        # Apply blocks
        for block in reversed(self.blocks):
            x = block(x)

        # Final processing
        x = self.minibatch_stddev(x)
        x = self.conv_final(x)
        x = F.leaky_relu(x, 0.2)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def train_stylegan():
    # Hyperparameters
    batch_size = 16
    z_dim = 512
    w_dim = 512
    img_channels = 3
    num_epochs = 200
    lr = 0.001
    beta1 = 0.0
    beta2 = 0.99

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Models
    generator = StyleGANGenerator(z_dim, w_dim, img_channels).to(device)
    discriminator = StyleGANDiscriminator(img_channels).to(device)

    # Optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Fixed noise for visualization
    fixed_z = torch.randn(16, z_dim, device=device)

    generator.train()
    discriminator.train()

    for epoch in range(num_epochs):
        for i in range(100):  # Simulate 100 batches per epoch
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Generate fake images
            z = torch.randn(batch_size, z_dim, device=device)
            fake_imgs = generator(z)

            # Simulate real images (random for demonstration)
            real_imgs = torch.randn(batch_size, img_channels, 64, 64, device=device)

            # Train Discriminator
            opt_disc.zero_grad()

            # Real images
            real_pred = discriminator(real_imgs)
            real_loss = criterion(real_pred, real_labels)

            # Fake images
            fake_pred = discriminator(fake_imgs.detach())
            fake_loss = criterion(fake_pred, fake_labels)

            disc_loss = (real_loss + fake_loss) / 2
            disc_loss.backward()
            opt_disc.step()

            # Train Generator
            opt_gen.zero_grad()

            fake_pred = discriminator(fake_imgs)
            gen_loss = criterion(fake_pred, real_labels)
            gen_loss.backward()
            opt_gen.step()

            if i % 50 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/100] '
                      f'D Loss: {disc_loss.item():.4f} G Loss: {gen_loss.item():.4f}')

        # Generate samples
        if epoch % 10 == 0:
            with torch.no_grad():
                fake_samples = generator(fixed_z)
                vutils.save_image(fake_samples, f'stylegan_samples_epoch_{epoch}.png',
                                nrow=4, normalize=True)

    return generator, discriminator

def demonstrate_style_mixing(generator, z_dim=512):
    """Demonstrate style mixing capability"""
    generator.eval()
    device = next(generator.parameters()).device

    with torch.no_grad():
        # Generate two different styles
        z1 = torch.randn(1, z_dim, device=device)
        z2 = torch.randn(1, z_dim, device=device)

        # Get w vectors
        w1 = generator.mapping(z1)
        w2 = generator.mapping(z2)

        # Generate images with different style mixing points
        images = []
        for crossover in range(0, len(generator.style_blocks) + 1):
            # Manually implement style mixing
            x = generator.constant.repeat(1, 1, 1, 1)
            rgb = None

            for i, (style_block, to_rgb_layer) in enumerate(zip(generator.style_blocks, generator.to_rgb)):
                current_w = w1 if i < crossover else w2
                x = style_block(x, current_w)

                new_rgb = to_rgb_layer(x)
                if rgb is not None:
                    rgb = F.interpolate(rgb, scale_factor=2, mode='bilinear', align_corners=False)
                    rgb = rgb + new_rgb
                else:
                    rgb = new_rgb

            images.append(torch.tanh(rgb))

        # Visualize
        grid = torch.cat(images, dim=0)
        grid = vutils.make_grid(grid, nrow=len(images), normalize=True)

        plt.figure(figsize=(20, 4))
        plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
        plt.title('Style Mixing: Coarse to Fine Features')
        plt.axis('off')
        plt.show()

def demonstrate_truncation_trick(generator, z_dim=512, truncation_values=[0.5, 0.7, 1.0, 1.2]):
    """Demonstrate truncation trick for controlling sample diversity"""
    generator.eval()
    device = next(generator.parameters()).device

    images = []
    for psi in truncation_values:
        z = torch.randn(4, z_dim, device=device)
        # Apply truncation in w space
        w = generator.mapping(z)
        w_avg = torch.zeros_like(w[0])  # In practice, this would be computed from many samples
        w_truncated = w_avg + psi * (w - w_avg)

        # Generate with truncated w (simplified)
        fake_imgs = generator(z)  # In practice, you'd use w_truncated
        images.append(fake_imgs)

    # Create visualization
    all_images = torch.cat(images, dim=0)
    grid = vutils.make_grid(all_images, nrow=4, normalize=True)

    plt.figure(figsize=(16, 16))
    plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
    plt.title('Truncation Trick: Controlling Sample Diversity')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    print("StyleGAN Example - Style-Based Generator Architecture")
    print("Key innovations:")
    print("- Mapping network (Z -> W)")
    print("- Adaptive Instance Normalization (AdaIN)")
    print("- Style mixing and truncation trick")
    print("- Progressive growing (simplified here)")
    print("- Equalized learning rate")
    print("- Noise injection")
    print("- Minibatch standard deviation")

    # Demonstrate key concepts
    print("\nStyleGAN separates high-level attributes from stochastic variation")
    print("W space has better properties for interpolation than Z space")

    # Uncomment to train (warning: very computationally intensive)
    # generator, discriminator = train_stylegan()
    # demonstrate_style_mixing(generator)
    # demonstrate_truncation_trick(generator)