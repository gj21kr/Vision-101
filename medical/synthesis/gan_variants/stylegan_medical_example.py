"""
Medical StyleGAN Implementation (Simplified)

의료 이미지 생성을 위한 StyleGAN 구현입니다.
StyleGAN은 스타일과 콘텐츠를 분리하여 제어 가능한 고품질 의료 이미지 생성을 제공합니다.

의료 분야에서의 활용:
- 다양한 스타일의 의료 이미지 생성 (예: 다른 촬영 조건, 다른 장비)
- 특정 의료 특징을 강조한 이미지 생성
- 의료 이미지 augmentation을 위한 다양성 제공

의료 특화 기능:
- 의료 이미지 특화 데이터 로더
- 자동 결과 저장 및 로깅 시스템
- 스타일 혼합 및 제어 가능한 생성
- 의료 이미지 품질 평가 메트릭

StyleGAN의 핵심 특징:
1. Mapping Network: Z → W 변환으로 더 disentangled 공간
2. AdaIN (Adaptive Instance Normalization): 스타일 제어
3. Noise Injection: 세밀한 디테일 추가
4. Progressive Growing: 안정적인 고해상도 생성

이 구현은 교육 목적으로 단순화되었습니다.

Reference:
- Karras, T., et al. (2019).
  "A style-based generator architecture for generative adversarial networks."
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

from ...medical_data_utils import (
    MedicalImageLoader,
    load_chest_xray_data,
    load_brain_mri_data,
)
from ...result_logger import create_logger_for_generating

class EqualizedLinear(nn.Module):
    """Equalized learning rate Linear layer"""
    def __init__(self, in_features, out_features, bias=True, lr_mul=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr_mul = lr_mul

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Scale for equalized learning rate
        self.scale = (1.0 / math.sqrt(in_features)) * lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.scale,
                       self.bias * self.lr_mul if self.bias is not None else None)

class EqualizedConv2d(nn.Module):
    """Equalized learning rate Conv2d layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Scale for equalized learning rate
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = math.sqrt(2.0 / fan_in)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)

class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    def __init__(self, in_channels, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channels)
        self.style_scale = EqualizedLinear(style_dim, in_channels)
        self.style_bias = EqualizedLinear(style_dim, in_channels)

    def forward(self, x, style):
        """
        Apply AdaIN: AdaIN(x, y) = y_s × normalize(x) + y_b

        Args:
            x: Feature maps [B, C, H, W]
            style: Style vector [B, style_dim]
        """
        normalized = self.norm(x)  # Instance normalization
        style_scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(style).unsqueeze(2).unsqueeze(3)

        return style_scale * normalized + style_bias

class NoiseInjection(nn.Module):
    """Stochastic variation through noise injection"""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))  # Learnable scaling factor

    def forward(self, x, noise=None):
        if noise is None:
            batch, _, height, width = x.shape
            noise = torch.randn(batch, 1, height, width, device=x.device, dtype=x.dtype)

        return x + self.weight * noise

class StyleGANMappingNetwork(nn.Module):
    """
    Mapping Network: Z → W

    잠재 코드 z를 중간 잠재 공간 w로 변환하여
    더 disentangled하고 linear한 공간을 제공
    """
    def __init__(self, latent_dim=512, style_dim=512, n_layers=8):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.extend([
                EqualizedLinear(latent_dim if i == 0 else style_dim, style_dim),
                nn.LeakyReLU(0.2)
            ])

        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        """
        Map random latent code to style space

        Args:
            z: Random latent code [B, latent_dim]

        Returns:
            w: Style code [B, style_dim]
        """
        return self.mapping(z)

class StyleGANGenerator(nn.Module):
    """Simplified StyleGAN Generator for medical images"""
    def __init__(self, style_dim=512, n_layers=4, img_channels=1):
        super().__init__()
        self.style_dim = style_dim
        self.n_layers = n_layers

        # Constant input (learned)
        self.constant_input = nn.Parameter(torch.randn(1, 512, 4, 4))

        # Style-controlled layers
        self.style_blocks = nn.ModuleList()
        self.noise_injections = nn.ModuleList()
        self.adain_layers = nn.ModuleList()

        # Progressive upsampling layers
        channels = [512, 256, 128, 64]

        for i in range(n_layers):
            # Style block (conv layer)
            if i == 0:
                self.style_blocks.append(
                    EqualizedConv2d(512, channels[i], 3, padding=1)
                )
            else:
                self.style_blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    EqualizedConv2d(channels[i-1], channels[i], 3, padding=1)
                ))

            # Noise injection
            self.noise_injections.append(NoiseInjection())

            # AdaIN for style control
            self.adain_layers.append(AdaIN(channels[i], style_dim))

        # Final RGB conversion
        self.to_rgb = EqualizedConv2d(channels[-1], img_channels, 1)

    def forward(self, styles, noise=None):
        """
        Generate medical images with style control

        Args:
            styles: Style vectors [B, style_dim] or list of [B, style_dim]
            noise: Optional noise for injection

        Returns:
            Generated medical images [B, channels, H, W]
        """
        batch_size = styles[0].shape[0] if isinstance(styles, list) else styles.shape[0]

        # Start with constant input
        x = self.constant_input.repeat(batch_size, 1, 1, 1)

        # Apply style-controlled layers
        for i, (style_block, noise_inject, adain) in enumerate(
            zip(self.style_blocks, self.noise_injections, self.adain_layers)):

            x = style_block(x)
            x = noise_inject(x)

            # Use different styles for different layers (style mixing)
            if isinstance(styles, list):
                style = styles[i] if i < len(styles) else styles[-1]
            else:
                style = styles

            x = adain(x, style)
            x = F.leaky_relu(x, 0.2)

        # Convert to RGB
        rgb = torch.tanh(self.to_rgb(x))

        return rgb

class StyleGANDiscriminator(nn.Module):
    """Simplified discriminator for StyleGAN"""
    def __init__(self, img_channels=1):
        super().__init__()

        channels = [img_channels, 64, 128, 256, 512]

        layers = []
        for i in range(len(channels) - 1):
            layers.extend([
                EqualizedConv2d(channels[i], channels[i+1], 3, stride=2, padding=1),
                nn.LeakyReLU(0.2),
            ])

        self.conv_layers = nn.Sequential(*layers)
        self.final_conv = EqualizedConv2d(512, 1, 4)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        return x.view(x.size(0), -1)

class MedicalImageDataset(Dataset):
    def __init__(self, images, image_size=64, channels=1):
        self.images = images
        self.image_size = image_size
        self.channels = channels

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * channels, (0.5,) * channels)
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

def style_mixing(generator, mapping_net, batch_size, style_dim, device, mixing_prob=0.9):
    """
    Style mixing: 서로 다른 레이어에 다른 스타일 적용

    StyleGAN의 핵심 기능으로, 서로 다른 이미지의 스타일을 혼합하여
    더 다양하고 흥미로운 결과를 생성할 수 있습니다.
    """
    if np.random.random() < mixing_prob:
        # 두 개의 다른 잠재 코드 생성
        z1 = torch.randn(batch_size, style_dim, device=device)
        z2 = torch.randn(batch_size, style_dim, device=device)

        # 스타일 공간으로 변환
        w1 = mapping_net(z1)
        w2 = mapping_net(z2)

        # 랜덤하게 어느 레이어에서 스타일을 바꿀지 결정
        crossover_point = np.random.randint(1, generator.n_layers)

        # Style mixing: 처음 몇 레이어는 w1, 나머지는 w2
        styles = [w1] * crossover_point + [w2] * (generator.n_layers - crossover_point)
        return styles
    else:
        # 단일 스타일 사용
        z = torch.randn(batch_size, style_dim, device=device)
        w = mapping_net(z)
        return w

def calculate_quality_score(images):
    """의료 이미지 품질 평가"""
    if len(images) == 0:
        return 0.0
    images_np = images.detach().cpu().numpy()
    variance = np.var(images_np)
    return float(variance)

def train_medical_stylegan(dataset_type='chest_xray', data_path=None, num_epochs=100, save_interval=10):
    """
    Medical StyleGAN 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("stylegan", dataset_type)

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        num_epochs = int(os.getenv('TEST_EPOCHS', 5))
        logger.log(f"Quick test mode: training for {num_epochs} epochs only")

    # StyleGAN hyperparameters
    batch_size = 32  # StyleGAN은 메모리를 많이 사용
    image_size = 64  # 단순화된 버전에서는 64x64
    latent_dim = 512
    style_dim = 512
    lr_g = 0.001
    lr_d = 0.001
    beta1 = 0.0
    beta2 = 0.99

    # Save configuration
    config = {
        'algorithm': 'StyleGAN',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'image_size': image_size,
        'latent_dim': latent_dim,
        'style_dim': style_dim,
        'lr_g': lr_g,
        'lr_d': lr_d,
        'beta1': beta1,
        'beta2': beta2,
        'num_epochs': num_epochs,
        'data_path': data_path
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Load medical data
    logger.log(f"Loading {dataset_type} data...")
    if dataset_type == 'chest_xray':
        images = load_chest_xray_data(data_path, num_samples=1000, image_size=image_size)
        input_channels = 1
    elif dataset_type == 'brain_mri':
        images = load_brain_mri_data(data_path, num_samples=1000, image_size=image_size)
        input_channels = 1
    else:
        loader = MedicalImageLoader(dataset_type, image_size)
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
    dataset = MedicalImageDataset(images, image_size, input_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create networks
    mapping_net = StyleGANMappingNetwork(latent_dim, style_dim).to(device)
    generator = StyleGANGenerator(style_dim, n_layers=4, img_channels=input_channels).to(device)
    discriminator = StyleGANDiscriminator(input_channels).to(device)

    logger.log(f"Mapping Network parameters: {sum(p.numel() for p in mapping_net.parameters()):,}")
    logger.log(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    logger.log(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Optimizers
    optimizer_g = optim.Adam(list(mapping_net.parameters()) + list(generator.parameters()),
                            lr=lr_g, betas=(beta1, beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Fixed styles for visualization
    fixed_z = torch.randn(16, latent_dim, device=device)
    fixed_w = mapping_net(fixed_z)

    # Training loop
    logger.log("Starting StyleGAN training...")
    logger.log("Features: Mapping network, Style mixing, AdaIN, Noise injection")

    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        num_batches = 0

        for i, (real_data, _) in enumerate(dataloader):
            real_data = real_data.to(device)
            batch_size = real_data.size(0)

            # Labels for real and fake
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            optimizer_d.zero_grad()

            # Real images
            real_pred = discriminator(real_data)
            d_loss_real = criterion(real_pred, real_labels)

            # Generate fake images with style mixing
            styles = style_mixing(generator, mapping_net, batch_size, latent_dim, device)
            fake_data = generator(styles)

            # Fake images
            fake_pred = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_pred, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()

            fake_pred = discriminator(fake_data)
            g_loss = criterion(fake_pred, real_labels)  # Generator wants to fool discriminator

            g_loss.backward()
            optimizer_g.step()

            # Statistics
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

        # Calculate averages
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches

        # Log metrics
        logger.log_metrics(epoch + 1, avg_g_loss, discriminator_loss=avg_d_loss)

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Saving intermediate results at epoch {epoch + 1}...")

            # Generate samples with fixed styles
            with torch.no_grad():
                # Single style generation
                fake_single = generator(fixed_w)

                # Style mixing generation
                styles_mixed = style_mixing(generator, mapping_net, 16, latent_dim, device)
                fake_mixed = generator(styles_mixed)

                # Convert for saving
                def convert_for_saving(fake_images):
                    fake_images = fake_images.detach().cpu()
                    fake_images = (fake_images + 1) / 2
                    fake_images = torch.clamp(fake_images, 0, 1)

                    if input_channels == 1:
                        fake_images = fake_images.squeeze(1).numpy()
                        fake_images = (fake_images * 255).astype(np.uint8)
                    else:
                        fake_images = fake_images.permute(0, 2, 3, 1).numpy()
                        fake_images = (fake_images * 255).astype(np.uint8)

                    return fake_images

                fake_single_imgs = convert_for_saving(fake_single)
                fake_mixed_imgs = convert_for_saving(fake_mixed)

                # Save single style samples
                logger.save_image_grid(
                    [fake_single_imgs[j] for j in range(len(fake_single_imgs))],
                    f"generated_single_style_epoch_{epoch+1:03d}",
                    titles=[f"Single {j+1}" for j in range(len(fake_single_imgs))],
                    rows=4, cols=4,
                    cmap='gray' if input_channels == 1 else None
                )

                # Save style mixing samples
                logger.save_image_grid(
                    [fake_mixed_imgs[j] for j in range(len(fake_mixed_imgs))],
                    f"generated_style_mixing_epoch_{epoch+1:03d}",
                    titles=[f"Mixed {j+1}" for j in range(len(fake_mixed_imgs))],
                    rows=4, cols=4,
                    cmap='gray' if input_channels == 1 else None
                )

                # Quality metrics
                quality_score = calculate_quality_score(fake_single)
                logger.log(f"Epoch {epoch+1} - Quality Score: {quality_score:.4f}")

        # Save model checkpoints
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(mapping_net, f"stylegan_mapping_epoch_{epoch+1:03d}",
                             optimizer=optimizer_g, epoch=epoch+1, config=config)
            logger.save_model(generator, f"stylegan_generator_epoch_{epoch+1:03d}",
                             optimizer=optimizer_g, epoch=epoch+1, config=config)
            logger.save_model(discriminator, f"stylegan_discriminator_epoch_{epoch+1:03d}",
                             optimizer=optimizer_d, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Saving final results...")

    # Plot training curves
    logger.plot_training_curves()

    # Save final models
    logger.save_model(mapping_net, "stylegan_mapping_final", optimizer=optimizer_g,
                     epoch=num_epochs, config=config)
    logger.save_model(generator, "stylegan_generator_final", optimizer=optimizer_g,
                     epoch=num_epochs, config=config)
    logger.save_model(discriminator, "stylegan_discriminator_final", optimizer=optimizer_d,
                     epoch=num_epochs, config=config)

    # Final comprehensive generation
    with torch.no_grad():
        # Generate various style combinations
        num_samples = 32
        final_z = torch.randn(num_samples, latent_dim, device=device)
        final_w = mapping_net(final_z)
        final_fake = generator(final_w)

        final_images = final_fake.detach().cpu()
        final_images = (final_images + 1) / 2
        final_images = torch.clamp(final_images, 0, 1)

        if input_channels == 1:
            final_images = final_images.squeeze(1).numpy()
            final_images = (final_images * 255).astype(np.uint8)
        else:
            final_images = final_images.permute(0, 2, 3, 1).numpy()
            final_images = (final_images * 255).astype(np.uint8)

        logger.save_image_grid(
            [final_images[j] for j in range(num_samples)],
            "final_generated_samples",
            titles=[f"Final {j+1}" for j in range(num_samples)],
            rows=4, cols=8,
            cmap='gray' if input_channels == 1 else None
        )

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return mapping_net, generator, discriminator, results_dir

if __name__ == "__main__":
    print("Medical StyleGAN Implementation")
    print("===============================")

    # Configuration
    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'

    print(f"Training StyleGAN on {selected_dataset} images...")
    print("StyleGAN Key Features:")
    print("- Mapping network for disentangled style space")
    print("- AdaIN for style control at each layer")
    print("- Style mixing for diverse generation")
    print("- Noise injection for fine details")
    print("- Equalized learning rate for stable training")

    # Train the model
    try:
        mapping_net, generator, discriminator, results_dir = train_medical_stylegan(
            dataset_type=selected_dataset,
            data_path=None,
            num_epochs=50,
            save_interval=5
        )

        print(f"\nTraining completed successfully!")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Generated samples with single style and style mixing")
        print("- models/: Mapping network, generator, and discriminator checkpoints")
        print("- logs/: Training logs with StyleGAN-specific metrics")
        print("- plots/: Training loss curves")
        print("- metrics/: Training metrics in JSON format")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise