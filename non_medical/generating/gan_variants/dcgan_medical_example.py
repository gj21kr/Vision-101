"""
Medical DCGAN (Deep Convolutional Generative Adversarial Networks) Implementation

의료 이미지 생성을 위한 DCGAN 구현으로, 다음과 같은 의료 특화 기능을 제공합니다:
- 의료 이미지 특화 데이터 로더 (chest X-ray, brain MRI, skin lesion)
- 자동 결과 저장 및 로깅 시스템
- 의료 이미지 품질 평가 메트릭
- 훈련 과정 시각화

DCGAN의 핵심 특징:
1. 완전 연결 층을 합성곱 층으로 대체
2. Batch Normalization 적용
3. Generator에는 ReLU, Discriminator에는 LeakyReLU
4. Generator 출력: Tanh, Discriminator 출력: Sigmoid

Reference:
- Radford, A., Metz, L., & Chintala, S. (2015).
  "Unsupervised representation learning with deep convolutional generative adversarial networks."
"""

import torch
import torch.nn as nn
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


# Ensure project root is on sys.path
from medical.medical_data_utils import MedicalImageLoader, load_chest_xray_data, load_brain_mri_data
from non_medical.result_logger import create_logger_for_generating

def weights_init(m):
    """DCGAN 논문에서 권장하는 가중치 초기화"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DCGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(DCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class DCGANDiscriminator(nn.Module):
    def __init__(self, nc=1, ndf=64):
        super(DCGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class MedicalImageDataset(Dataset):
    def __init__(self, images, image_size=64, channels=1):
        self.images = images
        self.image_size = image_size
        self.channels = channels

        # Create transform for medical images
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * channels, (0.5,) * channels)  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Ensure proper format
        if len(image.shape) == 2:  # Grayscale
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and self.channels == 1:
            if image.shape[-1] == 3:
                image = np.mean(image, axis=-1, keepdims=True)

        # Apply transform
        if self.transform:
            image = self.transform(image.astype(np.uint8))

        return image, 0  # Return dummy label

def calculate_quality_score(images):
    """의료 이미지 품질 평가 메트릭"""
    if len(images) == 0:
        return 0.0

    images_np = images.detach().cpu().numpy()
    variance = np.var(images_np)
    return float(variance)

def train_medical_dcgan(dataset_type='chest_xray', data_path=None, num_epochs=100, save_interval=10):
    """
    Medical Image DCGAN 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("dcgan", dataset_type)

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        num_epochs = int(os.getenv('TEST_EPOCHS', 5))
        logger.log(f"Quick test mode: training for {num_epochs} epochs only")

    # Hyperparameters
    batch_size = 64
    image_size = 64
    nz = 100  # Size of latent vector
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    lr = 0.0002
    beta1 = 0.5

    # Save configuration
    config = {
        'algorithm': 'DCGAN',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'image_size': image_size,
        'nz': nz,
        'ngf': ngf,
        'ndf': ndf,
        'lr': lr,
        'beta1': beta1,
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
    else:  # skin_lesion or others
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
    netG = DCGANGenerator(nz, ngf, input_channels).to(device)
    netG.apply(weights_init)

    netD = DCGANDiscriminator(input_channels, ndf).to(device)
    netD.apply(weights_init)

    logger.log(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    logger.log(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

    # Loss function and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Labels
    real_label = 1.
    fake_label = 0.

    # Training loop
    logger.log("Starting DCGAN training...")

    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_d_x = 0
        epoch_d_g_z = 0
        num_batches = 0

        for i, (data, _) in enumerate(dataloader):
            # Update Discriminator
            netD.zero_grad()

            # Train with real batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake batch
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator
            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Accumulate metrics
            epoch_d_loss += errD.item()
            epoch_g_loss += errG.item()
            epoch_d_x += D_x
            epoch_d_g_z += D_G_z1
            num_batches += 1

        # Calculate average metrics
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_x = epoch_d_x / num_batches
        avg_d_g_z = epoch_d_g_z / num_batches

        # Log metrics
        logger.log_metrics(epoch + 1, avg_g_loss,
                          discriminator_loss=avg_d_loss,
                          D_x=avg_d_x,
                          D_G_z=avg_d_g_z)

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Saving intermediate results at epoch {epoch + 1}...")

            # Generate samples
            with torch.no_grad():
                fake = netG(fixed_noise)

                # Convert to proper format for saving
                fake_images = fake.detach().cpu()
                fake_images = (fake_images + 1) / 2  # Denormalize from [-1,1] to [0,1]
                fake_images = torch.clamp(fake_images, 0, 1)

                if input_channels == 1:
                    fake_images = fake_images.squeeze(1).numpy()
                    fake_images = (fake_images * 255).astype(np.uint8)
                else:
                    fake_images = fake_images.permute(0, 2, 3, 1).numpy()
                    fake_images = (fake_images * 255).astype(np.uint8)

                # Calculate quality metrics
                quality_score = calculate_quality_score(fake)
                pixel_mean = torch.mean(fake).item()
                pixel_std = torch.std(fake).item()

                logger.log(f"Epoch {epoch+1} - Quality Score: {quality_score:.4f}")
                logger.log(f"Epoch {epoch+1} - Pixel Mean: {pixel_mean:.4f}, Std: {pixel_std:.4f}")

                # Save generated samples
                logger.save_image_grid(
                    [fake_images[i] for i in range(min(64, len(fake_images)))],
                    f"generated_samples_epoch_{epoch+1:03d}",
                    titles=[f"Gen {i+1}" for i in range(min(64, len(fake_images)))],
                    rows=8, cols=8,
                    cmap='gray' if input_channels == 1 else None
                )

        # Save model checkpoint
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(netG, f"dcgan_generator_epoch_{epoch+1:03d}",
                             optimizer=optimizerG, epoch=epoch+1, config=config)
            logger.save_model(netD, f"dcgan_discriminator_epoch_{epoch+1:03d}",
                             optimizer=optimizerD, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Saving final results...")

    # Plot training curves
    logger.plot_training_curves()

    # Save final models
    logger.save_model(netG, "dcgan_generator_final", optimizer=optimizerG,
                     epoch=num_epochs, config=config)
    logger.save_model(netD, "dcgan_discriminator_final", optimizer=optimizerD,
                     epoch=num_epochs, config=config)

    # Final sample generation
    with torch.no_grad():
        final_fake = netG(fixed_noise)

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
            [final_images[i] for i in range(64)],
            "final_generated_samples",
            titles=[f"Final {i+1}" for i in range(64)],
            rows=8, cols=8,
            cmap='gray' if input_channels == 1 else None
        )

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return netG, netD, results_dir

if __name__ == "__main__":
    print("Medical DCGAN Implementation")
    print("============================")

    # Configuration
    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'  # Change this to test different datasets

    print(f"Training DCGAN on {selected_dataset} images...")
    print("Features:")
    print("- Deep convolutional architecture optimized for medical images")
    print("- Automatic result saving and logging")
    print("- Medical image quality metrics")
    print("- Support for grayscale and RGB medical images")

    # Train the model
    try:
        netG, netD, results_dir = train_medical_dcgan(
            dataset_type=selected_dataset,
            data_path=None,  # Use synthetic data - set to real path if available
            num_epochs=50,
            save_interval=5
        )

        print(f"\nTraining completed successfully!")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Generated medical image samples")
        print("- models/: Generator and discriminator checkpoints")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training loss curves")
        print("- metrics/: Training metrics in JSON format")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
