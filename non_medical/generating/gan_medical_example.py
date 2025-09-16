"""
Medical Image GAN with Result Logging

의료 이미지 생성에 특화된 GAN 구현으로, 다양한 의료 이미지 타입을
생성하고 훈련 과정을 상세히 기록합니다.

특징:
- 의료 이미지별 맞춤 전처리
- Discriminator와 Generator 손실 추적
- 생성 품질 메트릭 계산
- 실시간 생성 결과 저장
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Ensure project root is on sys.path
from medical.medical_data_utils import MedicalImageLoader, load_chest_xray_data, load_brain_mri_data
from non_medical.result_logger import create_logger_for_generating

class MedicalGenerator(nn.Module):
    def __init__(self, noise_dim=100, img_channels=1, img_size=64):
        super(MedicalGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # Calculate dimensions
        self.init_size = img_size // 4  # Initial size before upsampling

        # Linear layer to expand noise
        self.l1 = nn.Linear(noise_dim, 128 * self.init_size ** 2)

        # Convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()  # Output values in [-1, 1]
        )

    def forward(self, z):
        """
        Generator forward pass

        Args:
            z: Noise vector [batch_size, noise_dim]

        Returns:
            Generated images [batch_size, channels, height, width]
        """
        # Expand noise to feature map
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)

        # Apply convolutional blocks
        img = self.conv_blocks(out)

        return img

class MedicalDiscriminator(nn.Module):
    def __init__(self, img_channels=1, img_size=64):
        super(MedicalDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Discriminator block with conv, batch norm, and leaky relu"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Calculate size after convolutions
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        """
        Discriminator forward pass

        Args:
            img: Input images [batch_size, channels, height, width]

        Returns:
            Probability of being real [batch_size, 1]
        """
        features = self.model(img)
        features = features.view(features.shape[0], -1)
        validity = self.adv_layer(features)

        return validity

class MedicalImageDataset(Dataset):
    def __init__(self, images, image_size=64, channels=1):
        self.images = images
        self.image_size = image_size
        self.channels = channels

        # Create transform
        if channels == 1:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Ensure proper format
        if len(image.shape) == 2 and self.channels == 1:
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and self.channels == 1:
            if image.shape[-1] == 3:
                image = np.mean(image, axis=-1, keepdims=True)

        # Apply transform
        if self.transform:
            image = self.transform(image.astype(np.uint8))

        return image

def calculate_inception_score(images, batch_size=32):
    """
    Calculate Inception Score (simplified version)

    실제 구현에서는 pre-trained Inception-v3 모델을 사용하지만,
    의존성을 줄이기 위해 간단한 품질 메트릭을 계산합니다.

    Args:
        images: 생성된 이미지 리스트
        batch_size: 배치 크기

    Returns:
        float: 품질 점수 (높을수록 좋음)
    """
    if not images or len(images) == 0:
        return 1.0

    # 이미지 다양성 측정 (pixel variance 기반)
    images_array = np.array(images)
    if len(images_array.shape) == 4:  # [N, H, W, C]
        pixel_variance = np.var(images_array.reshape(len(images), -1), axis=1)
        diversity_score = np.mean(pixel_variance)
    else:  # [N, H, W]
        pixel_variance = np.var(images_array.reshape(len(images), -1), axis=1)
        diversity_score = np.mean(pixel_variance)

    # 정규화하여 IS와 유사한 범위로 변환 (1-5)
    normalized_score = min(5.0, max(1.0, 1.0 + diversity_score * 100))

    return normalized_score

def train_medical_gan(dataset_type='chest_xray', data_path=None, num_epochs=100, save_interval=10):
    """
    Medical Image GAN 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("gan", dataset_type)

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.0002
    beta1 = 0.5  # Adam optimizer beta1
    image_size = 64
    noise_dim = 100

    # Save configuration
    config = {
        'algorithm': 'GAN',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'beta1': beta1,
        'image_size': image_size,
        'noise_dim': noise_dim,
        'num_epochs': num_epochs,
        'data_path': data_path
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Load medical data
    logger.log(f"Loading {dataset_type} data...")
    if dataset_type == 'chest_xray':
        images = load_chest_xray_data(data_path, num_samples=2000, image_size=image_size)
        input_channels = 1
    elif dataset_type == 'brain_mri':
        images = load_brain_mri_data(data_path, num_samples=2000, image_size=image_size)
        input_channels = 1
    else:  # skin_lesion or others
        loader = MedicalImageLoader(dataset_type, image_size)
        if data_path and os.path.exists(data_path):
            images = loader.load_real_dataset(data_path, 2000)
        else:
            images = loader.create_synthetic_medical_data(2000)
        input_channels = 3 if len(images[0].shape) == 3 else 1

    logger.log(f"Loaded {len(images)} {dataset_type} images")
    logger.log(f"Image shape: {images[0].shape}")

    # Save sample original images
    sample_images = [images[i] for i in range(min(16, len(images)))]
    logger.save_image_grid(sample_images, "original_samples",
                          titles=[f"Real {i+1}" for i in range(len(sample_images))],
                          rows=4, cols=4,
                          cmap='gray' if input_channels == 1 else None)

    # Create dataset and dataloader
    dataset = MedicalImageDataset(images, image_size, input_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Generator and Discriminator
    generator = MedicalGenerator(noise_dim=noise_dim, img_channels=input_channels, img_size=image_size).to(device)
    discriminator = MedicalDiscriminator(img_channels=input_channels, img_size=image_size).to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999))

    logger.log(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    logger.log(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Fixed noise for consistent generation tracking
    fixed_noise = torch.randn(16, noise_dim, device=device)

    # Training loop
    logger.log("Starting Medical GAN training...")

    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_d_acc_real = 0
        epoch_d_acc_fake = 0
        num_batches = 0

        for i, real_images in enumerate(dataloader):
            batch_size_current = real_images.size(0)
            real_images = real_images.to(device)

            # Create labels
            real_labels = torch.ones(batch_size_current, 1, device=device)
            fake_labels = torch.zeros(batch_size_current, 1, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Real images
            real_validity = discriminator(real_images)
            d_real_loss = criterion(real_validity, real_labels)

            # Fake images
            noise = torch.randn(batch_size_current, noise_dim, device=device)
            fake_images = generator(noise)
            fake_validity = discriminator(fake_images.detach())
            d_fake_loss = criterion(fake_validity, fake_labels)

            # Combined discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake images and get discriminator's opinion
            fake_validity = discriminator(fake_images)
            g_loss = criterion(fake_validity, real_labels)  # Generator wants D to think fakes are real

            g_loss.backward()
            optimizer_G.step()

            # Calculate accuracies
            d_acc_real = (real_validity > 0.5).float().mean().item()
            d_acc_fake = (fake_validity <= 0.5).float().mean().item()

            # Accumulate losses
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_acc_real += d_acc_real
            epoch_d_acc_fake += d_acc_fake
            num_batches += 1

            # Log batch progress
            if i % 50 == 0:
                logger.log(f"Epoch {epoch+1}/{num_epochs}, Batch {i}/{len(dataloader)}, "
                          f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

        # Calculate epoch averages
        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        avg_d_acc_real = epoch_d_acc_real / num_batches
        avg_d_acc_fake = epoch_d_acc_fake / num_batches

        # Log epoch metrics
        logger.log_metrics(
            epoch + 1,
            avg_g_loss,  # Use G_loss as primary metric
            val_loss=avg_d_loss,
            D_loss=avg_d_loss,
            G_loss=avg_g_loss,
            D_acc_real=avg_d_acc_real,
            D_acc_fake=avg_d_acc_fake,
            D_acc_total=(avg_d_acc_real + avg_d_acc_fake) / 2
        )

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Saving results at epoch {epoch + 1}...")

            generator.eval()
            with torch.no_grad():
                # Generate samples from fixed noise
                fake_samples = generator(fixed_noise)

                # Convert to numpy for visualization
                fake_samples_np = fake_samples.cpu()
                fake_samples_np = (fake_samples_np + 1) / 2  # Denormalize from [-1,1] to [0,1]
                fake_samples_np = fake_samples_np.clamp(0, 1)

                if input_channels == 1:
                    fake_images_list = [fake_samples_np[i, 0].numpy() for i in range(16)]
                else:
                    fake_images_list = [fake_samples_np[i].permute(1, 2, 0).numpy() for i in range(16)]

                # Save generated samples
                logger.save_image_grid(
                    fake_images_list,
                    f"generated_samples_epoch_{epoch+1:03d}",
                    titles=[f"Gen {i+1}" for i in range(16)],
                    rows=4, cols=4,
                    cmap='gray' if input_channels == 1 else None
                )

                # Generate more samples for quality assessment
                large_noise = torch.randn(64, noise_dim, device=device)
                large_fake_batch = generator(large_noise)
                large_fake_np = large_fake_batch.cpu()
                large_fake_np = (large_fake_np + 1) / 2
                large_fake_np = large_fake_np.clamp(0, 1)

                # Calculate quality metrics
                if input_channels == 1:
                    quality_images = [large_fake_np[i, 0].numpy() for i in range(64)]
                else:
                    quality_images = [large_fake_np[i].permute(1, 2, 0).numpy() for i in range(64)]

                quality_score = calculate_inception_score(quality_images)

                # Calculate additional metrics
                pixel_mean = np.mean([np.mean(img) for img in quality_images])
                pixel_std = np.mean([np.std(img) for img in quality_images])

                # Save quality metrics
                with open(os.path.join(logger.dirs['metrics'], f'quality_epoch_{epoch+1:03d}.txt'), 'w') as f:
                    f.write(f"Epoch: {epoch+1}\n")
                    f.write(f"Quality Score: {quality_score:.4f}\n")
                    f.write(f"Pixel Mean: {pixel_mean:.4f}\n")
                    f.write(f"Pixel Std: {pixel_std:.4f}\n")
                    f.write(f"Generator Loss: {avg_g_loss:.4f}\n")
                    f.write(f"Discriminator Loss: {avg_d_loss:.4f}\n")
                    f.write(f"Discriminator Accuracy: {(avg_d_acc_real + avg_d_acc_fake) / 2:.4f}\n")

                # Create loss plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))

                # Loss curves
                if len(logger.metrics['epochs']) > 1:
                    epochs = logger.metrics['epochs']
                    g_losses = [logger.metrics['custom_metrics']['G_loss'][i] for i in range(len(epochs))]
                    d_losses = [logger.metrics['custom_metrics']['D_loss'][i] for i in range(len(epochs))]

                    axes[0, 0].plot(epochs, g_losses, 'b-', label='Generator', linewidth=2)
                    axes[0, 0].plot(epochs, d_losses, 'r-', label='Discriminator', linewidth=2)
                    axes[0, 0].set_title('Training Losses')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True)

                    # Discriminator accuracy
                    d_acc_real = [logger.metrics['custom_metrics']['D_acc_real'][i] for i in range(len(epochs))]
                    d_acc_fake = [logger.metrics['custom_metrics']['D_acc_fake'][i] for i in range(len(epochs))]

                    axes[0, 1].plot(epochs, d_acc_real, 'g-', label='Real Accuracy', linewidth=2)
                    axes[0, 1].plot(epochs, d_acc_fake, 'orange', label='Fake Accuracy', linewidth=2)
                    axes[0, 1].set_title('Discriminator Accuracy')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Accuracy')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)

                # Show sample generated image
                axes[1, 0].imshow(fake_images_list[0], cmap='gray' if input_channels == 1 else None)
                axes[1, 0].set_title(f'Sample Generated Image - Epoch {epoch+1}')
                axes[1, 0].axis('off')

                # Show sample real image for comparison
                sample_real = (real_images[0].cpu() + 1) / 2
                sample_real = sample_real.clamp(0, 1)
                if input_channels == 1:
                    axes[1, 1].imshow(sample_real[0].numpy(), cmap='gray')
                else:
                    axes[1, 1].imshow(sample_real.permute(1, 2, 0).numpy())
                axes[1, 1].set_title('Sample Real Image')
                axes[1, 1].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(logger.dirs['plots'], f'training_summary_epoch_{epoch+1:03d}.png'),
                           dpi=150, bbox_inches='tight')
                plt.close()

            generator.train()

        # Save model checkpoints
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(generator, f"generator_checkpoint_epoch_{epoch+1:03d}",
                             optimizer=optimizer_G, epoch=epoch+1, config=config)
            logger.save_model(discriminator, f"discriminator_checkpoint_epoch_{epoch+1:03d}",
                             optimizer=optimizer_D, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Generating final results...")

    generator.eval()
    with torch.no_grad():
        # Generate large batch for final evaluation
        final_noise = torch.randn(100, noise_dim, device=device)
        final_samples = generator(final_noise)
        final_samples_np = (final_samples.cpu() + 1) / 2
        final_samples_np = final_samples_np.clamp(0, 1)

        if input_channels == 1:
            final_images = [final_samples_np[i, 0].numpy() for i in range(100)]
        else:
            final_images = [final_samples_np[i].permute(1, 2, 0).numpy() for i in range(100)]

        # Save final generated samples (show first 64)
        logger.save_image_grid(
            final_images[:64],
            "final_generated_samples",
            titles=[f"Final {i+1}" for i in range(64)],
            rows=8, cols=8,
            cmap='gray' if input_channels == 1 else None
        )

        # Calculate final metrics
        final_quality_score = calculate_inception_score(final_images)
        final_pixel_mean = np.mean([np.mean(img) for img in final_images])
        final_pixel_std = np.mean([np.std(img) for img in final_images])

        with open(os.path.join(logger.dirs['base'], 'final_metrics.txt'), 'w') as f:
            f.write(f"Final Training Results - {dataset_type} GAN\n")
            f.write("="*50 + "\n")
            f.write(f"Total Epochs: {num_epochs}\n")
            f.write(f"Final Generator Loss: {avg_g_loss:.4f}\n")
            f.write(f"Final Discriminator Loss: {avg_d_loss:.4f}\n")
            f.write(f"Final Discriminator Accuracy: {(avg_d_acc_real + avg_d_acc_fake) / 2:.4f}\n")
            f.write(f"Final Quality Score: {final_quality_score:.4f}\n")
            f.write(f"Final Pixel Mean: {final_pixel_mean:.4f}\n")
            f.write(f"Final Pixel Std: {final_pixel_std:.4f}\n")
            f.write(f"Total Generated Samples: {len(final_images)}\n")

    # Plot final training curves
    logger.plot_training_curves()

    # Save final models
    logger.save_model(generator, "generator_final_model", optimizer=optimizer_G,
                     epoch=num_epochs, config=config)
    logger.save_model(discriminator, "discriminator_final_model", optimizer=optimizer_D,
                     epoch=num_epochs, config=config)

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return generator, discriminator, results_dir

if __name__ == "__main__":
    print("Medical Image GAN with Result Logging")
    print("=====================================")

    # Configuration
    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'  # Change this to test different datasets

    print(f"Training GAN on {selected_dataset} images...")
    print("Results will be automatically saved including:")
    print("- Training progress logs")
    print("- Generated sample images at each interval")
    print("- Discriminator and Generator loss curves")
    print("- Model checkpoints")
    print("- Quality metrics (Image diversity and statistics)")

    try:
        generator, discriminator, results_dir = train_medical_gan(
            dataset_type=selected_dataset,
            data_path=None,  # Use synthetic data - set to real path if available
            num_epochs=100,
            save_interval=10
        )

        print(f"\nTraining completed successfully!")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Generated samples and training comparisons")
        print("- models/: Generator and Discriminator checkpoints")
        print("- logs/: Training logs and configuration")
        print("- plots/: Loss curves and training summaries")
        print("- metrics/: Quality metrics and training statistics")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
