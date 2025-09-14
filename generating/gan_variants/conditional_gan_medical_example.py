"""
Medical Conditional GAN Implementation

의료 이미지 생성을 위한 조건부 GAN(Conditional GAN) 구현입니다.
조건부 GAN은 특정 클래스나 조건에 따라 이미지를 생성할 수 있습니다.

의료 분야에서의 활용:
- 특정 질병 유형에 따른 의료 이미지 생성
- 정상/비정상 조건부 이미지 생성
- 다양한 의료 이미지 카테고리별 생성

의료 특화 기능:
- 의료 이미지 특화 데이터 로더
- 자동 결과 저장 및 로깅 시스템
- 조건별 생성 결과 시각화
- 의료 이미지 품질 평가 메트릭

핵심 특징:
1. Generator와 Discriminator 모두 클래스 정보를 입력으로 받음
2. Label embedding을 통한 조건 정보 인코딩
3. 원하는 클래스의 이미지를 의도적으로 생성 가능

Reference:
- Mirza, M., & Osindero, S. (2014).
  "Conditional generative adversarial nets."
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from medical_data_utils import MedicalImageLoader, load_chest_xray_data, load_brain_mri_data
from result_logger import create_logger_for_generating

class ConditionalGenerator(nn.Module):
    """조건부 Generator: 노이즈 + 클래스 조건으로 이미지 생성"""
    def __init__(self, noise_dim=100, num_classes=2, embed_dim=100, img_size=64, img_channels=1):
        super(ConditionalGenerator, self).__init__()
        self.img_size = img_size
        self.img_channels = img_channels

        # Label embedding: 클래스를 벡터로 변환
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # 입력 차원: 노이즈 + 임베딩된 라벨
        input_dim = noise_dim + embed_dim

        # 완전연결층들
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),

            # 출력 차원: 이미지 크기에 맞춤
            nn.Linear(1024, img_channels * img_size * img_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        """
        조건부 이미지 생성

        Args:
            noise: 랜덤 노이즈 벡터 [batch_size, noise_dim]
            labels: 클래스 라벨 [batch_size]

        Returns:
            생성된 의료 이미지 [batch_size, channels, height, width]
        """
        # 라벨을 임베딩 벡터로 변환
        embedded_labels = self.label_embedding(labels)

        # 노이즈와 임베딩된 라벨 결합
        gen_input = torch.cat((noise, embedded_labels), dim=1)

        # 이미지 생성
        img = self.fc_layers(gen_input)
        img = img.view(img.size(0), self.img_channels, self.img_size, self.img_size)
        return img

class ConditionalDiscriminator(nn.Module):
    """조건부 Discriminator: 이미지 + 클래스 조건으로 진짜/가짜 판별"""
    def __init__(self, num_classes=2, embed_dim=100, img_size=64, img_channels=1):
        super(ConditionalDiscriminator, self).__init__()
        self.img_size = img_size
        self.img_channels = img_channels

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # 이미지 차원
        self.img_dim = img_channels * img_size * img_size

        # 판별기 네트워크: 이미지 + 임베딩된 라벨
        self.disc_layers = nn.Sequential(
            nn.Linear(self.img_dim + embed_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()  # 확률 출력
        )

    def forward(self, img, labels):
        """
        조건부 이미지 판별

        Args:
            img: 입력 이미지 [batch_size, channels, height, width]
            labels: 클래스 라벨 [batch_size]

        Returns:
            진짜/가짜 확률 [batch_size, 1]
        """
        # 이미지를 1차원으로 평면화
        img_flat = img.view(img.size(0), -1)

        # 라벨 임베딩
        embedded_labels = self.label_embedding(labels)

        # 이미지와 임베딩된 라벨 결합
        disc_input = torch.cat((img_flat, embedded_labels), dim=1)

        # 진짜/가짜 확률 계산
        validity = self.disc_layers(disc_input)
        return validity.squeeze()

class MedicalConditionalDataset(Dataset):
    """의료 이미지를 위한 조건부 데이터셋"""
    def __init__(self, images, labels, image_size=64, channels=1):
        self.images = images
        self.labels = labels
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
        label = self.labels[idx]

        # 이미지 형태 조정
        if len(image.shape) == 2:  # Grayscale
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and self.channels == 1:
            if image.shape[-1] == 3:
                image = np.mean(image, axis=-1, keepdims=True)

        if self.transform:
            image = self.transform(image.astype(np.uint8))

        return image, torch.tensor(label, dtype=torch.long)

def create_medical_labels(dataset_type, num_samples):
    """의료 이미지에 대한 합성 라벨 생성"""
    if dataset_type == 'chest_xray':
        # 0: 정상, 1: 폐렴
        labels = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
        class_names = ['Normal', 'Pneumonia']
    elif dataset_type == 'brain_mri':
        # 0: 정상, 1: 종양
        labels = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
        class_names = ['Normal', 'Tumor']
    elif dataset_type == 'skin_lesion':
        # 0: 양성, 1: 악성
        labels = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
        class_names = ['Benign', 'Malignant']
    else:
        # 기본적으로 2클래스
        labels = np.random.choice([0, 1], size=num_samples)
        class_names = ['Class_0', 'Class_1']

    return labels, class_names

def calculate_quality_score(images):
    """의료 이미지 품질 평가"""
    if len(images) == 0:
        return 0.0
    images_np = images.detach().cpu().numpy()
    variance = np.var(images_np)
    return float(variance)

def train_medical_conditional_gan(dataset_type='chest_xray', data_path=None, num_epochs=100, save_interval=10):
    """
    Medical Conditional GAN 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("conditional_gan", dataset_type)

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        num_epochs = int(os.getenv('TEST_EPOCHS', 5))
        logger.log(f"Quick test mode: training for {num_epochs} epochs only")

    # Hyperparameters
    batch_size = 64
    image_size = 64
    noise_dim = 100
    num_classes = 2  # 대부분의 의료 이미지는 2클래스 (정상/비정상)
    embed_dim = 100
    lr = 0.0002
    beta1 = 0.5

    # Save configuration
    config = {
        'algorithm': 'Conditional_GAN',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'image_size': image_size,
        'noise_dim': noise_dim,
        'num_classes': num_classes,
        'embed_dim': embed_dim,
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
    else:
        loader = MedicalImageLoader(dataset_type, image_size)
        if data_path and os.path.exists(data_path):
            images = loader.load_real_dataset(data_path, 1000)
        else:
            images = loader.create_synthetic_medical_data(1000)
        input_channels = 3 if len(images[0].shape) == 3 else 1

    # Create synthetic labels for medical conditions
    labels, class_names = create_medical_labels(dataset_type, len(images))

    logger.log(f"Loaded {len(images)} {dataset_type} images")
    logger.log(f"Classes: {class_names}")
    logger.log(f"Label distribution: {np.bincount(labels)}")

    # Save sample images by class
    for class_idx, class_name in enumerate(class_names):
        class_indices = np.where(labels == class_idx)[0][:9]  # 최대 9개
        if len(class_indices) > 0:
            class_images = [images[i] for i in class_indices]
            logger.save_image_grid(class_images, f"original_samples_class_{class_idx}_{class_name}",
                                  titles=[f"{class_name} {i+1}" for i in range(len(class_images))],
                                  cmap='gray' if input_channels == 1 else None)

    # Create dataset and dataloader
    dataset = MedicalConditionalDataset(images, labels, image_size, input_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create networks
    netG = ConditionalGenerator(noise_dim, num_classes, embed_dim, image_size, input_channels).to(device)
    netD = ConditionalDiscriminator(num_classes, embed_dim, image_size, input_channels).to(device)

    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    netG.apply(weights_init)
    netD.apply(weights_init)

    logger.log(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    logger.log(f"Discriminator parameters: {sum(p.numel() for p in netD.parameters()):,}")

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Fixed noise and labels for visualization
    fixed_noise = torch.randn(32, noise_dim, device=device)
    # 각 클래스별로 절반씩 고정 라벨 생성
    fixed_labels = torch.cat([torch.full((16,), i, dtype=torch.long, device=device)
                             for i in range(num_classes)])

    # Labels for training
    real_label = 1.
    fake_label = 0.

    # Training loop
    logger.log("Starting Conditional GAN training...")
    logger.log(f"Medical classes: {class_names}")

    for epoch in range(num_epochs):
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_d_x = 0
        epoch_d_g_z = 0
        num_batches = 0

        for i, (data, labels_batch) in enumerate(dataloader):
            # Move to device
            real_data = data.to(device)
            real_labels = labels_batch.to(device)
            batch_size = real_data.size(0)

            # Create label tensors
            label_real = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            label_fake = torch.full((batch_size,), fake_label, dtype=torch.float, device=device)

            # Update Discriminator: maximize log(D(x,y)) + log(1 - D(G(z,y),y))
            netD.zero_grad()

            # Train with real images
            output_real = netD(real_data, real_labels)
            errD_real = criterion(output_real, label_real)
            errD_real.backward()
            D_x = output_real.mean().item()

            # Train with fake images
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)  # 랜덤 클래스
            fake_data = netG(noise, fake_labels)

            output_fake = netD(fake_data.detach(), fake_labels)
            errD_fake = criterion(output_fake, label_fake)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator: maximize log(D(G(z,y),y))
            netG.zero_grad()
            output = netD(fake_data, fake_labels)
            errG = criterion(output, label_real)  # Generator wants to fool discriminator
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Statistics
            epoch_d_loss += errD.item()
            epoch_g_loss += errG.item()
            epoch_d_x += D_x
            epoch_d_g_z += D_G_z1
            num_batches += 1

        # Calculate averages
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

            # Generate samples with fixed noise and labels
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_labels)

                # Convert for saving
                fake_images = fake.detach().cpu()
                fake_images = (fake_images + 1) / 2
                fake_images = torch.clamp(fake_images, 0, 1)

                if input_channels == 1:
                    fake_images = fake_images.squeeze(1).numpy()
                    fake_images = (fake_images * 255).astype(np.uint8)
                else:
                    fake_images = fake_images.permute(0, 2, 3, 1).numpy()
                    fake_images = (fake_images * 255).astype(np.uint8)

                # Create titles with class information
                titles = []
                for j, label_idx in enumerate(fixed_labels.cpu().numpy()):
                    titles.append(f"{class_names[label_idx]} {j+1}")

                # Save generated samples
                logger.save_image_grid(
                    [fake_images[j] for j in range(len(fake_images))],
                    f"generated_samples_epoch_{epoch+1:03d}",
                    titles=titles,
                    rows=4, cols=8,
                    cmap='gray' if input_channels == 1 else None
                )

                # 클래스별 분리된 이미지도 저장
                for class_idx in range(num_classes):
                    class_mask = fixed_labels.cpu() == class_idx
                    if class_mask.sum() > 0:
                        class_fake_images = [fake_images[j] for j in range(len(fake_images)) if class_mask[j]]
                        logger.save_image_grid(
                            class_fake_images,
                            f"generated_class_{class_idx}_{class_names[class_idx]}_epoch_{epoch+1:03d}",
                            titles=[f"{class_names[class_idx]} {j+1}" for j in range(len(class_fake_images))],
                            cmap='gray' if input_channels == 1 else None
                        )

                # Calculate quality metrics
                quality_score = calculate_quality_score(fake)
                logger.log(f"Epoch {epoch+1} - Quality Score: {quality_score:.4f}")

        # Save model checkpoints
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(netG, f"conditional_gan_generator_epoch_{epoch+1:03d}",
                             optimizer=optimizerG, epoch=epoch+1, config=config)
            logger.save_model(netD, f"conditional_gan_discriminator_epoch_{epoch+1:03d}",
                             optimizer=optimizerD, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Saving final results...")

    # Plot training curves
    logger.plot_training_curves()

    # Save final models
    logger.save_model(netG, "conditional_gan_generator_final", optimizer=optimizerG,
                     epoch=num_epochs, config=config)
    logger.save_model(netD, "conditional_gan_discriminator_final", optimizer=optimizerD,
                     epoch=num_epochs, config=config)

    # Generate final samples for each class
    with torch.no_grad():
        # Generate more samples for each class
        final_samples_per_class = 32
        for class_idx, class_name in enumerate(class_names):
            class_noise = torch.randn(final_samples_per_class, noise_dim, device=device)
            class_labels = torch.full((final_samples_per_class,), class_idx, dtype=torch.long, device=device)

            final_fake = netG(class_noise, class_labels)

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
                [final_images[j] for j in range(final_samples_per_class)],
                f"final_generated_class_{class_idx}_{class_name}",
                titles=[f"{class_name} {j+1}" for j in range(final_samples_per_class)],
                rows=4, cols=8,
                cmap='gray' if input_channels == 1 else None
            )

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return netG, netD, results_dir, class_names

if __name__ == "__main__":
    print("Medical Conditional GAN Implementation")
    print("======================================")

    # Configuration
    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'

    print(f"Training Conditional GAN on {selected_dataset} images...")
    print("Key Features:")
    print("- Class-conditional medical image generation")
    print("- Separate generation for normal/abnormal conditions")
    print("- Label embedding for condition encoding")
    print("- Automatic result logging by medical condition")

    # Train the model
    try:
        netG, netD, results_dir, class_names = train_medical_conditional_gan(
            dataset_type=selected_dataset,
            data_path=None,
            num_epochs=50,
            save_interval=5
        )

        print(f"\nTraining completed successfully!")
        print(f"Medical classes: {class_names}")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Generated samples by medical condition")
        print("- models/: Generator and discriminator checkpoints")
        print("- logs/: Training logs with class-specific metrics")
        print("- plots/: Training loss curves")
        print("- metrics/: Training metrics in JSON format")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise