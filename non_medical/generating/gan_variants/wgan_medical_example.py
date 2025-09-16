"""
Medical WGAN (Wasserstein GAN) Implementation

의료 이미지 생성을 위한 WGAN 구현으로, Wasserstein distance를 사용하여
전통적인 GAN의 훈련 불안정성 문제를 해결합니다.

의료 특화 기능:
- 의료 이미지 특화 데이터 로더 (chest X-ray, brain MRI, skin lesion)
- 자동 결과 저장 및 로깅 시스템
- Wasserstein distance 추적 및 시각화
- 의료 이미지 품질 평가 메트릭

WGAN의 핵심 개선사항:
1. Wasserstein Distance 사용 (Earth Mover's distance)
2. Critic Network (확률이 아닌 실수값 스코어 출력)
3. 1-Lipschitz 제약 조건 (Weight clipping)
4. RMSprop optimizer 사용
5. 더 안정적인 훈련과 의미있는 손실 함수 값

Reference:
- Arjovsky, M., Chintala, S., & Bottou, L. (2017).
  "Wasserstein generative adversarial networks."
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

import matplotlib.pyplot as plt

# Ensure project root is on sys.path
from medical.medical_data_utils import MedicalImageLoader, load_chest_xray_data, load_brain_mri_data
from non_medical.result_logger import create_logger_for_generating

class WGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(WGANGenerator, self).__init__()
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

class WGANCritic(nn.Module):
    """
    WGAN Critic: 확률 대신 실수 점수 출력

    핵심 특징:
    - Sigmoid 활성화 함수 없음 (핵심 차이점)
    - 1-Lipschitz 제약 조건을 위한 weight clipping
    - Wasserstein distance 근사를 위한 설계
    """
    def __init__(self, nc=1, ndf=64):
        super(WGANCritic, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # No sigmoid! 이것이 DCGAN과의 핵심 차이점
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class MedicalImageDataset(Dataset):
    def __init__(self, images, image_size=64, channels=1):
        self.images = images
        self.image_size = image_size
        self.channels = channels

        # WGAN을 위한 정규화 [-1, 1]
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

        # Ensure proper format
        if len(image.shape) == 2:  # Grayscale
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and self.channels == 1:
            if image.shape[-1] == 3:
                image = np.mean(image, axis=-1, keepdims=True)

        if self.transform:
            image = self.transform(image.astype(np.uint8))

        return image, 0

def weights_init(m):
    """가중치 초기화"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def clip_weights(net, clip_value=0.01):
    """
    Weight Clipping: 1-Lipschitz 제약 조건 강제

    WGAN의 핵심 아이디어:
    - Critic이 1-Lipschitz 함수여야 Wasserstein distance 근사 가능
    - Weight clipping으로 이를 근사적으로 달성

    Args:
        net: Critic 네트워크
        clip_value: 클리핑 범위 [-c, c]
    """
    for param in net.parameters():
        param.data.clamp_(-clip_value, clip_value)

def calculate_wasserstein_distance(critic, real_data, fake_data):
    """
    Wasserstein Distance 근사 계산

    수식: W(P_r, P_g) ≈ E[C(x_real)] - E[C(x_fake)]
    여기서 C는 1-Lipschitz Critic
    """
    with torch.no_grad():
        real_score = torch.mean(critic(real_data))
        fake_score = torch.mean(critic(fake_data))
        return real_score - fake_score

def calculate_quality_score(images):
    """의료 이미지 품질 평가"""
    if len(images) == 0:
        return 0.0
    images_np = images.detach().cpu().numpy()
    variance = np.var(images_np)
    return float(variance)

def train_medical_wgan(dataset_type='chest_xray', data_path=None, num_epochs=100, save_interval=10):
    """
    Medical Image WGAN 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("wgan", dataset_type)

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        num_epochs = int(os.getenv('TEST_EPOCHS', 5))
        logger.log(f"Quick test mode: training for {num_epochs} epochs only")

    # WGAN Hyperparameters
    batch_size = 64
    image_size = 64
    nz = 100
    ngf = 64
    ndf = 64
    lr = 0.00005  # WGAN에서는 낮은 학습률 사용
    n_critic = 5  # Generator 1번 업데이트당 Critic 5번 업데이트
    clip_value = 0.01  # Weight clipping 값

    # Save configuration
    config = {
        'algorithm': 'WGAN',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'image_size': image_size,
        'nz': nz,
        'ngf': ngf,
        'ndf': ndf,
        'lr': lr,
        'n_critic': n_critic,
        'clip_value': clip_value,
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
    netG = WGANGenerator(nz, ngf, input_channels).to(device)
    netG.apply(weights_init)

    netC = WGANCritic(input_channels, ndf).to(device)
    netC.apply(weights_init)

    logger.log(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    logger.log(f"Critic parameters: {sum(p.numel() for p in netC.parameters()):,}")

    # WGAN에서는 RMSprop 사용 (Adam 대신)
    optimizerC = optim.RMSprop(netC.parameters(), lr=lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Training loop
    logger.log("Starting WGAN training...")
    logger.log(f"Training strategy: {n_critic} critic updates per 1 generator update")

    wasserstein_distances = []  # Track Wasserstein distance over time

    for epoch in range(num_epochs):
        epoch_c_loss = 0
        epoch_g_loss = 0
        epoch_wasserstein_d = 0
        num_critic_updates = 0
        num_generator_updates = 0

        for i, (data, _) in enumerate(dataloader):
            # Update Critic
            netC.zero_grad()

            # Train critic with real data
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)

            output_real = netC(real_cpu)
            errC_real = -torch.mean(output_real)  # Maximize for real

            # Train critic with fake data
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)

            output_fake = netC(fake.detach())
            errC_fake = torch.mean(output_fake)  # Minimize for fake

            # Total critic loss (Wasserstein loss)
            errC = errC_real + errC_fake
            errC.backward()
            optimizerC.step()

            # Clip critic weights (핵심: 1-Lipschitz 제약 조건)
            clip_weights(netC, clip_value)

            epoch_c_loss += errC.item()
            num_critic_updates += 1

            # Update Generator every n_critic iterations
            if i % n_critic == 0:
                netG.zero_grad()

                # Generate fake data
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)

                # Forward pass through critic
                output = netC(fake)
                errG = -torch.mean(output)  # Maximize critic output for fake
                errG.backward()
                optimizerG.step()

                epoch_g_loss += errG.item()
                num_generator_updates += 1

                # Calculate Wasserstein distance estimate
                wasserstein_d = -errC.item()
                epoch_wasserstein_d += wasserstein_d
                wasserstein_distances.append(wasserstein_d)

        # Calculate average metrics
        avg_c_loss = epoch_c_loss / num_critic_updates if num_critic_updates > 0 else 0
        avg_g_loss = epoch_g_loss / num_generator_updates if num_generator_updates > 0 else 0
        avg_wasserstein_d = epoch_wasserstein_d / num_generator_updates if num_generator_updates > 0 else 0

        # Log metrics
        logger.log_metrics(epoch + 1, avg_g_loss,
                          critic_loss=avg_c_loss,
                          wasserstein_distance=avg_wasserstein_d,
                          critic_updates=num_critic_updates,
                          generator_updates=num_generator_updates)

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Saving intermediate results at epoch {epoch + 1}...")

            # Generate samples
            with torch.no_grad():
                fake = netG(fixed_noise)

                # Calculate current Wasserstein distance
                if len(dataloader.dataset) > 0:
                    sample_real = next(iter(dataloader))[0][:min(64, len(dataloader.dataset))].to(device)
                    current_wd = calculate_wasserstein_distance(netC, sample_real, fake[:len(sample_real)])
                    logger.log(f"Current Wasserstein Distance: {current_wd.item():.4f}")

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
            logger.save_model(netG, f"wgan_generator_epoch_{epoch+1:03d}",
                             optimizer=optimizerG, epoch=epoch+1, config=config)
            logger.save_model(netC, f"wgan_critic_epoch_{epoch+1:03d}",
                             optimizer=optimizerC, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Saving final results...")

    # Plot training curves (including Wasserstein distance)
    logger.plot_training_curves()

    # Plot Wasserstein distance over time
    if wasserstein_distances:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(wasserstein_distances)
        plt.title('Wasserstein Distance Over Training')
        plt.xlabel('Iteration')
        plt.ylabel('Wasserstein Distance')
        plt.grid(True)
        plt.savefig(os.path.join(logger.dirs['plots'], 'wasserstein_distance_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Save final models
    logger.save_model(netG, "wgan_generator_final", optimizer=optimizerG,
                     epoch=num_epochs, config=config)
    logger.save_model(netC, "wgan_critic_final", optimizer=optimizerC,
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

    return netG, netC, results_dir

if __name__ == "__main__":
    print("Medical WGAN Implementation")
    print("===========================")

    # Configuration
    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'

    print(f"Training WGAN on {selected_dataset} images...")
    print("WGAN Key Features:")
    print("- Uses Wasserstein distance for better training stability")
    print("- Critic network (no sigmoid) with weight clipping")
    print("- Multiple critic updates per generator update")
    print("- RMSprop optimizer with lower learning rate")
    print("- Meaningful loss values for convergence monitoring")

    # Train the model
    try:
        netG, netC, results_dir = train_medical_wgan(
            dataset_type=selected_dataset,
            data_path=None,
            num_epochs=50,
            save_interval=5
        )

        print(f"\nTraining completed successfully!")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Generated medical image samples")
        print("- models/: Generator and critic checkpoints")
        print("- logs/: Training logs with Wasserstein distance tracking")
        print("- plots/: Training curves including Wasserstein distance")
        print("- metrics/: Training metrics in JSON format")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
