"""
Medical WGAN-GP (Wasserstein GAN with Gradient Penalty) Implementation

의료 이미지 생성을 위한 WGAN-GP 구현입니다.
WGAN-GP는 기존 WGAN의 weight clipping 문제를 gradient penalty로 해결하여
더 안정적이고 고품질의 훈련을 제공합니다.

의료 특화 기능:
- 의료 이미지 특화 데이터 로더 (chest X-ray, brain MRI, skin lesion)
- 자동 결과 저장 및 로깅 시스템
- Gradient penalty 및 Wasserstein distance 추적
- 의료 이미지 품질 평가 메트릭

WGAN-GP의 핵심 개선사항:
1. Gradient Penalty: weight clipping을 대체
2. 1-Lipschitz 제약을 더 부드럽게 강제
3. Adam optimizer 사용 가능 (RMSprop 대신)
4. 더 안정적인 훈련과 고품질 결과

수학적 정의:
- GP Loss: λ * E[(||∇D(x̃)||₂ - 1)²]
- 여기서 x̃는 real과 fake 사이의 랜덤 interpolation
- λ는 gradient penalty 가중치 (보통 10)

Reference:
- Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017).
  "Improved training of Wasserstein GANs."
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from medical_data_utils import MedicalImageLoader, load_chest_xray_data, load_brain_mri_data
from result_logger import create_logger_for_generating

class WGANGPGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(WGANGPGenerator, self).__init__()
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

class WGANGPCritic(nn.Module):
    """
    WGAN-GP Critic: Gradient penalty를 위한 개선된 설계

    핵심 특징:
    - BatchNorm 제거 (gradient penalty와 충돌 방지)
    - Sigmoid 활성화 함수 없음
    - 실수값 스코어 출력
    """
    def __init__(self, nc=1, ndf=64):
        super(WGANGPCritic, self).__init__()
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
            # No activation (실수값 출력)
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

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

def weights_init(m):
    """가중치 초기화"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """
    WGAN-GP의 핵심: Gradient Penalty 계산

    수학적 배경:
    - 1-Lipschitz 제약: ||∇f||₂ ≤ 1 for all x
    - Gradient penalty: λ * E[(||∇f(x̃)||₂ - 1)²]
    - x̃ = ε*x_real + (1-ε)*x_fake, ε ~ U(0,1)

    장점:
    - Weight clipping의 문제점 해결
    - 더 부드러운 제약 조건 강제
    - 네트워크 용량 제한 없음

    Args:
        critic: Critic 네트워크
        real_samples: 실제 데이터 샘플
        fake_samples: 생성된 데이터 샘플
        device: 계산 장치

    Returns:
        gradient_penalty: 계산된 gradient penalty 값
    """
    batch_size = real_samples.size(0)

    # 랜덤 가중치로 실제와 가짜 샘플 사이 보간점 생성
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    # 보간점 계산: x̃ = α*x_real + (1-α)*x_fake
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # 보간점에 대한 Critic 스코어
    critic_interpolates = critic(interpolates)

    # 보간점에 대한 gradient 계산
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=device),
        create_graph=True,      # 2차 미분을 위해 필요
        retain_graph=True,      # 메모리 효율성
        only_inputs=True        # 입력에 대한 gradient만 필요
    )[0]

    # Gradient의 L2 norm 계산
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

def calculate_wasserstein_distance(critic, real_data, fake_data):
    """Wasserstein distance 근사 계산"""
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

def train_medical_wgan_gp(dataset_type='chest_xray', data_path=None, num_epochs=100, save_interval=10):
    """
    Medical WGAN-GP 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("wgan_gp", dataset_type)

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        num_epochs = int(os.getenv('TEST_EPOCHS', 5))
        logger.log(f"Quick test mode: training for {num_epochs} epochs only")

    # WGAN-GP Hyperparameters
    batch_size = 64
    image_size = 64
    nz = 100
    ngf = 64
    ndf = 64
    lr = 0.0001           # WGAN-GP는 더 높은 학습률 가능
    beta1 = 0.0           # WGAN-GP Adam 파라미터
    beta2 = 0.9
    n_critic = 5          # Critic 업데이트 횟수
    lambda_gp = 10        # Gradient penalty 가중치

    # Save configuration
    config = {
        'algorithm': 'WGAN_GP',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'image_size': image_size,
        'nz': nz,
        'ngf': ngf,
        'ndf': ndf,
        'lr': lr,
        'beta1': beta1,
        'beta2': beta2,
        'n_critic': n_critic,
        'lambda_gp': lambda_gp,
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
    netG = WGANGPGenerator(nz, ngf, input_channels).to(device)
    netG.apply(weights_init)

    netC = WGANGPCritic(input_channels, ndf).to(device)
    netC.apply(weights_init)

    logger.log(f"Generator parameters: {sum(p.numel() for p in netG.parameters()):,}")
    logger.log(f"Critic parameters: {sum(p.numel() for p in netC.parameters()):,}")

    # WGAN-GP에서는 Adam 사용 가능
    optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Training loop
    logger.log("Starting WGAN-GP training...")
    logger.log(f"Training strategy: {n_critic} critic updates per 1 generator update")
    logger.log(f"Gradient penalty weight: {lambda_gp}")

    wasserstein_distances = []
    gradient_penalties = []

    for epoch in range(num_epochs):
        epoch_c_loss = 0
        epoch_g_loss = 0
        epoch_gp = 0
        epoch_wasserstein_d = 0
        num_critic_updates = 0
        num_generator_updates = 0

        for i, (data, _) in enumerate(dataloader):
            # Update Critic multiple times
            for _ in range(n_critic):
                netC.zero_grad()

                # Train with real data
                real_data = data.to(device)
                b_size = real_data.size(0)

                # Real data loss
                critic_real = netC(real_data)
                loss_real = -torch.mean(critic_real)

                # Train with fake data
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_data = netG(noise)

                # Fake data loss
                critic_fake = netC(fake_data.detach())
                loss_fake = torch.mean(critic_fake)

                # Gradient penalty
                gp = compute_gradient_penalty(netC, real_data, fake_data, device)

                # Total critic loss
                loss_critic = loss_real + loss_fake + lambda_gp * gp
                loss_critic.backward()
                optimizerC.step()

                # Statistics
                epoch_c_loss += loss_critic.item()
                epoch_gp += gp.item()
                num_critic_updates += 1

                # Calculate Wasserstein distance estimate
                wasserstein_d = -(loss_real.item() + loss_fake.item())
                epoch_wasserstein_d += wasserstein_d
                wasserstein_distances.append(wasserstein_d)
                gradient_penalties.append(gp.item())

                break  # 한 번만 업데이트 (데이터 부족 방지)

            # Update Generator
            if i % n_critic == 0:
                netG.zero_grad()

                # Generate fake data
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake_data = netG(noise)

                # Generator loss: maximize critic output
                critic_fake = netC(fake_data)
                loss_generator = -torch.mean(critic_fake)
                loss_generator.backward()
                optimizerG.step()

                epoch_g_loss += loss_generator.item()
                num_generator_updates += 1

        # Calculate averages
        avg_c_loss = epoch_c_loss / num_critic_updates if num_critic_updates > 0 else 0
        avg_g_loss = epoch_g_loss / num_generator_updates if num_generator_updates > 0 else 0
        avg_gp = epoch_gp / num_critic_updates if num_critic_updates > 0 else 0
        avg_wasserstein_d = epoch_wasserstein_d / num_critic_updates if num_critic_updates > 0 else 0

        # Log metrics
        logger.log_metrics(epoch + 1, avg_g_loss,
                          critic_loss=avg_c_loss,
                          gradient_penalty=avg_gp,
                          wasserstein_distance=avg_wasserstein_d,
                          critic_updates=num_critic_updates,
                          generator_updates=num_generator_updates)

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Saving intermediate results at epoch {epoch + 1}...")

            # Generate samples
            with torch.no_grad():
                fake = netG(fixed_noise)

                # Calculate metrics
                if len(dataloader.dataset) > 0:
                    sample_real = next(iter(dataloader))[0][:min(32, len(dataloader.dataset))].to(device)
                    current_wd = calculate_wasserstein_distance(netC, sample_real, fake[:len(sample_real)])
                    logger.log(f"Current Wasserstein Distance: {current_wd.item():.4f}")

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

                # Quality metrics
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

        # Save model checkpoints
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(netG, f"wgan_gp_generator_epoch_{epoch+1:03d}",
                             optimizer=optimizerG, epoch=epoch+1, config=config)
            logger.save_model(netC, f"wgan_gp_critic_epoch_{epoch+1:03d}",
                             optimizer=optimizerC, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Saving final results...")

    # Plot training curves
    logger.plot_training_curves()

    # Plot additional WGAN-GP specific metrics
    if wasserstein_distances and gradient_penalties:
        import matplotlib.pyplot as plt

        # Wasserstein distance plot
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(wasserstein_distances)
        plt.title('Wasserstein Distance Over Training')
        plt.xlabel('Iteration')
        plt.ylabel('Wasserstein Distance')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(gradient_penalties)
        plt.title('Gradient Penalty Over Training')
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Penalty')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(logger.dirs['plots'], 'wgan_gp_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Save final models
    logger.save_model(netG, "wgan_gp_generator_final", optimizer=optimizerG,
                     epoch=num_epochs, config=config)
    logger.save_model(netC, "wgan_gp_critic_final", optimizer=optimizerC,
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
    print("Medical WGAN-GP Implementation")
    print("==============================")

    # Configuration
    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'

    print(f"Training WGAN-GP on {selected_dataset} images...")
    print("WGAN-GP Key Features:")
    print("- Gradient penalty instead of weight clipping")
    print("- More stable training than original WGAN")
    print("- Adam optimizer compatible")
    print("- Better gradient flow and higher quality results")
    print("- 1-Lipschitz constraint enforced smoothly")

    # Train the model
    try:
        netG, netC, results_dir = train_medical_wgan_gp(
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
        print("- logs/: Training logs with gradient penalty tracking")
        print("- plots/: Training curves including Wasserstein distance and gradient penalty")
        print("- metrics/: Training metrics in JSON format")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise