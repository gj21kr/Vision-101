"""
Medical CycleGAN Implementation

의료 이미지간 변환을 위한 CycleGAN 구현입니다.
CycleGAN은 paired data 없이도 도메인 간 이미지 변환을 수행할 수 있습니다.

의료 분야에서의 활용:
- 서로 다른 의료 영상 모달리티 간 변환 (예: MRI ↔ CT)
- 다른 촬영 조건/장비 간 이미지 스타일 변환
- 정상 ↔ 비정상 이미지 변환 (데이터 증강)
- 서로 다른 해상도나 품질의 이미지 간 변환

의료 특화 기능:
- 의료 이미지 특화 데이터 로더
- 자동 결과 저장 및 로깅 시스템
- 양방향 변환 결과 시각화
- 의료 이미지 품질 평가 메트릭

CycleGAN의 핵심 특징:
1. 두 개의 Generator: G_A→B, G_B→A
2. 두 개의 Discriminator: D_A, D_B
3. Cycle Consistency Loss: ||G_B→A(G_A→B(A)) - A||
4. Adversarial Loss: Generator vs Discriminator
5. Identity Loss: 같은 도메인 입력 시 변화 최소화

Reference:
- Zhu, J. Y., et al. (2017).
  "Unpaired image-to-image translation using cycle-consistent adversarial networks."
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

class ResidualBlock(nn.Module):
    """Residual block for CycleGAN generator"""
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class CycleGANGenerator(nn.Module):
    """
    CycleGAN Generator: U-Net like architecture with residual blocks

    구조:
    1. Encoder: 입력을 점진적으로 다운샘플링
    2. Residual Blocks: 특징 변환을 위한 bottleneck
    3. Decoder: 출력을 점진적으로 업샘플링

    의료 이미지에 특화된 특징:
    - Reflection padding으로 경계 아티팩트 방지
    - Instance normalization으로 스타일 정규화
    - Skip connections 없음 (도메인 변환에 집중)
    """
    def __init__(self, input_nc=1, output_nc=1, n_residual_blocks=9, ngf=64):
        super(CycleGANGenerator, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling (Encoder)
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks (Feature transformation)
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling (Decoder)
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3,
                                 stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        도메인 변환 수행

        Args:
            x: 소스 도메인 이미지 [B, C, H, W]

        Returns:
            변환된 타겟 도메인 이미지 [B, C, H, W]
        """
        return self.model(x)

class CycleGANDiscriminator(nn.Module):
    """
    CycleGAN Discriminator: PatchGAN 구조

    특징:
    - 전체 이미지가 아닌 patch 단위로 진짜/가짜 판별
    - 70x70 receptive field (PatchGAN-70)
    - 의료 이미지의 local texture에 집중
    """
    def __init__(self, input_nc=1, ndf=64):
        super(CycleGANDiscriminator, self).__init__()

        # C64-C128-C256-C512
        model = [
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # Output layer (no sigmoid for LSGAN)
        model += [nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """
        Patch-wise 진짜/가짜 판별

        Returns:
            Patch별 판별 결과 [B, 1, H', W']
        """
        return self.model(x)

class MedicalCycleDataset(Dataset):
    """
    두 도메인의 의료 이미지를 위한 데이터셋

    CycleGAN은 unpaired 데이터를 사용하므로, 두 도메인의 이미지가
    일대일 대응될 필요가 없습니다.
    """
    def __init__(self, images_A, images_B, image_size=256, channels=1):
        self.images_A = images_A
        self.images_B = images_B
        self.image_size = image_size
        self.channels = channels

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * channels, (0.5,) * channels)
        ])

        # 길이를 더 긴 도메인에 맞춤
        self.length = max(len(images_A), len(images_B))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Cycle through datasets if one is shorter
        img_A = self.images_A[idx % len(self.images_A)]
        img_B = self.images_B[idx % len(self.images_B)]

        # Format images
        for img in [img_A, img_B]:
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            elif len(img.shape) == 3 and self.channels == 1:
                if img.shape[-1] == 3:
                    img = np.mean(img, axis=-1, keepdims=True)

        img_A = self.transform(img_A.astype(np.uint8))
        img_B = self.transform(img_B.astype(np.uint8))

        return img_A, img_B

def create_dual_domain_data(dataset_type_A, dataset_type_B, num_samples=500, image_size=256):
    """
    두 의료 이미지 도메인을 위한 데이터 생성

    예시:
    - chest_xray ↔ brain_mri
    - normal ↔ abnormal (같은 모달리티)
    - low_res ↔ high_res
    """
    # Load domain A
    if dataset_type_A == 'chest_xray':
        images_A = load_chest_xray_data(None, num_samples, image_size)
    elif dataset_type_A == 'brain_mri':
        images_A = load_brain_mri_data(None, num_samples, image_size)
    else:
        loader_A = MedicalImageLoader(dataset_type_A, image_size)
        images_A = loader_A.create_synthetic_medical_data(num_samples)

    # Load domain B
    if dataset_type_B == 'chest_xray':
        images_B = load_chest_xray_data(None, num_samples, image_size)
    elif dataset_type_B == 'brain_mri':
        images_B = load_brain_mri_data(None, num_samples, image_size)
    else:
        loader_B = MedicalImageLoader(dataset_type_B, image_size)
        images_B = loader_B.create_synthetic_medical_data(num_samples)

    return images_A, images_B

def calculate_quality_score(images):
    """의료 이미지 품질 평가"""
    if len(images) == 0:
        return 0.0
    images_np = images.detach().cpu().numpy()
    variance = np.var(images_np)
    return float(variance)

def train_medical_cyclegan(domain_A='chest_xray', domain_B='brain_mri',
                          data_path_A=None, data_path_B=None,
                          num_epochs=200, save_interval=20):
    """
    Medical CycleGAN 훈련 함수

    Args:
        domain_A, domain_B: 두 의료 이미지 도메인
        data_path_A, data_path_B: 실제 데이터 경로
        num_epochs: 훈련 에포크 수 (CycleGAN은 보통 더 많이 필요)
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating(f"cyclegan_{domain_A}_to_{domain_B}", f"{domain_A}_to_{domain_B}")

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        num_epochs = int(os.getenv('TEST_EPOCHS', 5))
        logger.log(f"Quick test mode: training for {num_epochs} epochs only")

    # CycleGAN hyperparameters
    batch_size = 16  # CycleGAN은 메모리를 많이 사용
    image_size = 128  # 의료 이미지에 적합한 크기
    lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    lambda_cycle = 10.0  # Cycle consistency loss weight
    lambda_identity = 5.0  # Identity loss weight

    # Save configuration
    config = {
        'algorithm': 'CycleGAN',
        'domain_A': domain_A,
        'domain_B': domain_B,
        'batch_size': batch_size,
        'image_size': image_size,
        'lr': lr,
        'beta1': beta1,
        'beta2': beta2,
        'lambda_cycle': lambda_cycle,
        'lambda_identity': lambda_identity,
        'num_epochs': num_epochs,
        'data_path_A': data_path_A,
        'data_path_B': data_path_B
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Load medical data for both domains
    logger.log(f"Loading {domain_A} and {domain_B} data...")
    images_A, images_B = create_dual_domain_data(domain_A, domain_B, 500, image_size)

    # Determine channels
    input_channels = 1 if len(images_A[0].shape) <= 3 and images_A[0].shape[-1] == 1 else 1

    logger.log(f"Domain A ({domain_A}): {len(images_A)} images, shape: {images_A[0].shape}")
    logger.log(f"Domain B ({domain_B}): {len(images_B)} images, shape: {images_B[0].shape}")

    # Save sample images from both domains
    sample_images_A = [images_A[i] for i in range(min(6, len(images_A)))]
    sample_images_B = [images_B[i] for i in range(min(6, len(images_B)))]

    logger.save_image_grid(sample_images_A, f"original_samples_domain_A_{domain_A}",
                          titles=[f"A_{i+1}" for i in range(len(sample_images_A))],
                          cmap='gray')
    logger.save_image_grid(sample_images_B, f"original_samples_domain_B_{domain_B}",
                          titles=[f"B_{i+1}" for i in range(len(sample_images_B))],
                          cmap='gray')

    # Create dataset and dataloader
    dataset = MedicalCycleDataset(images_A, images_B, image_size, input_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create networks
    # Two generators: A→B and B→A
    netG_A2B = CycleGANGenerator(input_channels, input_channels).to(device)
    netG_B2A = CycleGANGenerator(input_channels, input_channels).to(device)

    # Two discriminators: for domain A and B
    netD_A = CycleGANDiscriminator(input_channels).to(device)
    netD_B = CycleGANDiscriminator(input_channels).to(device)

    # Initialize weights
    def weights_init(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    netG_A2B.apply(weights_init)
    netG_B2A.apply(weights_init)
    netD_A.apply(weights_init)
    netD_B.apply(weights_init)

    logger.log(f"Generator A2B parameters: {sum(p.numel() for p in netG_A2B.parameters()):,}")
    logger.log(f"Generator B2A parameters: {sum(p.numel() for p in netG_B2A.parameters()):,}")
    logger.log(f"Discriminator A parameters: {sum(p.numel() for p in netD_A.parameters()):,}")
    logger.log(f"Discriminator B parameters: {sum(p.numel() for p in netD_B.parameters()):,}")

    # Loss functions
    criterion_GAN = nn.MSELoss()  # LSGAN loss
    criterion_cycle = nn.L1Loss()  # Cycle consistency loss
    criterion_identity = nn.L1Loss()  # Identity loss

    # Optimizers
    optimizer_G = optim.Adam(
        list(netG_A2B.parameters()) + list(netG_B2A.parameters()),
        lr=lr, betas=(beta1, beta2)
    )
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=lr, betas=(beta1, beta2))

    # Fixed samples for visualization
    fixed_A = torch.stack([dataset[i][0] for i in range(8)]).to(device)
    fixed_B = torch.stack([dataset[i][1] for i in range(8)]).to(device)

    # Training loop
    logger.log("Starting CycleGAN training...")
    logger.log(f"Domain transformation: {domain_A} ↔ {domain_B}")

    for epoch in range(num_epochs):
        epoch_loss_G = 0
        epoch_loss_D_A = 0
        epoch_loss_D_B = 0
        epoch_loss_cycle = 0
        epoch_loss_identity = 0
        num_batches = 0

        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            batch_size = real_A.size(0)

            # Labels for discriminator
            # Get discriminator output size dynamically
            with torch.no_grad():
                sample_output = netD_A(real_A[:1])  # Use first sample to get output size
                disc_output_size = sample_output.shape[2:]  # Get spatial dimensions (H, W)

            valid = torch.ones(batch_size, 1, *disc_output_size, device=device, requires_grad=False)
            fake = torch.zeros(batch_size, 1, *disc_output_size, device=device, requires_grad=False)

            # =================
            # Train Generators
            # =================
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B should be identity if real_B is fed: ||G_A2B(B) - B||
            identity_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(identity_B, real_B)

            # G_B2A should be identity if real_A is fed: ||G_B2A(A) - A||
            identity_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(identity_A, real_A)

            loss_identity = (loss_identity_A + loss_identity_B) / 2

            # GAN loss
            fake_B = netG_A2B(real_A)  # G_A2B(A)
            pred_fake_B = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake_B, valid)

            fake_A = netG_B2A(real_B)  # G_B2A(B)
            pred_fake_A = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake_A, valid)

            loss_GAN = (loss_GAN_A2B + loss_GAN_B2A) / 2

            # Cycle loss
            # Forward cycle: A -> B -> A
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)

            # Backward cycle: B -> A -> B
            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

            loss_cycle = (loss_cycle_ABA + loss_cycle_BAB) / 2

            # Total generator loss
            loss_G = loss_GAN + lambda_cycle * loss_cycle + lambda_identity * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # ====================
            # Train Discriminator A
            # ====================
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # ====================
            # Train Discriminator B
            # ====================
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            # Statistics
            epoch_loss_G += loss_G.item()
            epoch_loss_D_A += loss_D_A.item()
            epoch_loss_D_B += loss_D_B.item()
            epoch_loss_cycle += loss_cycle.item()
            epoch_loss_identity += loss_identity.item()
            num_batches += 1

        # Calculate averages
        avg_loss_G = epoch_loss_G / num_batches
        avg_loss_D_A = epoch_loss_D_A / num_batches
        avg_loss_D_B = epoch_loss_D_B / num_batches
        avg_loss_cycle = epoch_loss_cycle / num_batches
        avg_loss_identity = epoch_loss_identity / num_batches

        # Log metrics
        logger.log_metrics(epoch + 1, avg_loss_G,
                          discriminator_A_loss=avg_loss_D_A,
                          discriminator_B_loss=avg_loss_D_B,
                          cycle_loss=avg_loss_cycle,
                          identity_loss=avg_loss_identity)

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Saving intermediate results at epoch {epoch + 1}...")

            with torch.no_grad():
                # Generate translations
                fake_B_from_A = netG_A2B(fixed_A)
                fake_A_from_B = netG_B2A(fixed_B)

                # Cycle reconstructions
                cycle_A = netG_B2A(fake_B_from_A)
                cycle_B = netG_A2B(fake_A_from_B)

                # Convert for saving
                def convert_for_saving(tensor):
                    tensor = (tensor + 1) / 2
                    tensor = torch.clamp(tensor, 0, 1)
                    tensor = tensor.squeeze(1).cpu().numpy()
                    return (tensor * 255).astype(np.uint8)

                fixed_A_imgs = convert_for_saving(fixed_A)
                fixed_B_imgs = convert_for_saving(fixed_B)
                fake_B_imgs = convert_for_saving(fake_B_from_A)
                fake_A_imgs = convert_for_saving(fake_A_from_B)
                cycle_A_imgs = convert_for_saving(cycle_A)
                cycle_B_imgs = convert_for_saving(cycle_B)

                # Save transformation results
                logger.save_image_grid(
                    [fixed_A_imgs[j] for j in range(8)] +
                    [fake_B_imgs[j] for j in range(8)] +
                    [cycle_A_imgs[j] for j in range(8)],
                    f"transformation_A_to_B_epoch_{epoch+1:03d}",
                    titles=([f"Real_A_{j+1}" for j in range(8)] +
                           [f"Fake_B_{j+1}" for j in range(8)] +
                           [f"Cycle_A_{j+1}" for j in range(8)]),
                    rows=3, cols=8, cmap='gray'
                )

                logger.save_image_grid(
                    [fixed_B_imgs[j] for j in range(8)] +
                    [fake_A_imgs[j] for j in range(8)] +
                    [cycle_B_imgs[j] for j in range(8)],
                    f"transformation_B_to_A_epoch_{epoch+1:03d}",
                    titles=([f"Real_B_{j+1}" for j in range(8)] +
                           [f"Fake_A_{j+1}" for j in range(8)] +
                           [f"Cycle_B_{j+1}" for j in range(8)]),
                    rows=3, cols=8, cmap='gray'
                )

                # Quality metrics
                quality_score_A = calculate_quality_score(fake_A_from_B)
                quality_score_B = calculate_quality_score(fake_B_from_A)
                logger.log(f"Epoch {epoch+1} - Quality A: {quality_score_A:.4f}, Quality B: {quality_score_B:.4f}")

        # Save model checkpoints
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(netG_A2B, f"cyclegan_G_A2B_epoch_{epoch+1:03d}",
                             optimizer=optimizer_G, epoch=epoch+1, config=config)
            logger.save_model(netG_B2A, f"cyclegan_G_B2A_epoch_{epoch+1:03d}",
                             optimizer=optimizer_G, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Saving final results...")

    # Plot training curves
    logger.plot_training_curves()

    # Save final models
    logger.save_model(netG_A2B, f"cyclegan_G_A2B_final", optimizer=optimizer_G,
                     epoch=num_epochs, config=config)
    logger.save_model(netG_B2A, f"cyclegan_G_B2A_final", optimizer=optimizer_G,
                     epoch=num_epochs, config=config)
    logger.save_model(netD_A, f"cyclegan_D_A_final", optimizer=optimizer_D_A,
                     epoch=num_epochs, config=config)
    logger.save_model(netD_B, f"cyclegan_D_B_final", optimizer=optimizer_D_B,
                     epoch=num_epochs, config=config)

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return netG_A2B, netG_B2A, netD_A, netD_B, results_dir

if __name__ == "__main__":
    print("Medical CycleGAN Implementation")
    print("===============================")

    # Configuration - 의료 도메인 간 변환 설정
    domain_pairs = [
        ('chest_xray', 'brain_mri'),
        # ('normal_xray', 'abnormal_xray'),  # 같은 모달리티에서 정상/비정상
        # ('low_res_mri', 'high_res_mri'),   # 해상도 변환
    ]

    domain_A, domain_B = domain_pairs[0]  # 첫 번째 페어 선택

    print(f"Training CycleGAN for {domain_A} ↔ {domain_B} transformation...")
    print("CycleGAN Key Features:")
    print("- Unpaired domain-to-domain translation")
    print("- Cycle consistency for structure preservation")
    print("- Identity loss for color consistency")
    print("- Bidirectional transformation capability")
    print("- PatchGAN discriminator for local realism")

    # Train the model
    try:
        netG_A2B, netG_B2A, netD_A, netD_B, results_dir = train_medical_cyclegan(
            domain_A=domain_A,
            domain_B=domain_B,
            data_path_A=None,
            data_path_B=None,
            num_epochs=50,  # CycleGAN은 보통 더 많은 에포크 필요
            save_interval=10
        )

        print(f"\nTraining completed successfully!")
        print(f"Domain transformation: {domain_A} ↔ {domain_B}")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Bidirectional transformation results")
        print("- models/: Two generators and two discriminators")
        print("- logs/: Training logs with cycle and identity losses")
        print("- plots/: Training loss curves for all components")
        print("- metrics/: Training metrics in JSON format")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
