"""
Super Resolution Implementation

Super Resolution은 저해상도 이미지를 고해상도로 변환하는 컴퓨터 비전 기술입니다.
이 파일은 SRCNN, ESPCN, SRGAN 등 주요 딥러닝 기반 초해상도 기법들을 구현합니다.

주요 모델들:

1. **SRCNN (Super-Resolution CNN)**:
   - 최초의 CNN 기반 초해상도 방법
   - 3층 구조: 특징 추출 → 비선형 매핑 → 재구성
   - 입력: Bicubic 업샘플링된 저해상도 이미지

2. **ESPCN (Efficient Sub-Pixel CNN)**:
   - Sub-pixel convolution을 통한 효율적 업샘플링
   - 저해상도에서 직접 작업 후 마지막에 업샘플링
   - PixelShuffle 연산으로 r² 배 업샘플링

3. **SRGAN (Super-Resolution GAN)**:
   - GAN을 활용한 고품질 초해상도
   - Perceptual loss (VGG feature 기반) 사용
   - ResNet 기반 생성자 + PatchGAN 판별자

핵심 기술:
1. **Sub-pixel Convolution**:
   - 저해상도 공간에서 r²배 많은 채널 생성 후 재배열
   - 계산 효율성과 성능 향상을 동시에 달성

2. **Perceptual Loss**:
   - 픽셀 단위 MSE 대신 VGG 특징 공간에서 거리 측정
   - 인간의 지각과 더 일치하는 결과

3. **Residual Learning**:
   - Skip connection을 통한 깊은 네트워크 훈련
   - 고주파 세부사항 복원에 효과적

평가 메트릭:
- PSNR (Peak Signal-to-Noise Ratio): 객관적 품질 측정
- SSIM (Structural Similarity Index): 구조적 유사도
- LPIPS (Learned Perceptual Image Patch Similarity): 지각적 품질

응용 분야:
- 저해상도 사진/비디오 향상
- 의료 영상 해상도 개선
- 위성/감시 이미지 선명화
- 게임/영화 리마스터링

References:
- SRCNN: Dong et al. (2014) "Learning a deep convolutional network for image super-resolution"
- ESPCN: Shi et al. (2016) "Real-time single image and video super-resolution"
- SRGAN: Ledig et al. (2017) "Photo-realistic single image super-resolution using a GAN"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        """
        SRCNN forward pass: 3단계 CNN 기반 super resolution

        SRCNN의 3단계 구조:
        1. Patch Extraction: 9x9 컨볼루션으로 64차원 특징 추출
        2. Non-linear Mapping: 5x5 컨볼루션으로 32차원으로 매핑
        3. Reconstruction: 5x5 컨볼루션으로 최종 고해상도 이미지 생성

        Args:
            x (torch.Tensor): 업샘플링된 저해상도 입력 [batch, channels, H, W]

        Returns:
            torch.Tensor: 고해상도 출력 [batch, channels, H, W]

        주요 특징:
        - 비선형 연산 활용으로 복잡한 패턴 학습
        - End-to-end 학습으로 전체 파이프라인 최적화
        - 비슷한 크기 유지 (업샘플링 전에 수행)
        """
        x = F.relu(self.conv1(x))  # Patch extraction & feature mapping
        x = F.relu(self.conv2(x))  # Non-linear mapping
        x = self.conv3(x)          # Reconstruction
        return x

class ESPCN(nn.Module):
    def __init__(self, upscale_factor=3, num_channels=1):
        super(ESPCN, self).__init__()
        self.upscale_factor = upscale_factor

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, upscale_factor ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        """
        ESPCN forward pass: Sub-pixel 컨볼루션 기반 효율적 upscaling

        ESPCN (Efficient Sub-Pixel CNN)의 혁신:
        - 마지막에 upsampling 수행 (전체 계산량 감소)
        - Sub-pixel convolution으로 학습 가능한 upsampling
        - Real-time 영상 처리에 적합

        Args:
            x (torch.Tensor): 저해상도 입력 [batch, channels, H, W]

        Returns:
            torch.Tensor: 고해상도 출력 [batch, channels, H*scale, W*scale]

        주요 과정:
        1. 저해상도에서 feature extraction (conv1-3)
        2. upscale_factor^2 개의 채널 생성 (conv4)
        3. Pixel shuffle로 rearrangement 및 upsampling

        Pixel Shuffle 원리:
        - [B, r^2*C, H, W] → [B, C, r*H, r*W]
        - 채널 차원의 정보를 공간 차원으로 재배치
        """
        x = F.relu(self.conv1(x))     # Feature extraction
        x = F.relu(self.conv2(x))     # Deep feature learning
        x = F.relu(self.conv3(x))     # Feature refinement
        x = self.conv4(x)             # Generate r^2 channels
        x = self.pixel_shuffle(x)     # Sub-pixel upsampling
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual

class SRGAN_Generator(nn.Module):
    def __init__(self, upscale_factor=4, num_residual_blocks=16):
        super(SRGAN_Generator, self).__init__()
        self.upscale_factor = upscale_factor

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)

        # Residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])

        # Post-residual convolution
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(64)

        # Upsampling layers
        upsampling = []
        for _ in range(int(math.log(upscale_factor, 2))):
            upsampling += [
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU()
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out = self.residual_blocks(out1)
        out2 = self.bn(self.conv2(out))
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = torch.tanh(self.conv3(out))
        return out

class SRGAN_Discriminator(nn.Module):
    def __init__(self):
        super(SRGAN_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)

        # Adaptive pooling and classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)

        out = F.leaky_relu(self.conv1(x), 0.2)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2)
        out = F.leaky_relu(self.bn3(self.conv3(out)), 0.2)
        out = F.leaky_relu(self.bn4(self.conv4(out)), 0.2)
        out = F.leaky_relu(self.bn5(self.conv5(out)), 0.2)
        out = F.leaky_relu(self.bn6(self.conv6(out)), 0.2)
        out = F.leaky_relu(self.bn7(self.conv7(out)), 0.2)
        out = F.leaky_relu(self.bn8(self.conv8(out)), 0.2)

        out = self.adaptive_pool(out)
        out = out.view(batch_size, -1)
        out = self.classifier(out)

        return out

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, hr_imgs, sr_imgs):
        hr_features = self.feature_extractor(hr_imgs)
        sr_features = self.feature_extractor(sr_imgs)
        return F.mse_loss(hr_features, sr_features)

def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def train_srcnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    batch_size = 64
    lr = 1e-4
    epochs = 10

    # Model
    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Transforms for creating low-res images
    transform_lr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((14, 14)),  # Downsample
        transforms.Resize((28, 28)),  # Upsample back (bicubic)
        transforms.ToTensor()
    ])

    transform_hr = transforms.Compose([
        transforms.ToTensor()
    ])

    # Data loading
    dataset = datasets.MNIST('data', train=True, download=True, transform=transform_hr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (hr_imgs, _) in enumerate(dataloader):
            hr_imgs = hr_imgs.to(device)

            # Create LR images
            lr_imgs = torch.stack([transform_lr(img) for img in hr_imgs])
            lr_imgs = lr_imgs.to(device)

            # Forward pass
            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                psnr = calculate_psnr(sr_imgs, hr_imgs)
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, PSNR: {psnr:.2f} dB')

        print(f'Epoch {epoch} Average Loss: {epoch_loss/len(dataloader):.4f}')

    return model

def visualize_results(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for batch_idx, (hr_imgs, _) in enumerate(test_loader):
            if batch_idx > 0:
                break

            hr_imgs = hr_imgs[:4].to(device)

            # Create LR images
            transform_lr = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((14, 14)),
                transforms.Resize((28, 28)),
                transforms.ToTensor()
            ])

            lr_imgs = torch.stack([transform_lr(img) for img in hr_imgs])
            lr_imgs = lr_imgs.to(device)

            # Super-resolve
            sr_imgs = model(lr_imgs)

            # Visualize
            fig, axes = plt.subplots(3, 4, figsize=(12, 9))
            for i in range(4):
                # Original HR
                axes[0, i].imshow(hr_imgs[i].cpu().squeeze(), cmap='gray')
                axes[0, i].set_title('Original HR')
                axes[0, i].axis('off')

                # LR (bicubic upsampled)
                axes[1, i].imshow(lr_imgs[i].cpu().squeeze(), cmap='gray')
                axes[1, i].set_title('LR (Bicubic)')
                axes[1, i].axis('off')

                # Super-resolved
                axes[2, i].imshow(sr_imgs[i].cpu().squeeze(), cmap='gray')
                axes[2, i].set_title('Super-resolved')
                axes[2, i].axis('off')

            plt.tight_layout()
            plt.show()
            break

def demo_super_resolution():
    print("Super Resolution Examples")
    print("Available models:")
    print("1. SRCNN - Super-Resolution Convolutional Neural Network")
    print("2. ESPCN - Efficient Sub-Pixel CNN")
    print("3. SRGAN - Super-Resolution GAN")

    # Model comparison
    srcnn = SRCNN()
    espcn = ESPCN(upscale_factor=3)
    srgan_gen = SRGAN_Generator(upscale_factor=4)

    print(f"\nModel sizes:")
    print(f"SRCNN parameters: {sum(p.numel() for p in srcnn.parameters()):,}")
    print(f"ESPCN parameters: {sum(p.numel() for p in espcn.parameters()):,}")
    print(f"SRGAN Generator parameters: {sum(p.numel() for p in srgan_gen.parameters()):,}")

if __name__ == "__main__":
    demo_super_resolution()

    # Uncomment to train SRCNN
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = train_srcnn()
    #
    # # Test visualization
    # test_dataset = datasets.MNIST('data', train=False, transform=transforms.ToTensor())
    # test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    # visualize_results(model, test_loader, device)