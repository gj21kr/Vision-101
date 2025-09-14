"""
DCGAN (Deep Convolutional Generative Adversarial Networks) Implementation

DCGAN은 2015년 Radford et al.에 의해 제안된 GAN의 확장으로, 다음과 같은 핵심 특징을 가집니다:

1. 완전 연결 층을 합성곱 층으로 대체 (All convolutional net)
2. Pooling 층 제거, 대신 strided convolution과 transposed convolution 사용
3. Batch Normalization을 Generator와 Discriminator 모두에 적용
4. Generator에는 ReLU, Discriminator에는 LeakyReLU 활성화 함수 사용
5. Generator의 출력층에는 Tanh, Discriminator의 출력층에는 Sigmoid 사용

이러한 구조적 개선으로 인해 더 안정적인 훈련과 고품질 이미지 생성이 가능해졌습니다.

Reference:
- Radford, A., Metz, L., & Chintala, S. (2015).
  "Unsupervised representation learning with deep convolutional generative adversarial networks."
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def weights_init(m):
    """
    DCGAN 논문에서 권장하는 가중치 초기화 함수

    Args:
        m: 신경망 모듈

    설명:
        - Conv 층: 평균 0, 표준편차 0.02의 정규분포로 초기화
        - BatchNorm 층: 가중치는 평균 1, 표준편차 0.02로, 편향은 0으로 초기화
        - 이러한 초기화는 DCGAN의 안정적인 훈련에 중요한 역할을 함
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # 합성곱 층의 가중치 초기화 (평균=0, 표준편차=0.02)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # 배치 정규화 층의 가중치는 평균=1, 표준편차=0.02로 초기화
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        # 편향은 0으로 초기화
        nn.init.constant_(m.bias.data, 0)

class DCGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
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
    def __init__(self, nc=3, ndf=64):
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

def train_dcgan():
    # Hyperparameters
    batch_size = 64
    image_size = 64
    nc = 3  # Number of channels
    nz = 100  # Size of latent vector
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in discriminator
    num_epochs = 25
    lr = 0.0002
    beta1 = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Create networks
    netG = DCGANGenerator(nz, ngf, nc).to(device)
    netG.apply(weights_init)

    netD = DCGANDiscriminator(nc, ndf).to(device)
    netD.apply(weights_init)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Labels
    real_label = 1.
    fake_label = 0.

    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Update Discriminator
            netD.zero_grad()

            # Train with real batch
            real_cpu = data[0].to(device)
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

            # Print statistics
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

        # Generate samples
        with torch.no_grad():
            fake = netG(fixed_noise)

        # Save samples every 5 epochs
        if epoch % 5 == 0:
            vutils.save_image(fake.detach(), f'dcgan_samples_epoch_{epoch}.png',
                            normalize=True, nrow=8)

    return netG, netD

def generate_samples(generator, num_samples=64, nz=100):
    generator.eval()
    device = next(generator.parameters()).device

    with torch.no_grad():
        noise = torch.randn(num_samples, nz, 1, 1, device=device)
        fake_images = generator(noise)

        # Create grid and display
        grid = vutils.make_grid(fake_images, nrow=8, normalize=True)

        plt.figure(figsize=(15, 15))
        plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
        plt.title("DCGAN Generated Images")
        plt.axis('off')
        plt.show()

def interpolate_latent_space(generator, nz=100, steps=10):
    generator.eval()
    device = next(generator.parameters()).device

    with torch.no_grad():
        # Create two random points in latent space
        z1 = torch.randn(1, nz, 1, 1, device=device)
        z2 = torch.randn(1, nz, 1, 1, device=device)

        interpolated_images = []
        for i in range(steps):
            alpha = i / (steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            fake_img = generator(z_interp)
            interpolated_images.append(fake_img)

        # Concatenate and display
        interpolated_tensor = torch.cat(interpolated_images, dim=0)
        grid = vutils.make_grid(interpolated_tensor, nrow=steps, normalize=True)

        plt.figure(figsize=(20, 4))
        plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
        plt.title("DCGAN Latent Space Interpolation")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    print("DCGAN Example - Deep Convolutional GAN")
    print("Features:")
    print("- Deep convolutional architecture")
    print("- Batch normalization")
    print("- LeakyReLU and ReLU activations")
    print("- Transposed convolutions for upsampling")
    print("- Proper weight initialization")

    # Uncomment to train
    # generator, discriminator = train_dcgan()
    # generate_samples(generator)
    # interpolate_latent_space(generator)