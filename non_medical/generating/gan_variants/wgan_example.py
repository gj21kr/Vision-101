"""
WGAN (Wasserstein GAN) Implementation

WGAN은 Arjovsky et al. (2017)에서 제안된 GAN의 개선 버전으로, 전통적인 GAN의
훈련 불안정성 문제를 해결하기 위해 Wasserstein distance를 사용합니다.

핵심 개선사항:

1. **Wasserstein Distance 사용**:
   - Jensen-Shannon divergence 대신 Earth Mover's distance 사용
   - 분포가 겹치지 않아도 의미있는 거리 측정 가능
   - W(P_r, P_g) = inf_γ∈Π(P_r,P_g) E_(x,y)~γ[||x - y||]

2. **Critic Network (not Discriminator)**:
   - 확률 출력이 아닌 실수값 스코어 출력
   - Sigmoid 활성화 함수 제거
   - 더 의미있는 손실 함수 값

3. **1-Lipschitz Constraint**:
   - Critic이 1-Lipschitz 함수가 되도록 제약
   - Weight clipping으로 구현: w ∈ [-c, c]
   - 이론적 보장을 위한 필수 조건

수학적 목표:
- max_D E_x~P_r[D(x)] - E_z~P_z[D(G(z))]  (Critic 최대화)
- min_G E_z~P_z[D(G(z))]                   (Generator 최소화)

훈련 특징:
- Discriminator → Critic (여러 번 업데이트)
- Generator 1번 업데이트당 Critic n_critic번 업데이트 (보통 5)
- RMSprop optimizer 사용 (Adam 대신)
- 낮은 학습률 (0.00005)

장점:
- 훈련 안정성 크게 개선
- Mode collapse 현상 감소
- 의미있는 손실 함수 값 (수렴 판단 가능)
- 더 나은 이론적 보장

단점:
- Weight clipping으로 인한 문제들
- 용량(capacity) 제한
- 느린 수렴 속도

이후 WGAN-GP에서 gradient penalty로 weight clipping 문제 해결

Reference:
- Arjovsky, M., Chintala, S., & Bottou, L. (2017).
  "Wasserstein generative adversarial networks."
  International Conference on Machine Learning (ICML).
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

class WGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
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
        """
        Generator forward pass: 잠재 벡터를 이미지로 변환

        Args:
            input (torch.Tensor): 잠재 벡터 z [batch_size, nz, 1, 1]

        Returns:
            torch.Tensor: 생성된 이미지 [batch_size, nc, 64, 64], 범위 [-1, 1]

        Note:
            Tanh 활성화로 출력을 [-1, 1] 범위로 정규화
        """
        return self.main(input)

class WGANCritic(nn.Module):
    def __init__(self, nc=3, ndf=64):
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
            # No sigmoid! This is the key difference from DCGAN
        )

    def forward(self, input):
        """
        Critic forward pass: 이미지의 "진짜도" 점수 출력

        WGAN Critic의 핵심 특징:
        - Sigmoid 없음: 확률이 아닌 실수 점수 출력
        - 높을수록 "더 진짜같은" 이미지를 의미
        - 1-Lipschitz 제약을 통해 Wasserstein distance 근사

        Args:
            input (torch.Tensor): 입력 이미지 [batch_size, nc, 64, 64]

        Returns:
            torch.Tensor: Critic 점수 [batch_size], 실수값

        수학적 의미:
            f(x) ≈ Wasserstein distance의 dual form
            E[f(x_real)] - E[f(x_fake)] → Wasserstein distance
        """
        return self.main(input).view(-1, 1).squeeze(1)

def weights_init(m):
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
    - |f(x₁) - f(x₂)| ≤ |x₁ - x₂| (Lipschitz 조건)
    - Weight clipping으로 이를 근사적으로 달성

    Args:
        net (nn.Module): Critic 네트워크
        clip_value (float): 클리핑 범위 [-c, c]

    문제점:
    - Weight 분포가 극단값에 집중됨
    - 네트워크 용량(capacity) 제한
    - WGAN-GP에서 gradient penalty로 해결

    수학적 근거:
        w ∈ [-c, c] ⇒ 대략적으로 1-Lipschitz 조건 만족
    """
    for param in net.parameters():
        param.data.clamp_(-clip_value, clip_value)

def train_wgan():
    # Hyperparameters
    batch_size = 64
    image_size = 64
    nc = 3  # Number of channels
    nz = 100  # Size of latent vector
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in critic
    num_epochs = 25
    lr = 0.00005  # Lower learning rate for WGAN
    n_critic = 5  # Number of critic updates per generator update
    clip_value = 0.01  # Clipping parameter

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
    netG = WGANGenerator(nz, ngf, nc).to(device)
    netG.apply(weights_init)

    netC = WGANCritic(nc, ndf).to(device)
    netC.apply(weights_init)

    # Optimizers (RMSprop is recommended for WGAN)
    optimizerC = optim.RMSprop(netC.parameters(), lr=lr)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lr)

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Update Critic
            netC.zero_grad()

            # Train critic with real data
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)

            # Forward pass real batch through C
            output_real = netC(real_cpu)
            # Calculate loss on real data (maximize)
            errC_real = -torch.mean(output_real)

            # Train critic with fake data
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)

            # Forward pass fake batch through C
            output_fake = netC(fake.detach())
            # Calculate loss on fake data (minimize)
            errC_fake = torch.mean(output_fake)

            # Total critic loss
            errC = errC_real + errC_fake
            errC.backward()
            optimizerC.step()

            # Clip critic weights
            clip_weights(netC, clip_value)

            # Update Generator every n_critic iterations
            if i % n_critic == 0:
                netG.zero_grad()

                # Generate fake data
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)

                # Forward pass through critic
                output = netC(fake)
                # Calculate generator loss (maximize critic output for fake data)
                errG = -torch.mean(output)
                errG.backward()
                optimizerG.step()

                # Calculate Wasserstein distance estimate
                wasserstein_d = -errC.item()

                # Print statistics
                if i % 50 == 0:
                    print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                          f'Loss_C: {errC.item():.4f} Loss_G: {errG.item():.4f} '
                          f'Wasserstein Distance: {wasserstein_d:.4f}')

        # Generate samples
        with torch.no_grad():
            fake = netG(fixed_noise)

        # Save samples every 5 epochs
        if epoch % 5 == 0:
            vutils.save_image(fake.detach(), f'wgan_samples_epoch_{epoch}.png',
                            normalize=True, nrow=8)

    return netG, netC

def calculate_wasserstein_distance(critic, real_data, fake_data):
    """
    Wasserstein Distance 근사 계산

    Kantorovich-Rubinstein Duality를 이용한 근사:
    W(P_r, P_g) = sup_{||f||_L≤1} E_x~P_r[f(x)] - E_x~P_g[f(x)]

    여기서 f는 1-Lipschitz 함수 (Critic이 근사)

    Args:
        critic (nn.Module): 훈련된 WGAN Critic
        real_data (torch.Tensor): 실제 데이터 샘플
        fake_data (torch.Tensor): 생성된 데이터 샘플

    Returns:
        torch.Tensor: 근사된 Wasserstein distance

    수학적 해석:
    - 양수: 실제 분포와 생성 분포 간 거리
    - 0에 가까울수록: 두 분포가 유사함
    - 음수: Critic이 잘못 학습됨 (드뭄)

    장점:
    - 분포가 겹치지 않아도 의미있는 거리 측정
    - 수렴 여부 판단 가능한 지표 제공
    """
    with torch.no_grad():
        real_score = torch.mean(critic(real_data))
        fake_score = torch.mean(critic(fake_data))
        return real_score - fake_score

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
        plt.title("WGAN Generated Images")
        plt.axis('off')
        plt.show()

def plot_wasserstein_distance_over_time(critic, generator, dataloader, nz=100):
    """Plot Wasserstein distance over training batches"""
    critic.eval()
    generator.eval()
    device = next(critic.parameters()).device

    distances = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= 50:  # Limit to first 50 batches for speed
                break

            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)

            # Generate fake data
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = generator(noise)

            # Calculate distance
            distance = calculate_wasserstein_distance(critic, real_cpu, fake)
            distances.append(distance.item())

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('Wasserstein Distance Over Batches')
    plt.xlabel('Batch')
    plt.ylabel('Wasserstein Distance')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print("WGAN Example - Wasserstein GAN")
    print("Key differences from standard GAN:")
    print("- Uses Wasserstein distance instead of JS divergence")
    print("- Critic (not discriminator) with no sigmoid")
    print("- Weight clipping for Lipschitz constraint")
    print("- RMSprop optimizer with lower learning rate")
    print("- Multiple critic updates per generator update")

    # Uncomment to train
    # generator, critic = train_wgan()
    # generate_samples(generator)

    # Demo of Wasserstein distance calculation
    print("\nWasserstein distance provides a meaningful metric even when")
    print("distributions don't overlap, unlike JS divergence which saturates.")