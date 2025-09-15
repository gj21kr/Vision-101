"""
GAN (Generative Adversarial Networks) Implementation

GAN은 Goodfellow et al. (2014)에서 제안된 생성 모델로, 두 개의 신경망이 적대적으로
경쟁하며 학습하는 혁신적인 방법론입니다.

핵심 구조:
1. **Generator (G)**:
   - 잠재 벡터 z (보통 가우시안 노이즈)를 입력으로 받음
   - 실제 데이터와 유사한 가짜 데이터 생성
   - 목표: Discriminator를 속이는 것

2. **Discriminator (D)**:
   - 입력 데이터가 실제인지 가짜인지 분류
   - 이진 분류기 (Real: 1, Fake: 0)
   - 목표: 실제와 가짜를 정확히 구분하는 것

수학적 목표 (Minimax Game):
min_G max_D V(D,G) = E_x~p_data(x)[log D(x)] + E_z~p_z(z)[log(1 - D(G(z)))]

훈련 과정:
1. Discriminator 업데이트: 실제 데이터는 1로, 가짜 데이터는 0으로 분류하도록 학습
2. Generator 업데이트: Discriminator가 가짜 데이터를 1로 분류하도록 속이는 방향으로 학습

주요 특징:
- 명시적 확률 분포 모델링 불필요
- 고품질 샘플 생성 가능
- 훈련 불안정성 (mode collapse, gradient vanishing 등)
- Nash Equilibrium 달성이 목표

이 구현은 기본적인 fully-connected GAN입니다.
더 고급 변형 (DCGAN, WGAN 등)은 gan_variants/ 디렉토리를 참조하세요.

Reference:
- Goodfellow, I., et al. (2014).
  "Generative adversarial nets."
  Neural Information Processing Systems (NIPS).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_dim=100, img_dim=784):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Generator forward pass: 노이즈 벡터를 이미지로 변환

        기본 GAN Generator의 구조:
        - Fully connected layers로 구성
        - LeakyReLU 활성화로 gradient flow 개선
        - Tanh 출력으로 [-1, 1] 범위 정규화

        Args:
            x (torch.Tensor): 입력 노이즈 벡터 [batch_size, noise_dim]

        Returns:
            torch.Tensor: 생성된 이미지 [batch_size, img_dim], 범위 [-1, 1]

        Note:
            Tanh 활성화를 사용하므로 입력 데이터도 [-1, 1]로 정규화 필요
        """
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Discriminator forward pass: 이미지의 진위 판별

        기본 GAN Discriminator의 구조:
        - Fully connected layers로 구성
        - LeakyReLU 활성화와 Dropout으로 overfitting 방지
        - Sigmoid 출력으로 확률값 [0, 1] 반환

        Args:
            x (torch.Tensor): 입력 이미지 [batch_size, img_dim]

        Returns:
            torch.Tensor: 진짜일 확률 [batch_size, 1], 범위 [0, 1]
                         1에 가까울수록 진짜, 0에 가까울수록 가짜

        수학적 목표:
            D(x_real) → 1 (실제 이미지에 대해 높은 확률)
            D(G(z)) → 0 (생성 이미지에 대해 낮은 확률)
        """
        return self.disc(x)

def train_gan():
    # Hyperparameters
    lr = 3e-4
    noise_dim = 64
    img_dim = 28 * 28 * 1
    batch_size = 32
    num_epochs = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize models
    gen = Generator(noise_dim, img_dim).to(device)
    disc = Discriminator(img_dim).to(device)

    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)

    # Loss function
    criterion = nn.BCELoss()

    # Data loading
    transforms_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST(root="dataset/", transform=transforms_mnist, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]

            # Train Discriminator
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            lossD = (lossD_real + lossD_fake) / 2
            disc.zero_grad()
            lossD.backward(retain_graph=True)
            opt_disc.step()

            # Train Generator
            output = disc(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            lossG.backward()
            opt_gen.step()
    
            print(f"Epoch [{epoch}/{num_epochs}] Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

def generate_samples(generator, num_samples=16):
	generator.eval()
	with torch.no_grad():
		noise = torch.randn(num_samples, 64)
		fake_images = generator(noise)
		fake_images = fake_images.view(-1, 1, 28, 28)

		# Visualize
		fig, axes = plt.subplots(4, 4, figsize=(8, 8))
		for i, ax in enumerate(axes.flatten()):
			ax.imshow(fake_images[i].squeeze(), cmap='gray')
			ax.axis('off')
		plt.tight_layout()
		plt.show()

if __name__ == "__main__":
	# train_gan()
	print("GAN Example - Generative Adversarial Networks")
	print("This example demonstrates a simple GAN for generating MNIST-like images.")
	print("Uncomment train_gan() to start training.")
	