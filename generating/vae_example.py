"""
VAE (Variational Autoencoder) Implementation

VAE는 Kingma & Welling (2013)에서 제안된 확률적 생성 모델로, 다음과 같은 핵심 특징을 가집니다:

핵심 개념:
1. **Encoder (Recognition Network)**: q_φ(z|x)
   - 입력 데이터 x를 잠재 변수 z의 분포 파라미터 (μ, σ)로 매핑
   - 변분 추론(Variational Inference)을 통해 사후 분포 근사

2. **Decoder (Generative Network)**: p_θ(x|z)
   - 잠재 변수 z에서 데이터 x를 재구성
   - 조건부 확률 분포로 생성 과정 모델링

3. **Reparameterization Trick**: z = μ + σ ⊙ ε (ε ~ N(0,I))
   - 확률적 노드를 통한 역전파 가능하게 함
   - 그래디언트 추정의 분산을 크게 줄임

수학적 목표:
- Evidence Lower Bound (ELBO) 최대화:
  log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x)||p(z))

손실 함수:
L = Reconstruction Loss + β × KL Divergence
  = ||x - x̂||² + β × KL(q(z|x)||N(0,I))

장점:
- 연속적이고 의미있는 잠재 공간
- 안정적인 훈련 과정
- 잠재 공간에서의 부드러운 보간 가능
- 새로운 샘플 생성 및 데이터 압축 동시 가능

Reference:
- Kingma, D. P., & Welling, M. (2013).
  "Auto-encoding variational bayes."
  International Conference on Learning Representations (ICLR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu layer
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar layer

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """
        인코더: 입력 데이터를 잠재 분포의 파라미터로 변환

        Recognition Network q_φ(z|x)를 구현합니다.
        입력 이미지를 받아 잠재 변수 z의 분포 파라미터 (μ, log σ²)를 출력합니다.

        Args:
            x (torch.Tensor): 입력 이미지 [batch_size, 784]

        Returns:
            tuple: (mu, logvar)
                - mu: 잠재 분포의 평균 [batch_size, latent_dim]
                - logvar: 잠재 분포의 log 분산 [batch_size, latent_dim]

        Note:
            logvar를 사용하는 이유: 수치적 안정성 (σ > 0 보장) 및 계산 효율성
        """
        h1 = F.relu(self.fc1(x))  # 첫 번째 hidden layer
        return self.fc21(h1), self.fc22(h1)  # μ와 log σ² 분리 출력

    def reparameterize(self, mu, logvar):
        """
        Reparameterization Trick: 확률적 샘플링을 미분 가능하게 만듦

        핵심 아이디어:
        - 기존: z ~ N(μ, σ²) → 미분 불가능한 샘플링
        - 개선: z = μ + σ ⊙ ε, ε ~ N(0,I) → 미분 가능한 변환

        이를 통해 확률적 노드를 통한 역전파가 가능해집니다.

        Args:
            mu (torch.Tensor): 평균 파라미터 [batch_size, latent_dim]
            logvar (torch.Tensor): log 분산 파라미터 [batch_size, latent_dim]

        Returns:
            torch.Tensor: 샘플링된 잠재 변수 z [batch_size, latent_dim]

        수학적 정의:
            z = μ + σ ⊙ ε
            where σ = exp(0.5 * log σ²), ε ~ N(0,I)
        """
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5 * log σ²) = √(σ²)
        eps = torch.randn_like(std)    # ε ~ N(0,I), 표준정규분포에서 샘플링
        return mu + eps * std          # z = μ + σ ⊙ ε

    def decode(self, z):
        """
        디코더: 잠재 변수를 원본 데이터로 재구성

        Generative Network p_θ(x|z)를 구현합니다.
        잠재 변수 z에서 원본 이미지를 재구성합니다.

        Args:
            z (torch.Tensor): 잠재 변수 [batch_size, latent_dim]

        Returns:
            torch.Tensor: 재구성된 이미지 [batch_size, 784], 값 범위 [0, 1]

        Note:
            Sigmoid 활성화로 출력을 [0, 1] 범위로 제한 (MNIST는 이진 이미지)
        """
        h3 = F.relu(self.fc3(z))           # Hidden layer
        return torch.sigmoid(self.fc4(h3)) # 확률값 출력 [0, 1]

    def forward(self, x):
        """
        VAE의 전체 forward pass

        전체 과정: x → encode → reparameterize → decode → x_recon

        Args:
            x (torch.Tensor): 입력 이미지 [batch_size, channels, height, width]

        Returns:
            tuple: (재구성된 이미지, μ, log σ²)
                - 재구성된 이미지: [batch_size, 784]
                - μ, log σ²: 손실 계산에 필요한 분포 파라미터
        """
        mu, logvar = self.encode(x.view(-1, 784))  # 2D → 1D 변환 후 인코딩
        z = self.reparameterize(mu, logvar)        # 잠재 변수 샘플링
        return self.decode(z), mu, logvar          # 디코딩 및 파라미터 반환

def loss_function(recon_x, x, mu, logvar):
    """
    VAE 손실 함수: Reconstruction Loss + KL Divergence

    ELBO (Evidence Lower Bound)를 최대화하는 것과 동일:
    log p(x) ≥ E_q[log p(x|z)] - KL(q(z|x)||p(z))

    Args:
        recon_x (torch.Tensor): 재구성된 이미지 [batch_size, 784]
        x (torch.Tensor): 원본 이미지 [batch_size, channels, height, width]
        mu (torch.Tensor): 인코더 출력 평균 [batch_size, latent_dim]
        logvar (torch.Tensor): 인코더 출력 log 분산 [batch_size, latent_dim]

    Returns:
        torch.Tensor: 총 손실 (스칼라)

    구성 요소:
    1. Reconstruction Loss: 원본과 재구성 이미지 간의 차이
    2. KL Divergence: 사후 분포 q(z|x)와 사전 분포 p(z)=N(0,I) 간의 차이
    """
    # Reconstruction Loss: Binary Cross Entropy
    # 픽셀 단위로 재구성 품질 측정
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # KL Divergence: KL(q(z|x)||p(z)) where p(z) = N(0,I)
    # 가우시안 분포 간 KL divergence의 closed form:
    # KL = -0.5 * Σ(1 + log σ² - μ² - σ²)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD  # Total loss

def train_vae():
    # Hyperparameters
    batch_size = 128
    epochs = 10
    learning_rate = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    train_loss = 0

    for epoch in range(epochs):
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(dataloader.dataset):.4f}')
        train_loss = 0

    return model

def generate_samples(model, num_samples=16):
    model.eval()
    with torch.no_grad():
        # Sample from latent space
        sample = torch.randn(num_samples, 20)
        sample = model.decode(sample).cpu()

        # Visualize
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(sample[i].view(28, 28), cmap='gray')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

def interpolate_samples(model, num_steps=10):
    model.eval()
    with torch.no_grad():
        # Create two random points in latent space
        z1 = torch.randn(1, 20)
        z2 = torch.randn(1, 20)

        # Interpolate between them
        interpolations = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            sample = model.decode(z_interp).cpu()
            interpolations.append(sample.view(28, 28))

        # Visualize interpolation
        fig, axes = plt.subplots(1, num_steps, figsize=(15, 2))
        for i, ax in enumerate(axes):
            ax.imshow(interpolations[i], cmap='gray')
            ax.axis('off')
        plt.title('Latent Space Interpolation')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("VAE Example - Variational Autoencoder")
    print("This example demonstrates a VAE for generating MNIST-like images.")
    print("Features:")
    print("- Latent space sampling")
    print("- Image reconstruction")
    print("- Latent space interpolation")

    # Uncomment to train and generate samples
    model = train_vae()
    generate_samples(model)
    interpolate_samples(model)