import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class WGANGPGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
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
    def __init__(self, nc=3, ndf=64):
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
            # No activation function
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)

    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    critic_interpolates = critic(interpolates)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(critic_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty

def train_wgan_gp():
    # Hyperparameters
    batch_size = 64
    image_size = 64
    nc = 3  # Number of channels
    nz = 100  # Size of latent vector
    ngf = 64  # Size of feature maps in generator
    ndf = 64  # Size of feature maps in critic
    num_epochs = 25
    lr = 0.0001  # Learning rate
    beta1 = 0.0  # Beta1 for Adam optimizer
    beta2 = 0.9  # Beta2 for Adam optimizer
    n_critic = 5  # Number of critic updates per generator update
    lambda_gp = 10  # Gradient penalty coefficient

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
    netG = WGANGPGenerator(nz, ngf, nc).to(device)
    netG.apply(weights_init)

    netC = WGANGPCritic(nc, ndf).to(device)
    netC.apply(weights_init)

    # Optimizers (Adam is suitable for WGAN-GP)
    optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Training loop
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # Update Critic
            for _ in range(n_critic):
                netC.zero_grad()

                # Train critic with real data
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)

                # Forward pass real batch through C
                output_real = netC(real_cpu)
                errC_real = -torch.mean(output_real)

                # Train critic with fake data
                noise = torch.randn(b_size, nz, 1, 1, device=device)
                fake = netG(noise)

                # Forward pass fake batch through C
                output_fake = netC(fake.detach())
                errC_fake = torch.mean(output_fake)

                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(netC, real_cpu, fake, device)

                # Total critic loss
                errC = errC_real + errC_fake + lambda_gp * gradient_penalty
                errC.backward()
                optimizerC.step()

            # Update Generator
            netG.zero_grad()

            # Generate fake data
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = netG(noise)

            # Forward pass through critic
            output = netC(fake)
            # Calculate generator loss
            errG = -torch.mean(output)
            errG.backward()
            optimizerG.step()

            # Calculate Wasserstein distance estimate
            wasserstein_d = -errC_real.item() - errC_fake.item()

            # Print statistics
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_C: {errC.item():.4f} Loss_G: {errG.item():.4f} '
                      f'GP: {gradient_penalty.item():.4f} '
                      f'Wasserstein Distance: {wasserstein_d:.4f}')

        # Generate samples
        with torch.no_grad():
            fake = netG(fixed_noise)

        # Save samples every 5 epochs
        if epoch % 5 == 0:
            vutils.save_image(fake.detach(), f'wgan_gp_samples_epoch_{epoch}.png',
                            normalize=True, nrow=8)

    return netG, netC

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
        plt.title("WGAN-GP Generated Images")
        plt.axis('off')
        plt.show()

def analyze_gradient_penalty_effect(critic, real_data, fake_data, device):
    """Analyze the effect of gradient penalty on gradient norms"""
    batch_size = real_data.size(0)

    gradient_norms = []
    alphas = torch.linspace(0, 1, 100)

    for alpha in alphas:
        alpha_tensor = torch.full((batch_size, 1, 1, 1), alpha, device=device)
        interpolates = (alpha_tensor * real_data + (1 - alpha_tensor) * fake_data).requires_grad_(True)

        critic_interpolates = critic(interpolates)

        gradients = autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(critic_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1).mean()
        gradient_norms.append(gradient_norm.item())

    plt.figure(figsize=(10, 6))
    plt.plot(alphas, gradient_norms)
    plt.axhline(y=1, color='r', linestyle='--', label='Target norm = 1')
    plt.title('Gradient Norms Along Interpolation Path')
    plt.xlabel('Interpolation Parameter α')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_gradient_penalties():
    """Demonstrate different gradient penalty formulations"""
    print("WGAN-GP uses gradient penalty to enforce 1-Lipschitz constraint")
    print("Original penalty: E[(||∇_x̂ D(x̂)||₂ - 1)²]")
    print("Where x̂ = εx + (1-ε)G(z) with ε ~ U(0,1)")
    print()
    print("Benefits over weight clipping:")
    print("- No restriction on weight values")
    print("- Better gradient flow")
    print("- More stable training")
    print("- Higher quality samples")

if __name__ == "__main__":
    print("WGAN-GP Example - Wasserstein GAN with Gradient Penalty")
    print("Key improvements over WGAN:")
    print("- Gradient penalty instead of weight clipping")
    print("- Better training stability")
    print("- Higher quality generated samples")
    print("- Adam optimizer can be used")
    print("- No vanishing gradient problems")

    # Demonstrate gradient penalty concept
    compare_gradient_penalties()

    # Uncomment to train
    # generator, critic = train_wgan_gp()
    # generate_samples(generator)

    print("\nThe gradient penalty ensures the critic is 1-Lipschitz")
    print("by penalizing gradients that deviate from norm 1.")