import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm

# Import components from previous examples
from ddpm_example import TimeEmbedding

class ScoreNetwork(nn.Module):
    """Score network for score-based diffusion"""
    def __init__(self, in_channels=1, model_channels=128, out_channels=1,
                 time_emb_dim=512, num_res_blocks=4):
        super().__init__()

        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        channels = [model_channels, model_channels * 2, model_channels * 4, model_channels * 8]

        for i in range(len(channels) - 1):
            self.encoder_blocks.append(
                ScoreResBlock(channels[i], channels[i + 1], time_emb_dim, downsample=True)
            )

        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ScoreResBlock(channels[-1], channels[-1], time_emb_dim),
            ScoreResBlock(channels[-1], channels[-1], time_emb_dim),
        ])

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoder_blocks.append(
                ScoreResBlock(channels[i] * 2, channels[i - 1], time_emb_dim, upsample=True)
            )

        # Output convolution
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t)

        # Initial convolution
        x = self.conv_in(x)

        # Store skip connections
        skip_connections = []

        # Encoder
        for block in self.encoder_blocks:
            skip_connections.append(x)
            x = block(x, t_emb)

        # Middle
        for block in self.middle_blocks:
            x = block(x, t_emb)

        # Decoder
        for block in self.decoder_blocks:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)

        # Output
        return self.conv_out(x)

class ScoreResBlock(nn.Module):
    """Residual block for score network"""
    def __init__(self, in_channels, out_channels, time_emb_dim,
                 downsample=False, upsample=False):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample

        # Time projection
        self.time_proj = nn.Linear(time_emb_dim, out_channels)

        # Convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Normalization
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # Activation
        self.act = nn.SiLU()

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

        # Up/down sampling
        if downsample:
            self.downsample_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        residual = x

        # First convolution
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        # Add time embedding
        t_proj = self.act(self.time_proj(t_emb))
        x = x + t_proj[:, :, None, None]

        # Second convolution
        x = self.conv2(x)
        x = self.norm2(x)

        # Residual connection
        residual = self.residual_conv(residual)
        x = x + residual
        x = self.act(x)

        # Up/down sampling
        if self.downsample:
            x = self.downsample_conv(x)
        elif self.upsample:
            x = self.upsample_conv(x)

        return x

class ScoreBasedDiffusion:
    """Score-based diffusion model using SDE framework"""
    def __init__(self, sigma_min=0.01, sigma_max=50.0, N=1000, device="cuda"):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N  # Number of discretization steps
        self.device = device

        # Geometric noise schedule
        self.sigmas = self.get_sigmas().to(device)

    def get_sigmas(self):
        """Get geometric noise schedule"""
        return torch.exp(torch.linspace(
            np.log(self.sigma_max), np.log(self.sigma_min), self.N
        ))

    def marginal_prob_std(self, t):
        """Standard deviation of the perturbation kernel"""
        return torch.sqrt((self.sigma_max ** (2 * t) - self.sigma_min ** (2 * t)) * 2 * np.log(self.sigma_max / self.sigma_min) + self.sigma_min ** (2 * t))

    def diffusion_coeff(self, t):
        """Diffusion coefficient of the SDE"""
        return torch.tensor(self.sigma_min * (self.sigma_max / self.sigma_min) ** t * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min)), device=self.device)

    def perturb_data(self, x, t):
        """Perturb data according to the SDE"""
        std = self.marginal_prob_std(t)
        z = torch.randn_like(x)
        perturbed_x = x + std[:, None, None, None] * z
        return perturbed_x, z

    def sample_time(self, batch_size):
        """Sample random time points"""
        return torch.rand(batch_size, device=self.device)

    def euler_maruyama_sample(self, score_model, shape, num_steps=500):
        """Euler-Maruyama sampler for the reverse SDE"""
        print(f"Sampling with Euler-Maruyama ({num_steps} steps)...")
        score_model.eval()

        with torch.no_grad():
            # Start from random noise
            x = torch.randn(shape, device=self.device) * self.sigma_max

            time_steps = torch.linspace(1.0, 1e-3, num_steps)
            step_size = time_steps[0] - time_steps[1]

            for time_step in tqdm(time_steps):
                batch_time_step = torch.ones(shape[0], device=self.device) * time_step

                # Predict score
                score = score_model(x, batch_time_step)

                # Drift term
                drift = -0.5 * (self.diffusion_coeff(time_step) ** 2) * score

                # Diffusion term
                diffusion = self.diffusion_coeff(time_step)
                noise = torch.randn_like(x)

                # Euler-Maruyama update
                x = x + drift * step_size + diffusion * np.sqrt(step_size) * noise

        score_model.train()
        return x

    def pc_sampler(self, score_model, shape, num_steps=500, snr=0.16):
        """Predictor-Corrector sampler"""
        print(f"Sampling with PC sampler ({num_steps} steps)...")
        score_model.eval()

        with torch.no_grad():
            # Start from random noise
            x = torch.randn(shape, device=self.device) * self.sigma_max

            time_steps = torch.linspace(1.0, 1e-3, num_steps)
            step_size = time_steps[0] - time_steps[1]

            for time_step in tqdm(time_steps):
                batch_time_step = torch.ones(shape[0], device=self.device) * time_step

                # Predictor step (Euler-Maruyama)
                score = score_model(x, batch_time_step)
                drift = -0.5 * (self.diffusion_coeff(time_step) ** 2) * score
                diffusion = self.diffusion_coeff(time_step)
                noise = torch.randn_like(x)
                x = x + drift * step_size + diffusion * np.sqrt(step_size) * noise

                # Corrector step (Langevin MCMC)
                for _ in range(5):  # 5 corrector steps
                    score = score_model(x, batch_time_step)
                    noise = torch.randn_like(x)
                    grad_norm = torch.norm(score.view(score.shape[0], -1), dim=-1).mean()
                    noise_norm = np.sqrt(np.prod(x.shape[1:]))
                    langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
                    x = x + langevin_step_size * score + torch.sqrt(2 * langevin_step_size) * noise

        score_model.train()
        return x

    def ode_sample(self, score_model, shape, num_steps=500, rtol=1e-5, atol=1e-5):
        """ODE-based sampling (deterministic)"""
        from scipy.integrate import solve_ivp

        print(f"Sampling with ODE solver ({num_steps} steps)...")
        score_model.eval()

        def ode_func(t, x_flat):
            x = torch.tensor(x_flat.reshape(shape), dtype=torch.float32, device=self.device)
            t_tensor = torch.ones(shape[0], device=self.device) * t

            with torch.no_grad():
                score = score_model(x, t_tensor)

            drift = -0.5 * (self.diffusion_coeff(torch.tensor(t, device=self.device)) ** 2) * score
            return drift.cpu().numpy().flatten()

        # Initial condition
        x0 = torch.randn(shape, device=self.device) * self.sigma_max
        x0_flat = x0.cpu().numpy().flatten()

        # Solve ODE
        solution = solve_ivp(
            ode_func, [1.0, 1e-3], x0_flat,
            rtol=rtol, atol=atol, method='RK45'
        )

        # Convert back to tensor
        x_final = torch.tensor(
            solution.y[:, -1].reshape(shape),
            dtype=torch.float32, device=self.device
        )

        score_model.train()
        return x_final

def train_score_based_model():
    """Train score-based diffusion model"""
    # Hyperparameters
    batch_size = 32
    epochs = 200
    lr = 2e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    transforms_pipeline = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [-1, 1]
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms_pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and diffusion
    score_model = ScoreNetwork().to(device)
    optimizer = optim.Adam(score_model.parameters(), lr=lr)
    diffusion = ScoreBasedDiffusion(device=device)

    print(f"Score model has {sum(p.numel() for p in score_model.parameters()):,} parameters")

    # Training loop
    score_model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0

        for images, _ in pbar:
            images = images.to(device)
            batch_size = images.shape[0]

            # Sample random time points
            t = diffusion.sample_time(batch_size)

            # Perturb data
            perturbed_x, noise = diffusion.perturb_data(images, t)

            # Predict score (negative of noise direction)
            score_pred = score_model(perturbed_x, t)

            # Score matching loss (Fisher divergence)
            # The optimal score is -noise / std^2, but we can use simplified loss
            std = diffusion.marginal_prob_std(t)
            target_score = -noise / (std[:, None, None, None] ** 2)

            # Denoising score matching loss
            loss = torch.mean(torch.sum((score_pred - target_score) ** 2, dim=(1, 2, 3)))

            # Alternative: simple denoising loss (works well in practice)
            # loss = torch.mean(torch.sum((score_pred * std[:, None, None, None] + noise) ** 2, dim=(1, 2, 3)))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(Loss=loss.item())

        print(f"Epoch {epoch} | Average Loss: {epoch_loss/len(dataloader):.4f}")

        # Sample images every 25 epochs
        if epoch % 25 == 0 and epoch > 0:
            samples = diffusion.euler_maruyama_sample(score_model, (16, 1, 32, 32), num_steps=200)
            save_samples(samples, f"score_based_samples_epoch_{epoch}.png")

    return score_model, diffusion

def save_samples(samples, filename):
    """Save generated samples"""
    samples = (samples + 1) / 2  # Denormalize to [0, 1]
    samples = torch.clamp(samples, 0, 1)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i].squeeze().cpu().numpy(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

def compare_samplers(score_model, diffusion):
    """Compare different sampling methods"""
    print("Comparing sampling methods:")
    shape = (4, 1, 32, 32)

    methods = {
        "Euler-Maruyama": lambda: diffusion.euler_maruyama_sample(score_model, shape, num_steps=200),
        "PC Sampler": lambda: diffusion.pc_sampler(score_model, shape, num_steps=100),
        # "ODE Sampler": lambda: diffusion.ode_sample(score_model, shape, num_steps=100),
    }

    for name, sampler in methods.items():
        print(f"\n{name} sampling:")
        try:
            samples = sampler()
            print(f"  Successfully generated {samples.shape[0]} samples")
        except Exception as e:
            print(f"  Error: {e}")

def demonstrate_sde_framework():
    """Explain the SDE framework for score-based models"""
    print("Score-Based Diffusion SDE Framework:")
    print("=" * 50)
    print("Forward SDE:")
    print("  dx = f(x,t)dt + g(t)dw")
    print("  where f(x,t) = 0, g(t) = σ(t)")
    print("  So: dx = σ(t)dw (pure diffusion)")
    print()
    print("Reverse SDE:")
    print("  dx = [f(x,t) - g(t)²∇_x log p_t(x)]dt + g(t)dw̄")
    print("  dx = -σ(t)²∇_x log p_t(x)dt + σ(t)dw̄")
    print()
    print("Score function: s_θ(x,t) ≈ ∇_x log p_t(x)")
    print("Training objective: ||s_θ(x,t) - ∇_x log p_t(x)||²")
    print()
    print("Key advantages:")
    print("- Unified framework for different noise schedules")
    print("- Multiple sampling algorithms (SDE, ODE, PC)")
    print("- Better theoretical understanding")
    print("- Flexible noise schedules")

def visualize_noise_schedule():
    """Visualize the geometric noise schedule"""
    diffusion = ScoreBasedDiffusion()

    t_values = torch.linspace(0, 1, 1000)
    sigma_values = diffusion.sigma_min * (diffusion.sigma_max / diffusion.sigma_min) ** t_values

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(t_values, sigma_values)
    plt.xlabel('Time t')
    plt.ylabel('σ(t)')
    plt.title('Noise Schedule σ(t)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.loglog(t_values, sigma_values)
    plt.xlabel('Time t')
    plt.ylabel('σ(t)')
    plt.title('Noise Schedule σ(t) (log scale)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Score-Based Diffusion Example")
    print("Key concepts:")
    print("- Score matching with denoising")
    print("- SDE framework for diffusion processes")
    print("- Multiple sampling algorithms")
    print("- Continuous-time formulation")
    print("- Predictor-Corrector sampling")

    # Demonstrate concepts
    demonstrate_sde_framework()

    # Initialize diffusion
    diffusion = ScoreBasedDiffusion()
    print(f"\nScore-based diffusion with σ_min={diffusion.sigma_min}, σ_max={diffusion.sigma_max}")
    print(f"Using geometric noise schedule with {diffusion.N} discretization steps")

    # Visualize noise schedule
    # visualize_noise_schedule()

    # Uncomment to train (computationally intensive!)
    # score_model, diffusion = train_score_based_model()
    #
    # # Compare different sampling methods
    # compare_samplers(score_model, diffusion)

    print("\nScore-based models provide a principled SDE framework for diffusion")