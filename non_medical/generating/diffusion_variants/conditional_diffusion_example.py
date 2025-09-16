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
from ddpm_example import TimeEmbedding, ResidualBlock, AttentionBlock

class ClassEmbedding(nn.Module):
    """Embedding layer for class labels"""
    def __init__(self, num_classes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)

    def forward(self, labels):
        return self.embedding(labels)

class ConditionalUNet(nn.Module):
    """U-Net with class conditioning for diffusion models"""
    def __init__(self, in_channels=1, model_channels=128, out_channels=1,
                 num_res_blocks=2, attention_resolutions=[16],
                 channel_mult=[1, 2, 2, 2], num_classes=10,
                 time_emb_dim=512, class_emb_dim=512):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_classes = num_classes

        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Class embedding
        self.class_embed = nn.Sequential(
            ClassEmbedding(num_classes, class_emb_dim),
            nn.Linear(class_emb_dim, time_emb_dim),
        )

        # Combined embedding processing
        self.combined_embed = nn.Sequential(
            nn.Linear(time_emb_dim * 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Initial convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down sampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [model_channels]

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ConditionalResidualBlock(ch, mult * model_channels, time_emb_dim)]
                ch = mult * model_channels

                # Add attention at specified resolutions
                if 32 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)

        # Middle
        self.middle_block = nn.ModuleList([
            ConditionalResidualBlock(ch, ch, time_emb_dim),
            AttentionBlock(ch),
            ConditionalResidualBlock(ch, ch, time_emb_dim),
        ])

        # Up sampling
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ConditionalResidualBlock(ch + ich, mult * model_channels, time_emb_dim)]
                ch = mult * model_channels

                # Add attention at specified resolutions
                if 32 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                # Upsample (except last iteration of last level)
                if level != 0 and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

                self.up_blocks.append(nn.ModuleList(layers))

        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, class_labels):
        # Time embedding
        time_emb = self.time_embed(timesteps)

        # Class embedding
        class_emb = self.class_embed(class_labels)

        # Combine time and class embeddings
        combined_emb = torch.cat([time_emb, class_emb], dim=-1)
        combined_emb = self.combined_embed(combined_emb)

        # Initial conv
        x = self.input_conv(x)

        # Store skip connections
        skip_connections = [x]

        # Down sampling
        for layers in self.down_blocks:
            for layer in layers:
                if isinstance(layer, ConditionalResidualBlock):
                    x = layer(x, combined_emb)
                else:
                    x = layer(x)
            skip_connections.append(x)

        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ConditionalResidualBlock):
                x = layer(x, combined_emb)
            else:
                x = layer(x)

        # Up sampling
        for layers in self.up_blocks:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)

            for layer in layers:
                if isinstance(layer, ConditionalResidualBlock):
                    x = layer(x, combined_emb)
                else:
                    x = layer(x)

        # Output
        return self.output_conv(x)

class ConditionalResidualBlock(nn.Module):
    """Residual block with time and class conditioning"""
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.emb_proj = nn.Linear(emb_dim, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.activation = nn.SiLU()

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Add combined time/class embedding
        emb_out = self.activation(self.emb_proj(emb))
        x = x + emb_out[:, :, None, None]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        return x + residual

class ConditionalDDPM:
    """Conditional Denoising Diffusion Probabilistic Model"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02,
                 img_size=32, num_classes=10, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.num_classes = num_classes
        self.device = device

        # Precompute noise schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        """Linear noise schedule"""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """Add noise to images according to diffusion process"""
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        """Sample random timesteps"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, class_labels=None, channels=1, guidance_scale=3.0, ddim_steps=50):
        """
        Sample new images with class conditioning
        guidance_scale: Controls strength of class conditioning
        """
        print(f"Sampling {n} images with class conditioning...")
        model.eval()

        if class_labels is None:
            # Sample random classes
            class_labels = torch.randint(0, self.num_classes, (n,)).to(self.device)
        else:
            class_labels = class_labels.to(self.device)

        with torch.no_grad():
            # Start from random noise
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)

            # DDIM sampling with classifier-free guidance
            skip = self.noise_steps // ddim_steps
            seq = range(0, self.noise_steps, skip)
            seq = list(reversed(seq))

            for i, j in enumerate(tqdm(seq[:-1])):
                t = torch.full((n,), j, dtype=torch.long).to(self.device)
                next_t = torch.full((n,), seq[i + 1], dtype=torch.long).to(self.device)

                # Conditional prediction
                predicted_noise_cond = model(x, t, class_labels)

                # Unconditional prediction (for classifier-free guidance)
                # Use a special "null" class or random class for unconditional
                null_labels = torch.full_like(class_labels, self.num_classes)  # Assuming num_classes is null class
                predicted_noise_uncond = model(x, t, null_labels)

                # Classifier-free guidance
                predicted_noise = predicted_noise_uncond + guidance_scale * (predicted_noise_cond - predicted_noise_uncond)

                # DDIM step
                alpha_t = self.alpha_hat[t][:, None, None, None]
                alpha_t_next = self.alpha_hat[next_t][:, None, None, None]

                pred_x0 = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                direction = torch.sqrt(1 - alpha_t_next) * predicted_noise
                x = torch.sqrt(alpha_t_next) * pred_x0 + direction

        model.train()

        # Denormalize
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x, class_labels

    def sample_specific_class(self, model, class_id, n=16, guidance_scale=3.0, ddim_steps=50):
        """Sample images of a specific class"""
        class_labels = torch.full((n,), class_id, dtype=torch.long)
        images, _ = self.sample(model, n, class_labels, guidance_scale=guidance_scale, ddim_steps=ddim_steps)
        return images

    def interpolate_classes(self, model, class1, class2, steps=10, guidance_scale=3.0, ddim_steps=50):
        """Interpolate between two classes"""
        model.eval()

        # Fixed noise for consistency
        x = torch.randn((1, 1, self.img_size, self.img_size)).to(self.device)

        interpolated_images = []

        for i in range(steps):
            # Simple interpolation: alternate between classes
            alpha = i / (steps - 1)
            if alpha < 0.5:
                class_label = torch.tensor([class1], dtype=torch.long)
            else:
                class_label = torch.tensor([class2], dtype=torch.long)

            # Sample with fixed noise and interpolated class
            images, _ = self.sample(model, 1, class_label, guidance_scale=guidance_scale, ddim_steps=ddim_steps)
            interpolated_images.append(images[0])

        return torch.stack(interpolated_images)

def train_conditional_ddpm():
    """Train Conditional DDPM"""
    # Hyperparameters
    batch_size = 16
    epochs = 100
    lr = 2e-4
    img_size = 32
    noise_steps = 1000
    num_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    transforms_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize to [-1, 1]
    ])

    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms_pipeline)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model and diffusion
    model = ConditionalUNet(num_classes=num_classes + 1).to(device)  # +1 for null class
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = ConditionalDDPM(noise_steps=noise_steps, img_size=img_size,
                               num_classes=num_classes, device=device)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training loop
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        epoch_loss = 0

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            # Classifier-free guidance training
            # Randomly replace some labels with null class (for unconditional training)
            mask = torch.rand(labels.shape[0]) < 0.1  # 10% unconditional
            labels[mask] = num_classes  # null class

            # Sample random timesteps
            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            # Add noise to images
            x_t, noise = diffusion.noise_images(images, t)

            # Predict noise
            predicted_noise = model(x_t, t, labels)

            # Calculate loss
            loss = mse(noise, predicted_noise)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())

        print(f"Epoch {epoch} | Average Loss: {epoch_loss/len(dataloader):.4f}")

        # Sample images every 10 epochs
        if epoch % 10 == 0 and epoch > 0:
            # Sample one image from each class
            all_samples = []
            for class_id in range(num_classes):
                samples = diffusion.sample_specific_class(model, class_id, n=1, ddim_steps=20)
                all_samples.append(samples[0])

            # Save class-specific samples
            save_class_grid(torch.stack(all_samples), f"conditional_diffusion_epoch_{epoch}.png")

    return model, diffusion

def save_class_grid(images, path):
    """Save grid of images by class"""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))

    for i in range(10):
        row, col = i // 5, i % 5
        axes[row, col].imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
        axes[row, col].set_title(f'Class {i}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()

def demonstrate_class_conditioning(model, diffusion):
    """Demonstrate class-specific generation"""
    print("Class-specific generation:")

    for class_id in range(10):
        print(f"Generating digit {class_id}...")
        samples = diffusion.sample_specific_class(model, class_id, n=4, ddim_steps=20)

        # Display samples (in practice, you'd save/show them)
        print(f"  Generated {len(samples)} samples of digit {class_id}")

def demonstrate_guidance_scale_effect():
    """Demonstrate effect of guidance scale"""
    print("\nGuidance Scale Effects:")
    print("-" * 30)
    print("guidance_scale = 1.0: No guidance (unconditional-like)")
    print("guidance_scale = 3.0: Moderate guidance (balanced)")
    print("guidance_scale = 7.0: Strong guidance (more class-specific)")
    print("guidance_scale = 15.0: Very strong guidance (may reduce diversity)")

    print("\nHigher guidance scale:")
    print("+ More class-specific features")
    print("+ Better class adherence")
    print("- Reduced sample diversity")
    print("- Potential artifacts")

def compare_conditional_vs_unconditional():
    """Compare conditional vs unconditional diffusion"""
    print("\nConditional vs Unconditional Diffusion:")
    print("=" * 50)

    print("Unconditional Diffusion:")
    print("- Random generation from learned distribution")
    print("- No control over output class/attributes")
    print("- Simple training procedure")
    print("- Good for modeling overall data distribution")

    print("\nConditional Diffusion:")
    print("- Controlled generation based on class labels")
    print("- Can generate specific classes on demand")
    print("- Classifier-free guidance for better quality")
    print("- Enables interpolation between classes")
    print("- More complex training (needs labels)")

if __name__ == "__main__":
    print("Conditional Diffusion Example")
    print("Key features:")
    print("- Class-conditioned image generation")
    print("- Classifier-free guidance for better quality")
    print("- Can generate specific digits on demand")
    print("- Interpolation between classes")
    print("- Combined time and class embeddings")

    # Demonstrate concepts
    compare_conditional_vs_unconditional()
    demonstrate_guidance_scale_effect()

    # Initialize model
    conditional_ddpm = ConditionalDDPM()
    print(f"\nConditional DDPM with {conditional_ddpm.num_classes} classes")

    # Uncomment to train (computationally intensive!)
    # model, diffusion = train_conditional_ddpm()
    #
    # # Generate specific digits
    # for digit in range(10):
    #     samples = diffusion.sample_specific_class(model, digit, n=4)
    #     print(f"Generated {len(samples)} images of digit {digit}")
    #
    # # Demonstrate class conditioning
    # demonstrate_class_conditioning(model, diffusion)

    print("\nConditional Diffusion enables precise control over generation")