import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, embed_dim=100, img_channels=1):
        super(ConditionalGenerator, self).__init__()

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Input dimension is noise + embedded label
        input_dim = noise_dim + embed_dim

        self.gen = nn.Sequential(
            # First layer
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),

            # Second layer
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),

            # Third layer
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),

            # Output layer
            nn.Linear(1024, img_channels * 28 * 28),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels
        embedded_labels = self.label_embedding(labels)

        # Concatenate noise and embedded labels
        gen_input = torch.cat((noise, embedded_labels), dim=1)

        # Generate image
        img = self.gen(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img

class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10, embed_dim=100, img_channels=1):
        super(ConditionalDiscriminator, self).__init__()

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Image processing
        self.img_dim = img_channels * 28 * 28

        self.disc = nn.Sequential(
            # First layer (image + embedded label)
            nn.Linear(self.img_dim + embed_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Second layer
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Third layer
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Output layer
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Flatten image
        img_flat = img.view(img.size(0), -1)

        # Embed labels
        embedded_labels = self.label_embedding(labels)

        # Concatenate image and embedded labels
        disc_input = torch.cat((img_flat, embedded_labels), dim=1)

        # Discriminate
        validity = self.disc(disc_input)
        return validity

class ConvConditionalGenerator(nn.Module):
    """Convolutional version of conditional generator"""
    def __init__(self, noise_dim=100, num_classes=10, img_channels=1):
        super(ConvConditionalGenerator, self).__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Initial linear layer
        self.fc = nn.Linear(noise_dim + num_classes, 256 * 7 * 7)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # 7x7 -> 14x14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 14x14 -> 28x28
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Output layer
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels
        embedded_labels = self.label_embedding(labels)

        # Concatenate noise and labels
        gen_input = torch.cat((noise, embedded_labels), dim=1)

        # Fully connected
        x = self.fc(gen_input)
        x = x.view(x.size(0), 256, 7, 7)

        # Convolutional layers
        img = self.conv_layers(x)
        return img

class ConvConditionalDiscriminator(nn.Module):
    """Convolutional version of conditional discriminator"""
    def __init__(self, num_classes=10, img_channels=1):
        super(ConvConditionalDiscriminator, self).__init__()

        # Label embedding (project to image space)
        self.label_embedding = nn.Embedding(num_classes, 28 * 28)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # 28x28 -> 14x14 (input has img_channels + 1 for label)
            nn.Conv2d(img_channels + 1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 14x14 -> 7x7
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 7x7 -> 1x1
            nn.Conv2d(128, 1, 7, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Embed and reshape labels to image dimensions
        embedded_labels = self.label_embedding(labels)
        embedded_labels = embedded_labels.view(labels.size(0), 1, 28, 28)

        # Concatenate image and label
        disc_input = torch.cat((img, embedded_labels), dim=1)

        # Apply convolutional layers
        validity = self.conv_layers(disc_input)
        return validity.view(-1, 1).squeeze(1)

def train_conditional_gan():
    # Hyperparameters
    batch_size = 64
    noise_dim = 100
    num_classes = 10
    num_epochs = 50
    lr = 0.0002
    beta1 = 0.5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    generator = ConvConditionalGenerator(noise_dim, num_classes).to(device)
    discriminator = ConvConditionalDiscriminator(num_classes).to(device)

    # Loss function
    criterion = nn.BCELoss()

    # Optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Fixed noise and labels for visualization
    fixed_noise = torch.randn(100, noise_dim, device=device)
    fixed_labels = torch.arange(0, 10, device=device).repeat(10)

    # Training loop
    for epoch in range(num_epochs):
        for i, (real_imgs, real_labels) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = real_labels.to(device)

            # Real and fake labels
            valid = torch.ones(batch_size, device=device, dtype=torch.float)
            fake = torch.zeros(batch_size, device=device, dtype=torch.float)

            # Train Discriminator
            opt_disc.zero_grad()

            # Real images
            real_pred = discriminator(real_imgs, real_labels)
            real_loss = criterion(real_pred, valid)

            # Fake images
            noise = torch.randn(batch_size, noise_dim, device=device)
            fake_labels = torch.randint(0, num_classes, (batch_size,), device=device)
            fake_imgs = generator(noise, fake_labels)
            fake_pred = discriminator(fake_imgs.detach(), fake_labels)
            fake_loss = criterion(fake_pred, fake)

            disc_loss = (real_loss + fake_loss) / 2
            disc_loss.backward()
            opt_disc.step()

            # Train Generator
            opt_gen.zero_grad()

            fake_pred = discriminator(fake_imgs, fake_labels)
            gen_loss = criterion(fake_pred, valid)
            gen_loss.backward()
            opt_gen.step()

            # Print statistics
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {disc_loss.item():.4f} Loss_G: {gen_loss.item():.4f}')

        # Generate samples with fixed noise and labels
        if epoch % 5 == 0:
            with torch.no_grad():
                fake_samples = generator(fixed_noise, fixed_labels)
                vutils.save_image(fake_samples[:100], f'conditional_gan_samples_epoch_{epoch}.png',
                                nrow=10, normalize=True)

    return generator, discriminator

def generate_specific_digits(generator, device, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], num_samples_per_digit=10):
    """Generate specific digits using conditional GAN"""
    generator.eval()

    with torch.no_grad():
        all_samples = []

        for digit in digits:
            # Generate noise
            noise = torch.randn(num_samples_per_digit, 100, device=device)
            # Create labels for this digit
            labels = torch.full((num_samples_per_digit,), digit, device=device, dtype=torch.long)

            # Generate samples
            samples = generator(noise, labels)
            all_samples.append(samples)

        # Concatenate all samples
        all_samples = torch.cat(all_samples, dim=0)

        # Create grid
        grid = vutils.make_grid(all_samples, nrow=num_samples_per_digit, normalize=True)

        plt.figure(figsize=(15, 12))
        plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
        plt.title('Conditional GAN: Generated Digits 0-9')
        plt.axis('off')
        plt.show()

def interpolate_between_classes(generator, device, class1=3, class2=8, steps=10):
    """Interpolate between two different classes"""
    generator.eval()

    with torch.no_grad():
        # Fixed noise
        noise = torch.randn(1, 100, device=device)

        # Create interpolated labels using one-hot encoding and interpolation
        interpolated_samples = []

        for i in range(steps):
            alpha = i / (steps - 1)

            # Create mixed label representation (simplified)
            # In practice, you might interpolate in embedding space
            if alpha < 0.5:
                label = torch.tensor([class1], device=device, dtype=torch.long)
            else:
                label = torch.tensor([class2], device=device, dtype=torch.long)

            sample = generator(noise, label)
            interpolated_samples.append(sample)

        # Show transition
        interpolated_tensor = torch.cat(interpolated_samples, dim=0)
        grid = vutils.make_grid(interpolated_tensor, nrow=steps, normalize=True)

        plt.figure(figsize=(15, 3))
        plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)))
        plt.title(f'Class Interpolation: {class1} → {class2}')
        plt.axis('off')
        plt.show()

def demonstrate_label_conditioning():
    """Demonstrate the concept of conditional generation"""
    print("Conditional GAN allows controlled generation based on class labels")
    print("Generator: G(z, y) where z is noise and y is class label")
    print("Discriminator: D(x, y) evaluates real/fake given image x and label y")
    print()
    print("Loss functions:")
    print("L_D = E[log D(x,y)] + E[log(1 - D(G(z,y),y))]")
    print("L_G = E[log D(G(z,y),y)]")
    print()
    print("Applications:")
    print("- Generate specific digits (MNIST)")
    print("- Text-to-image synthesis")
    print("- Class-conditional image synthesis")
    print("- Controllable face generation")

def compare_conditional_vs_unconditional():
    """Compare conditional vs unconditional generation"""
    print("\nConditional vs Unconditional GANs:")
    print()
    print("Unconditional GAN:")
    print("- Generator: G(z) → x")
    print("- No control over output class/attributes")
    print("- Generates random samples from learned distribution")
    print()
    print("Conditional GAN:")
    print("- Generator: G(z, y) → x")
    print("- Full control over output class/attributes")
    print("- Can generate specific types of samples on demand")
    print("- Enables applications like text-to-image, class-specific generation")

if __name__ == "__main__":
    print("Conditional GAN Example - Class-Conditional Image Generation")
    print("Key features:")
    print("- Generator conditioned on class labels")
    print("- Discriminator receives both image and label")
    print("- Enables controlled generation of specific classes")
    print("- Both FC and Convolutional versions implemented")

    demonstrate_label_conditioning()
    compare_conditional_vs_unconditional()

    # Uncomment to train
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # generator, discriminator = train_conditional_gan()
    # generate_specific_digits(generator, device)
    # interpolate_between_classes(generator, device, class1=3, class2=8)

    print("\nConditional GANs enable precise control over generation")
    print("by incorporating additional information (labels, text, etc.)")