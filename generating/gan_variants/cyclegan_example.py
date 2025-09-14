import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import itertools

class ResidualBlock(nn.Module):
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
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(CycleGANGenerator, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class CycleGANDiscriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(CycleGANDiscriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [
            nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        model += [
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        # Average pooling and flatten
        return torch.mean(x, dim=[2, 3])

class ImagePool:
    """Buffer to store previously generated images"""
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = torch.rand(1)
                if p > 0.5:
                    random_id = torch.randint(0, self.pool_size, (1,)).item()
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class DummyDataset(Dataset):
    """Dummy dataset for demonstration"""
    def __init__(self, size=1000):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random images for domain A and B
        img_A = torch.randn(3, 256, 256) * 0.1 + 0.5  # Slightly different distributions
        img_B = torch.randn(3, 256, 256) * 0.1 - 0.5

        # Clamp to [-1, 1] range
        img_A = torch.clamp(img_A, -1, 1)
        img_B = torch.clamp(img_B, -1, 1)

        return {'A': img_A, 'B': img_B}

def train_cyclegan():
    # Hyperparameters
    batch_size = 1  # CycleGAN typically uses batch_size=1
    num_epochs = 200
    lr = 0.0002
    beta1 = 0.5
    lambda_cycle = 10.0  # Cycle consistency loss weight
    lambda_identity = 0.5  # Identity loss weight (optional)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets (using dummy data for demonstration)
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize generators and discriminators
    G_AB = CycleGANGenerator().to(device)  # A -> B
    G_BA = CycleGANGenerator().to(device)  # B -> A
    D_A = CycleGANDiscriminator().to(device)  # Discriminator for domain A
    D_B = CycleGANDiscriminator().to(device)  # Discriminator for domain B

    # Apply weight initialization
    G_AB.apply(weights_init)
    G_BA.apply(weights_init)
    D_A.apply(weights_init)
    D_B.apply(weights_init)

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers
    optimizer_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                           lr=lr, betas=(beta1, 0.999))
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))

    # Learning rate schedulers
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch - 100) / float(100 + 1)
        return lr_l

    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_rule)
    scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_rule)

    # Image pools
    fake_A_pool = ImagePool()
    fake_B_pool = ImagePool()

    # Training loop
    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # Labels
            valid = torch.ones(real_A.size(0), 1, device=device, requires_grad=False)
            fake = torch.zeros(real_A.size(0), 1, device=device, requires_grad=False)

            # Train Generators
            optimizer_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(G_BA(real_A), real_A)
            loss_id_B = criterion_identity(G_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN losses
            fake_B = G_AB(real_A)
            pred_fake = D_B(fake_B)
            loss_GAN_AB = criterion_GAN(pred_fake, valid)

            fake_A = G_BA(real_B)
            pred_fake = D_A(fake_A)
            loss_GAN_BA = criterion_GAN(pred_fake, valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle consistency losses
            recovered_A = G_BA(fake_B)
            loss_cycle_A = criterion_cycle(recovered_A, real_A)

            recovered_B = G_AB(fake_A)
            loss_cycle_B = criterion_cycle(recovered_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total generator loss
            loss_G = loss_GAN + lambda_cycle * loss_cycle + lambda_identity * loss_identity
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator A
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = D_A(real_A)
            loss_D_real = criterion_GAN(pred_real, valid)

            # Fake loss (use image pool)
            fake_A_ = fake_A_pool.query(fake_A)
            pred_fake = D_A(fake_A_.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake)

            # Total discriminator A loss
            loss_D_A = (loss_D_real + loss_D_fake) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # Train Discriminator B
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = D_B(real_B)
            loss_D_real = criterion_GAN(pred_real, valid)

            # Fake loss (use image pool)
            fake_B_ = fake_B_pool.query(fake_B)
            pred_fake = D_B(fake_B_.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake)

            # Total discriminator B loss
            loss_D_B = (loss_D_real + loss_D_fake) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            # Print statistics
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_G: {loss_G.item():.4f} '
                      f'Loss_D_A: {loss_D_A.item():.4f} '
                      f'Loss_D_B: {loss_D_B.item():.4f} '
                      f'Loss_Cycle: {loss_cycle.item():.4f}')

        # Update learning rates
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # Save sample images
        if epoch % 10 == 0:
            with torch.no_grad():
                fake_B = G_AB(real_A[:4])
                fake_A = G_BA(real_B[:4])

                # Create comparison grid
                comparison = torch.cat([real_A[:4], fake_B, real_B[:4], fake_A], dim=0)
                vutils.save_image(comparison, f'cyclegan_comparison_epoch_{epoch}.png',
                                nrow=4, normalize=True)

    return G_AB, G_BA, D_A, D_B

def demonstrate_cycle_consistency():
    """Demonstrate the cycle consistency concept"""
    print("CycleGAN Cycle Consistency:")
    print("Forward cycle: A -> G_AB(A) -> G_BA(G_AB(A)) ≈ A")
    print("Backward cycle: B -> G_BA(B) -> G_AB(G_BA(B)) ≈ B")
    print()
    print("Loss = L_GAN(G_AB, D_B) + L_GAN(G_BA, D_A)")
    print("     + λ * (L_cyc(G_BA, G_AB) + L_cyc(G_AB, G_BA))")
    print()
    print("Where L_cyc enforces cycle consistency without paired training data")

def visualize_translation(G_AB, G_BA, real_A, real_B):
    """Visualize domain translation results"""
    G_AB.eval()
    G_BA.eval()

    with torch.no_grad():
        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        # Cycle consistency check
        recovered_A = G_BA(fake_B)
        recovered_B = G_AB(fake_A)

        # Create visualization grid
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # First row: A -> B -> A
        axes[0, 0].imshow(np.transpose(vutils.make_grid(real_A[:1], normalize=True).cpu(), (1, 2, 0)))
        axes[0, 0].set_title('Real A')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(np.transpose(vutils.make_grid(fake_B[:1], normalize=True).cpu(), (1, 2, 0)))
        axes[0, 1].set_title('Fake B (A->B)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(np.transpose(vutils.make_grid(recovered_A[:1], normalize=True).cpu(), (1, 2, 0)))
        axes[0, 2].set_title('Recovered A (A->B->A)')
        axes[0, 2].axis('off')

        axes[0, 3].axis('off')

        # Second row: B -> A -> B
        axes[1, 0].imshow(np.transpose(vutils.make_grid(real_B[:1], normalize=True).cpu(), (1, 2, 0)))
        axes[1, 0].set_title('Real B')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(np.transpose(vutils.make_grid(fake_A[:1], normalize=True).cpu(), (1, 2, 0)))
        axes[1, 1].set_title('Fake A (B->A)')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(np.transpose(vutils.make_grid(recovered_B[:1], normalize=True).cpu(), (1, 2, 0)))
        axes[1, 2].set_title('Recovered B (B->A->B)')
        axes[1, 2].axis('off')

        axes[1, 3].axis('off')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("CycleGAN Example - Cycle-Consistent Adversarial Networks")
    print("Key features:")
    print("- Unpaired image-to-image translation")
    print("- Two generators: G_AB (A->B) and G_BA (B->A)")
    print("- Two discriminators: D_A and D_B")
    print("- Cycle consistency loss")
    print("- Identity loss (optional)")
    print("- Image pool for stable training")

    demonstrate_cycle_consistency()

    # Uncomment to train
    # G_AB, G_BA, D_A, D_B = train_cyclegan()

    print("\nCycleGAN enables learning mappings between domains")
    print("without paired training examples (e.g., horses ↔ zebras)")