"""
Medical Image VAE Example with Result Logging

의료 이미지에 특화된 VAE 구현으로, 훈련 과정과 결과를 체계적으로 저장합니다.
- 다양한 의료 이미지 타입 지원 (chest X-ray, brain MRI, skin lesion)
- 실시간 훈련 진행 상황 저장
- 생성 결과 및 latent space 시각화 저장
- 모델 체크포인트 자동 저장
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from ..medical_data_utils import (
    MedicalImageLoader,
    load_chest_xray_data,
    load_brain_mri_data,
)
from ..result_logger import create_logger_for_generating

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
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.fc1.in_features))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, recon_x.size(1)), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class MedicalImageDataset(Dataset):
    def __init__(self, images, image_size=64, channels=1):
        self.images = images
        self.image_size = image_size
        self.channels = channels

        # Create transform
        if channels == 1:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Ensure proper format
        if len(image.shape) == 2:  # Grayscale
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and self.channels == 1:
            # Convert to grayscale if needed
            if image.shape[-1] == 3:
                image = np.mean(image, axis=-1, keepdims=True)

        # Apply transform
        if self.transform:
            image = self.transform(image.astype(np.uint8))

        return image, 0  # Return dummy label

def train_medical_vae(dataset_type='chest_xray', data_path=None, num_epochs=50, save_interval=10):
    """
    Medical Image VAE 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_generating("vae", dataset_type)

    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    image_size = 64

    # Save configuration
    config = {
        'algorithm': 'VAE',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'image_size': image_size,
        'num_epochs': num_epochs,
        'data_path': data_path
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Load medical data
    logger.log(f"Loading {dataset_type} data...")
    if dataset_type == 'chest_xray':
        images = load_chest_xray_data(data_path, num_samples=1000, image_size=image_size)
        input_channels = 1
    elif dataset_type == 'brain_mri':
        images = load_brain_mri_data(data_path, num_samples=1000, image_size=image_size)
        input_channels = 1
    else:  # skin_lesion or others
        loader = MedicalImageLoader(dataset_type, image_size)
        if data_path and os.path.exists(data_path):
            images = loader.load_real_dataset(data_path, 1000)
        else:
            images = loader.create_synthetic_medical_data(1000)
        input_channels = 3 if len(images[0].shape) == 3 else 1

    logger.log(f"Loaded {len(images)} {dataset_type} images")
    logger.log(f"Image shape: {images[0].shape}")

    # Save sample original images
    sample_images = [images[i] for i in range(min(9, len(images)))]
    logger.save_image_grid(sample_images, "original_samples",
                          titles=[f"Original {i+1}" for i in range(len(sample_images))],
                          cmap='gray' if input_channels == 1 else None)

    # Create dataset and dataloader
    dataset = MedicalImageDataset(images, image_size, input_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    input_dim = image_size * image_size * input_channels
    latent_dim = 32
    hidden_dim = 512

    model = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    logger.log(f"VAE Architecture:")
    logger.log(f"- Input dim: {input_dim}")
    logger.log(f"- Hidden dim: {hidden_dim}")
    logger.log(f"- Latent dim: {latent_dim}")
    logger.log(f"- Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    model.train()
    logger.log("Starting VAE training...")

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_bce_loss = 0
        epoch_kld_loss = 0
        num_batches = 0

        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            data_flat = data.view(data.size(0), -1)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data_flat)

            # Calculate individual loss components
            bce = F.binary_cross_entropy(recon_batch, data_flat, reduction='sum')
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = bce + kld

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_bce_loss += bce.item()
            epoch_kld_loss += kld.item()
            num_batches += 1

        # Calculate average losses
        avg_loss = epoch_loss / num_batches
        avg_bce = epoch_bce_loss / num_batches
        avg_kld = epoch_kld_loss / num_batches

        # Log metrics
        logger.log_metrics(epoch + 1, avg_loss, BCE_loss=avg_bce, KLD_loss=avg_kld)

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Saving intermediate results at epoch {epoch + 1}...")

            # Generate samples
            model.eval()
            with torch.no_grad():
                # Generate from random latent vectors
                sample_z = torch.randn(16, latent_dim).to(device)
                generated = model.decode(sample_z).cpu()

                # Reshape to image format
                if input_channels == 1:
                    generated_images = generated.view(16, image_size, image_size).numpy()
                else:
                    generated_images = generated.view(16, input_channels, image_size, image_size)
                    generated_images = generated_images.permute(0, 2, 3, 1).numpy()

                # Convert to proper range
                generated_images = np.clip(generated_images, 0, 1)
                if input_channels == 1:
                    generated_images = (generated_images * 255).astype(np.uint8)
                else:
                    generated_images = (generated_images * 255).astype(np.uint8)

                # Save generated samples
                logger.save_image_grid(
                    [generated_images[i] for i in range(16)],
                    f"generated_samples_epoch_{epoch+1:03d}",
                    titles=[f"Gen {i+1}" for i in range(16)],
                    rows=4, cols=4,
                    cmap='gray' if input_channels == 1 else None
                )

                # Reconstruction comparison
                test_batch = next(iter(dataloader))[0][:8].to(device)
                test_batch_flat = test_batch.view(test_batch.size(0), -1)
                recon_batch, _, _ = model(test_batch_flat)

                # Reshape for visualization
                original = test_batch.cpu().numpy()
                if input_channels == 1:
                    original = original.squeeze(1)
                    reconstructed = recon_batch.view(-1, image_size, image_size).cpu().numpy()
                else:
                    original = original.permute(0, 2, 3, 1).numpy()
                    reconstructed = recon_batch.view(-1, input_channels, image_size, image_size)
                    reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().numpy()

                # Create comparison grid
                comparison_images = []
                comparison_titles = []
                for i in range(8):
                    comparison_images.append(original[i])
                    comparison_images.append(reconstructed[i])
                    comparison_titles.extend([f"Orig {i+1}", f"Recon {i+1}"])

                logger.save_image_grid(
                    comparison_images,
                    f"reconstruction_comparison_epoch_{epoch+1:03d}",
                    titles=comparison_titles,
                    rows=8, cols=2,
                    cmap='gray' if input_channels == 1 else None
                )

                # Latent space interpolation
                z1 = torch.randn(1, latent_dim).to(device)
                z2 = torch.randn(1, latent_dim).to(device)

                interpolation_steps = 10
                interpolated_images = []
                for i in range(interpolation_steps):
                    alpha = i / (interpolation_steps - 1)
                    z_interp = (1 - alpha) * z1 + alpha * z2
                    interp_img = model.decode(z_interp).cpu()

                    if input_channels == 1:
                        interp_img = interp_img.view(image_size, image_size).numpy()
                    else:
                        interp_img = interp_img.view(input_channels, image_size, image_size)
                        interp_img = interp_img.permute(1, 2, 0).numpy()

                    interp_img = np.clip(interp_img, 0, 1)
                    if input_channels == 1:
                        interp_img = (interp_img * 255).astype(np.uint8)
                    else:
                        interp_img = (interp_img * 255).astype(np.uint8)

                    interpolated_images.append(interp_img)

                logger.save_image_grid(
                    interpolated_images,
                    f"latent_interpolation_epoch_{epoch+1:03d}",
                    titles=[f"Step {i+1}" for i in range(interpolation_steps)],
                    rows=1, cols=interpolation_steps,
                    cmap='gray' if input_channels == 1 else None
                )

            model.train()

        # Save model checkpoint
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(model, f"vae_checkpoint_epoch_{epoch+1:03d}",
                             optimizer=optimizer, epoch=epoch+1,
                             config=config)

    # Final results
    logger.log("Training completed! Saving final results...")

    # Plot training curves
    logger.plot_training_curves()

    # Save final model
    logger.save_model(model, "vae_final_model", optimizer=optimizer,
                     epoch=num_epochs, config=config)

    # Final sample generation
    model.eval()
    with torch.no_grad():
        # Large batch of samples
        sample_z = torch.randn(64, latent_dim).to(device)
        final_generated = model.decode(sample_z).cpu()

        if input_channels == 1:
            final_images = final_generated.view(64, image_size, image_size).numpy()
        else:
            final_images = final_generated.view(64, input_channels, image_size, image_size)
            final_images = final_images.permute(0, 2, 3, 1).numpy()

        final_images = np.clip(final_images * 255, 0, 255).astype(np.uint8)

        logger.save_image_grid(
            [final_images[i] for i in range(64)],
            "final_generated_samples",
            titles=[f"Final {i+1}" for i in range(64)],
            rows=8, cols=8,
            cmap='gray' if input_channels == 1 else None
        )

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return model, results_dir

if __name__ == "__main__":
    print("Medical Image VAE with Result Logging")
    print("=====================================")

    # Configuration
    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'  # Change this to test different datasets

    print(f"Training VAE on {selected_dataset} images...")
    print("Results will be automatically saved including:")
    print("- Training progress logs")
    print("- Generated sample images")
    print("- Reconstruction comparisons")
    print("- Latent space interpolations")
    print("- Model checkpoints")
    print("- Training curves")

    # Train the model
    try:
        model, results_dir = train_medical_vae(
            dataset_type=selected_dataset,
            data_path=None,  # Use synthetic data - set to real path if available
            num_epochs=30,
            save_interval=5
        )

        print(f"\nTraining completed successfully!")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Generated samples and comparisons")
        print("- models/: Model checkpoints")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves")
        print("- metrics/: Training metrics in JSON format")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise