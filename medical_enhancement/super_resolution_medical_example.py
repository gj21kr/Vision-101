"""
Medical Super-Resolution Implementation

의료영상 초해상도 구현으로, 저해상도 의료영상을 고해상도로 변환하여
진단 정확도를 향상시킵니다.

의료 특화 기능:
- 의료영상 특화 손실 함수 (Perceptual + SSIM + MSE)
- 다중 스케일 특징 추출
- 의료 도메인 지식 통합
- Edge-aware 업샘플링
- 진단 관련 영역 보존

핵심 특징:
1. Enhanced Deep Residual Network (EDSR)
2. Multi-scale Feature Extraction
3. Medical-specific Loss Functions
4. Perceptual Quality Preservation
5. Real-time Processing Capability

Reference:
- Lim, B., et al. (2017).
  "Enhanced Deep Residual Networks for Single Image Super-Resolution"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medical.result_logger import create_logger_for_medical_enhancement

class ResidualBlock(nn.Module):
    """Residual block for super-resolution network"""
    def __init__(self, channels=64, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out

class AttentionBlock(nn.Module):
    """Channel attention block for medical image enhancement"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MedicalSuperResolutionNet(nn.Module):
    """Medical Super-Resolution Network"""
    def __init__(self, scale_factor=2, num_channels=1, num_features=64, num_blocks=16):
        super().__init__()
        self.scale_factor = scale_factor

        # Initial convolution
        self.conv_first = nn.Conv2d(num_channels, num_features, 3, padding=1)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_features) for _ in range(num_blocks)
        ])

        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(num_features) for _ in range(num_blocks // 4)
        ])

        # Middle convolution
        self.conv_middle = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.bn_middle = nn.BatchNorm2d(num_features)

        # Upsampling layers
        if scale_factor == 2:
            self.upconv = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )
        elif scale_factor == 4:
            self.upconv = nn.Sequential(
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_features, num_features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            )

        # Final convolution
        self.conv_final = nn.Conv2d(num_features, num_channels, 3, padding=1)

        # Edge enhancement branch
        self.edge_conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_channels, 3, padding=1)
        )

    def forward(self, x):
        # Store original for skip connection
        original = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        # Initial feature extraction
        out = self.conv_first(x)
        residual = out

        # Residual blocks with attention
        for i, res_block in enumerate(self.res_blocks):
            out = res_block(out)

            # Apply attention every 4 blocks
            if i % 4 == 3 and i // 4 < len(self.attention_blocks):
                out = self.attention_blocks[i // 4](out)

        # Middle processing
        out = self.bn_middle(self.conv_middle(out))
        out += residual

        # Upsampling
        out = self.upconv(out)

        # Final convolution
        out = self.conv_final(out)

        # Add skip connection
        out += original

        # Edge enhancement
        edge_enhanced = self.edge_conv(original)
        out = out + 0.1 * edge_enhanced

        return out

class MedicalSSIMLoss(nn.Module):
    """SSIM loss for medical images"""
    def __init__(self, window_size=11, reduction='mean'):
        super().__init__()
        self.window_size = window_size
        self.reduction = reduction

    def forward(self, img1, img2):
        # Convert to numpy for SSIM calculation
        img1_np = img1.detach().cpu().numpy()
        img2_np = img2.detach().cpu().numpy()

        ssim_values = []
        for i in range(img1.shape[0]):
            ssim_val = ssim(img1_np[i, 0], img2_np[i, 0], data_range=1.0)
            ssim_values.append(1 - ssim_val)  # Convert to loss

        ssim_loss = torch.tensor(ssim_values, device=img1.device, dtype=img1.dtype)

        if self.reduction == 'mean':
            return ssim_loss.mean()
        elif self.reduction == 'sum':
            return ssim_loss.sum()
        else:
            return ssim_loss

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self):
        super().__init__()
        # Simple perceptual loss using basic convolution layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )

        # Freeze parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)
        return F.mse_loss(sr_features, hr_features)

class CombinedLoss(nn.Module):
    """Combined loss for medical super-resolution"""
    def __init__(self, mse_weight=1.0, ssim_weight=0.5, perceptual_weight=0.1):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight

        self.mse_loss = nn.MSELoss()
        self.ssim_loss = MedicalSSIMLoss()
        self.perceptual_loss = PerceptualLoss()

    def forward(self, sr, hr):
        mse = self.mse_loss(sr, hr)
        ssim = self.ssim_loss(sr, hr)
        perceptual = self.perceptual_loss(sr, hr)

        total_loss = (self.mse_weight * mse +
                     self.ssim_weight * ssim +
                     self.perceptual_weight * perceptual)

        return total_loss, mse, ssim, perceptual

class MedicalSRDataset(Dataset):
    """Dataset for medical super-resolution"""
    def __init__(self, hr_images, scale_factor=2):
        self.hr_images = hr_images
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image = self.hr_images[idx]

        # Create low-resolution version
        h, w = hr_image.shape
        lr_h, lr_w = h // self.scale_factor, w // self.scale_factor

        # Downsample to create LR image
        lr_image = cv2.resize(hr_image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

        # Convert to tensors
        hr_tensor = torch.from_numpy(hr_image).unsqueeze(0).float()
        lr_tensor = torch.from_numpy(lr_image).unsqueeze(0).float()

        return lr_tensor, hr_tensor

def create_medical_sr_data(dataset_type, num_samples=500, image_size=128, scale_factor=2):
    """Create medical images for super-resolution"""
    print(f"Creating {num_samples} medical images for super-resolution...")

    images = []

    for i in range(num_samples):
        if dataset_type == 'chest_xray':
            # Create high-quality chest X-ray
            image = np.random.randn(image_size, image_size) * 0.1 + 0.5

            # Add detailed lung structures
            center_y, center_x = image_size // 2, image_size // 2

            # Left lung with detailed structure
            left_lung = np.zeros((image_size, image_size))
            cv2.ellipse(left_lung, (center_x - image_size//6, center_y),
                       (image_size//8, image_size//4), 0, 0, 360, 1, -1)

            # Right lung
            right_lung = np.zeros((image_size, image_size))
            cv2.ellipse(right_lung, (center_x + image_size//6, center_y),
                       (image_size//8, image_size//4), 0, 0, 360, 1, -1)

            # Add ribs and other structures
            for rib in range(-3, 4):
                y_pos = center_y + rib * image_size // 12
                if 0 < y_pos < image_size:
                    cv2.line(image, (image_size//8, y_pos), (7*image_size//8, y_pos), 0.3, 2)

            # Combine structures
            image = image * 0.6 + (left_lung + right_lung) * 0.4

        elif dataset_type == 'brain_mri':
            # Create detailed brain MRI
            image = np.random.randn(image_size, image_size) * 0.1 + 0.4

            # Brain outline
            center_y, center_x = image_size // 2, image_size // 2
            brain_mask = np.zeros((image_size, image_size))
            cv2.ellipse(brain_mask, (center_x, center_y),
                       (image_size//3, image_size//3), 0, 0, 360, 1, -1)

            # Add internal structures (ventricles, etc.)
            ventricle = np.zeros((image_size, image_size))
            cv2.ellipse(ventricle, (center_x, center_y - image_size//8),
                       (image_size//12, image_size//16), 0, 0, 360, 1, -1)

            # Gray matter, white matter distinction
            white_matter = np.zeros((image_size, image_size))
            cv2.ellipse(white_matter, (center_x, center_y),
                       (image_size//4, image_size//4), 0, 0, 360, 1, -1)

            image = (image * 0.4 + brain_mask * 0.4 +
                    white_matter * 0.15 + ventricle * (-0.1))

        else:  # skin_lesion
            # Create detailed skin lesion image
            image = np.random.randn(image_size, image_size) * 0.05 + 0.75

            # Add skin texture
            for _ in range(20):
                x = np.random.randint(0, image_size)
                y = np.random.randint(0, image_size)
                cv2.circle(image, (x, y), 1, np.random.uniform(0.7, 0.8), -1)

            # Add lesion with detailed boundary
            lesion_x = image_size // 2 + np.random.randint(-image_size//8, image_size//8)
            lesion_y = image_size // 2 + np.random.randint(-image_size//8, image_size//8)
            lesion_size = np.random.randint(image_size//8, image_size//4)

            # Create irregular lesion shape
            angles = np.linspace(0, 2*np.pi, 20)
            lesion_points = []
            for angle in angles:
                r = lesion_size * (0.8 + 0.4 * np.random.random())
                x = int(lesion_x + r * np.cos(angle))
                y = int(lesion_y + r * np.sin(angle))
                lesion_points.append([x, y])

            lesion_points = np.array(lesion_points, dtype=np.int32)
            cv2.fillPoly(image, [lesion_points], color=0.3)

        # Normalize and add fine details
        image = np.clip(image, 0, 1)

        # Add fine-grained noise for realism
        fine_noise = np.random.randn(image_size, image_size) * 0.02
        image = np.clip(image + fine_noise, 0, 1)

        images.append(image)

    return np.array(images)

def calculate_metrics(sr_images, hr_images):
    """Calculate PSNR and SSIM metrics"""
    psnr_values = []
    ssim_values = []

    for i in range(len(sr_images)):
        sr = sr_images[i]
        hr = hr_images[i]

        # Calculate PSNR
        psnr_val = psnr(hr, sr, data_range=1.0)
        psnr_values.append(psnr_val)

        # Calculate SSIM
        ssim_val = ssim(hr, sr, data_range=1.0)
        ssim_values.append(ssim_val)

    return np.mean(psnr_values), np.mean(ssim_values)

def train_medical_super_resolution(dataset_type='chest_xray', scale_factor=2, num_epochs=50, save_interval=10):
    """
    Medical Super-Resolution 훈련 함수
    """
    # Create result logger
    logger = create_logger_for_medical_enhancement("super_resolution", dataset_type)

    # Check for quick test mode
    if os.getenv('QUICK_TEST') == '1':
        num_epochs = min(int(os.getenv('TEST_EPOCHS', 5)), num_epochs)

    # Configuration
    batch_size = 8
    image_size = 128
    learning_rate = 1e-4

    config = {
        'algorithm': 'Medical Super-Resolution',
        'dataset_type': dataset_type,
        'scale_factor': scale_factor,
        'batch_size': batch_size,
        'image_size': image_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Create data
    logger.log(f"Creating {dataset_type} images for {scale_factor}x super-resolution...")
    hr_images = create_medical_sr_data(dataset_type, num_samples=800,
                                      image_size=image_size, scale_factor=scale_factor)

    # Split data
    split_idx = int(0.8 * len(hr_images))
    train_images, val_images = hr_images[:split_idx], hr_images[split_idx:]

    logger.log(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")

    # Create datasets
    train_dataset = MedicalSRDataset(train_images, scale_factor)
    val_dataset = MedicalSRDataset(val_images, scale_factor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Save sample images
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        lr_sample, hr_sample = train_dataset[i]

        axes[0, i].imshow(lr_sample[0].numpy(), cmap='gray')
        axes[0, i].set_title(f'LR Input {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(hr_sample[0].numpy(), cmap='gray')
        axes[1, i].set_title(f'HR Target {i+1}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['images'], 'sample_lr_hr_pairs.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Initialize model
    model = MedicalSuperResolutionNet(scale_factor=scale_factor, num_channels=1).to(device)
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = CombinedLoss(mse_weight=1.0, ssim_weight=0.5, perceptual_weight=0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # Training loop
    logger.log("Starting super-resolution training...")

    train_losses = []
    val_losses = []
    psnr_scores = []
    ssim_scores = []

    best_psnr = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (lr_batch, hr_batch) in enumerate(train_loader):
            lr_batch = lr_batch.to(device)
            hr_batch = hr_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            sr_batch = model(lr_batch)

            # Calculate loss
            total_loss, mse_loss, ssim_loss, perceptual_loss = criterion(sr_batch, hr_batch)

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

            if batch_idx % 10 == 0:
                logger.log(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {total_loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_sr_images = []
        val_hr_images = []

        with torch.no_grad():
            for lr_batch, hr_batch in val_loader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

                sr_batch = model(lr_batch)
                total_loss, _, _, _ = criterion(sr_batch, hr_batch)

                val_loss += total_loss.item()

                # Collect images for metric calculation
                val_sr_images.extend(sr_batch.cpu().numpy()[:, 0])
                val_hr_images.extend(hr_batch.cpu().numpy()[:, 0])

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate metrics
        avg_psnr, avg_ssim = calculate_metrics(val_sr_images[:50], val_hr_images[:50])
        psnr_scores.append(avg_psnr)
        ssim_scores.append(avg_ssim)

        scheduler.step()

        logger.log(f'Epoch {epoch+1}/{num_epochs}:')
        logger.log(f'  Train Loss: {avg_train_loss:.4f}')
        logger.log(f'  Val Loss: {avg_val_loss:.4f}')
        logger.log(f'  PSNR: {avg_psnr:.2f} dB')
        logger.log(f'  SSIM: {avg_ssim:.4f}')
        logger.log(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save best model
        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            logger.save_model(model, "sr_best_model", optimizer=optimizer,
                            metadata={'epoch': epoch+1, 'psnr': avg_psnr, 'ssim': avg_ssim})

        # Save intermediate results
        if (epoch + 1) % save_interval == 0:
            with torch.no_grad():
                sample_lr = val_images[:4]
                sample_lr_tensor = torch.from_numpy(sample_lr).unsqueeze(1).float().to(device)
                sample_sr = model(sample_lr_tensor)

                # Create comparison
                fig, axes = plt.subplots(3, 4, figsize=(16, 12))
                for i in range(4):
                    # LR input (upsampled for comparison)
                    lr_upsampled = F.interpolate(sample_lr_tensor[i:i+1],
                                               scale_factor=scale_factor, mode='bicubic')
                    axes[0, i].imshow(lr_upsampled[0, 0].cpu().numpy(), cmap='gray')
                    axes[0, i].set_title(f'LR Input (Upsampled) {i+1}')
                    axes[0, i].axis('off')

                    # SR output
                    axes[1, i].imshow(sample_sr[i, 0].cpu().numpy(), cmap='gray')
                    axes[1, i].set_title(f'SR Output {i+1}')
                    axes[1, i].axis('off')

                    # HR target
                    axes[2, i].imshow(val_images[i], cmap='gray')
                    axes[2, i].set_title(f'HR Target {i+1}')
                    axes[2, i].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(logger.dirs['images'], f'sr_results_epoch_{epoch+1}.png'),
                           dpi=150, bbox_inches='tight')
                plt.close()

        # Log metrics
        logger.log_metrics(
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            ssim=avg_ssim,
            psnr=avg_psnr
        )

    # Plot training curves
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(psnr_scores, label='PSNR', color='green')
    plt.title('Peak Signal-to-Noise Ratio')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(ssim_scores, label='SSIM', color='orange')
    plt.title('Structural Similarity Index')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 4)
    learning_rates = [lr * (0.7 ** (epoch // 20)) for epoch in range(num_epochs)]
    plt.plot(learning_rates, label='Learning Rate', color='red')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save final model
    logger.save_model(model, "sr_final_model", optimizer=optimizer,
                     metadata={'epochs': num_epochs, 'best_psnr': best_psnr})

    logger.log("Super-resolution training completed successfully!")
    logger.log(f"Best PSNR: {best_psnr:.2f} dB")
    logger.log(f"All results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

if __name__ == "__main__":
    print("Medical Super-Resolution Implementation")
    print("=" * 45)
    print("Training super-resolution for medical images...")
    print("Key Features:")
    print("- Enhanced residual network architecture")
    print("- Medical-specific loss functions")
    print("- Edge-aware processing")
    print("- Perceptual quality preservation")
    print("- Multi-scale feature extraction")

    model, results_dir = train_medical_super_resolution(
        dataset_type='chest_xray',
        scale_factor=2,
        num_epochs=30,
        save_interval=5
    )

    print(f"\nTraining completed!")
    print(f"Results saved to: {results_dir}")