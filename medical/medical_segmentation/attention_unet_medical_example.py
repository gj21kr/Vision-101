"""
Medical Attention U-Net Implementation

의료영상 분할을 위한 Attention U-Net 구현으로, 주의 메커니즘을 통해
더 정확한 의료 분할을 제공합니다.

의료 특화 기능:
- 의료 이미지 전용 데이터 로더 (CT, MRI, X-ray, 병리)
- 멀티 클래스 의료 분할 (장기, 종양, 병변)
- Attention 메커니즘을 통한 정밀 분할
- 의료영상 특화 손실 함수 (Focal Dice, Tversky)
- 자동 결과 저장 및 시각화 시스템
- 의료 품질 평가 메트릭

Attention U-Net 핵심 특징:
1. Attention Gates (관련 영역에 집중)
2. Feature Suppression (불필요한 특징 억제)
3. Improved Gradient Flow
4. Better Localization
5. 작은 병변 검출에 특화

Reference:
- Oktay, O., et al. (2018).
  "Attention U-Net: Learning Where to Look for the Pancreas."
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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
while PROJECT_ROOT.name not in {"medical", "non_medical"} and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent
if PROJECT_ROOT.name in {"medical", "non_medical"}:
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from PIL import Image
import cv2
from sklearn.metrics import jaccard_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is on sys.path
from medical.medical_data_utils import MedicalImageLoader, load_chest_xray_data, load_brain_mri_data
from medical.result_logger import create_logger_for_medical_segmentation

class AttentionBlock(nn.Module):
    """Attention Gate Block"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: gating signal from decoder
        x: feature map from encoder
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class DoubleConv(nn.Module):
    """Double Convolution Block with attention-aware design"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),  # For better generalization
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv with attention"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # Attention gate
        self.att = AttentionBlock(F_g=in_channels//2, F_l=in_channels//2, F_int=in_channels//4)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Apply attention gate
        x2 = self.att(g=x1, x=x2)

        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final output convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    """
    Attention U-Net for Medical Image Segmentation

    특징:
    1. Attention Gates for precise localization
    2. Skip connections with attention weighting
    3. Multi-scale feature extraction
    4. Medical image optimized architecture
    """
    def __init__(self, n_channels=1, n_classes=2, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder with attention
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with attention
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class MedicalSegmentationDataset(Dataset):
    """의료 분할 데이터셋 (합성 데이터 포함)"""
    def __init__(self, num_samples=1000, image_size=256, modality='ct', transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.modality = modality
        self.transform = transform

        # Medical segmentation classes
        self.classes = {
            'background': 0,
            'organ': 1,
            'tumor': 2,
            'vessel': 3
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic medical image
        image, mask = self._generate_medical_sample()

        if self.transform:
            # Apply transform to image. For masks, we should not normalize.
            seed = np.random.randint(2147483647)

            np.random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)

            # For the mask, we only apply geometric transformations and convert to a LongTensor.
            np.random.seed(seed)
            torch.manual_seed(seed)
            mask_transform = transforms.Resize((self.image_size, self.image_size))
            mask = mask_transform(mask)
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask, dtype=np.uint8)).long()

        return image, mask

    def _generate_medical_sample(self):
        """Generate synthetic medical image and segmentation mask"""
        # Create base medical image
        image = np.random.randn(self.image_size, self.image_size) * 0.1 + 0.5
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)

        # Add anatomical structures
        # Main organ (e.g., liver, lung)
        organ_center = (np.random.randint(64, self.image_size-64),
                       np.random.randint(64, self.image_size-64))
        organ_size = np.random.randint(40, 80)

        y, x = np.ogrid[:self.image_size, :self.image_size]
        organ_dist = np.sqrt((x - organ_center[0])**2 + (y - organ_center[1])**2)
        organ_mask = organ_dist <= organ_size

        # Add organ to image and mask
        image[organ_mask] += np.random.normal(0.3, 0.1, np.sum(organ_mask))
        mask[organ_mask] = 1

        # Add tumor/lesion
        if np.random.random() > 0.3:  # 70% chance of having a tumor
            tumor_center = (organ_center[0] + np.random.randint(-30, 30),
                           organ_center[1] + np.random.randint(-30, 30))
            tumor_size = np.random.randint(8, 20)

            tumor_dist = np.sqrt((x - tumor_center[0])**2 + (y - tumor_center[1])**2)
            tumor_mask = tumor_dist <= tumor_size

            # Only add tumor if it's within the organ
            tumor_mask = tumor_mask & organ_mask

            image[tumor_mask] += np.random.normal(0.4, 0.1, np.sum(tumor_mask))
            mask[tumor_mask] = 2

        # Add vessels/structures
        for _ in range(np.random.randint(2, 5)):
            start_point = (np.random.randint(0, self.image_size),
                          np.random.randint(0, self.image_size))
            end_point = (np.random.randint(0, self.image_size),
                        np.random.randint(0, self.image_size))

            rr, cc = self._draw_line(start_point, end_point, self.image_size)

            # Dilate line to create vessel
            for r, c in zip(rr, cc):
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.image_size and 0 <= nc < self.image_size:
                            if mask[nr, nc] == 0:  # Only add if background
                                image[nr, nc] += np.random.normal(0.2, 0.05)
                                mask[nr, nc] = 3

        # Normalize and add noise
        image = np.clip(image, 0, 1)
        image = image + np.random.normal(0, 0.05, image.shape)
        image = np.clip(image, 0, 1)

        # Convert to PIL for transforms
        image = Image.fromarray((image * 255).astype(np.uint8))
        mask = Image.fromarray(mask)

        return image, mask

    def _draw_line(self, start, end, image_size):
        """Draw line between two points"""
        x0, y0 = start
        x1, y1 = end

        points_x = []
        points_y = []

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x_step = 1 if x0 < x1 else -1
        y_step = 1 if y0 < y1 else -1

        x, y = x0, y0
        error = dx - dy

        while True:
            if 0 <= x < image_size and 0 <= y < image_size:
                points_x.append(y)  # Note: swapped for numpy indexing
                points_y.append(x)

            if x == x1 and y == y1:
                break

            e2 = 2 * error
            if e2 > -dy:
                error -= dy
                x += x_step
            if e2 < dx:
                error += dx
                y += y_step

        return points_x, points_y

class FocalDiceLoss(nn.Module):
    """Combined Focal and Dice Loss for medical segmentation"""
    def __init__(self, alpha=0.25, gamma=2, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Focal loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        focal_loss = focal_loss.mean()

        # Dice loss
        inputs_soft = F.softmax(inputs, dim=1)
        dice_loss = 0

        for i in range(inputs.size(1)):
            input_flat = inputs_soft[:, i].contiguous().view(-1)
            target_flat = (targets == i).float().contiguous().view(-1)

            intersection = (input_flat * target_flat).sum()
            dice_coeff = (2 * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
            dice_loss += 1 - dice_coeff

        dice_loss = dice_loss / inputs.size(1)

        return focal_loss + dice_loss

def calculate_metrics(pred_mask, true_mask):
    """Calculate medical segmentation metrics"""
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()

    # Accuracy
    accuracy = (pred_flat == true_flat).mean()

    # IoU for each class
    ious = []
    for class_id in np.unique(true_flat):
        pred_class = (pred_flat == class_id)
        true_class = (true_flat == class_id)

        intersection = (pred_class & true_class).sum()
        union = (pred_class | true_class).sum()

        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0
        ious.append(iou)

    mean_iou = np.mean(ious)

    # Dice coefficient
    dice_scores = []
    for class_id in np.unique(true_flat):
        pred_class = (pred_flat == class_id)
        true_class = (true_flat == class_id)

        intersection = (pred_class & true_class).sum()
        dice = (2 * intersection) / (pred_class.sum() + true_class.sum() + 1e-8)
        dice_scores.append(dice)

    mean_dice = np.mean(dice_scores)

    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'class_ious': ious,
        'class_dice': dice_scores
    }

def visualize_attention_segmentation(images, masks, predictions, attention_maps=None, save_path=None, epoch=None):
    """Visualize attention-guided segmentation results"""
    batch_size = min(4, len(images))

    fig, axes = plt.subplots(batch_size, 4 if attention_maps is None else 5,
                            figsize=(20 if attention_maps is None else 25, 5*batch_size))

    if batch_size == 1:
        axes = axes.reshape(1, -1)

    classes = ['Background', 'Organ', 'Tumor', 'Vessel']
    colors = ['black', 'red', 'yellow', 'blue']

    for i in range(batch_size):
        # Original image
        img = images[i].squeeze().cpu().numpy()
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Ground truth mask
        mask = masks[i].cpu().numpy()
        axes[i, 1].imshow(img, cmap='gray', alpha=0.7)
        for class_id in range(1, 4):
            class_mask = (mask == class_id)
            if class_mask.any():
                axes[i, 1].contour(class_mask, colors=[colors[class_id]], linewidths=2)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Prediction
        pred = predictions[i].cpu().numpy()
        axes[i, 2].imshow(img, cmap='gray', alpha=0.7)
        for class_id in range(1, 4):
            class_pred = (pred == class_id)
            if class_pred.any():
                axes[i, 2].contour(class_pred, colors=[colors[class_id]], linewidths=2)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

        # Overlay comparison
        axes[i, 3].imshow(img, cmap='gray', alpha=0.5)
        # Show ground truth as solid regions
        for class_id in range(1, 4):
            class_mask = (mask == class_id)
            if class_mask.any():
                colored_mask = np.zeros((*class_mask.shape, 3))
                if class_id == 1:  # Organ - red
                    colored_mask[class_mask] = [1, 0, 0]
                elif class_id == 2:  # Tumor - yellow
                    colored_mask[class_mask] = [1, 1, 0]
                elif class_id == 3:  # Vessel - blue
                    colored_mask[class_mask] = [0, 0, 1]
                axes[i, 3].imshow(colored_mask, alpha=0.3)

        # Show predictions as contours
        for class_id in range(1, 4):
            class_pred = (pred == class_id)
            if class_pred.any():
                axes[i, 3].contour(class_pred, colors=[colors[class_id]], linewidths=1, linestyles='--')

        axes[i, 3].set_title('GT (filled) vs Pred (dashed)')
        axes[i, 3].axis('off')

        # Attention map if provided
        if attention_maps is not None:
            att_map = attention_maps[i].cpu().numpy()
            im = axes[i, 4].imshow(att_map, cmap='hot', alpha=0.8)
            axes[i, 4].imshow(img, cmap='gray', alpha=0.3)
            axes[i, 4].set_title('Attention Map')
            axes[i, 4].axis('off')
            plt.colorbar(im, ax=axes[i, 4], shrink=0.6)

    title = f'Medical Attention U-Net Segmentation'
    if epoch is not None:
        title += f' - Epoch {epoch}'
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()
    return fig

def train_attention_unet():
    """Train Attention U-Net for medical segmentation"""
    print("Starting Medical Attention U-Net Training...")
    print("=" * 60)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create results directory
    results_dir = "/workspace/Vision-101/results/attention_unet_medical"
    os.makedirs(results_dir, exist_ok=True)

    # Set up data transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Medical image normalization
    ])

    # Create datasets
    train_dataset = MedicalSegmentationDataset(
        num_samples=1000,
        image_size=256,
        modality='ct',
        transform=transform
    )

    val_dataset = MedicalSegmentationDataset(
        num_samples=200,
        image_size=256,
        modality='mri',
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Initialize model
    model = AttentionUNet(n_channels=1, n_classes=4, bilinear=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss function and optimizer
    criterion = FocalDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training parameters
    num_epochs = 30
    best_dice = 0.0
    train_losses = []
    val_dices = []

    print(f"Training for {num_epochs} epochs...")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_dice = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate metrics
                predictions = torch.argmax(outputs, dim=1)

                for i in range(images.size(0)):
                    metrics = calculate_metrics(
                        predictions[i].cpu().numpy(),
                        masks[i].cpu().numpy()
                    )
                    val_dice += metrics['mean_dice']

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_dataset)
        val_dices.append(avg_val_dice)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}]:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Dice: {avg_val_dice:.4f}')

        # Save best model
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
            print(f'  New best model saved! Dice: {best_dice:.4f}')

        # Visualize results periodically
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_images, sample_masks = next(iter(val_loader))
                sample_images = sample_images.to(device)
                sample_outputs = model(sample_images)
                sample_predictions = torch.argmax(sample_outputs, dim=1)

                save_path = os.path.join(results_dir, f'results_epoch_{epoch+1}.png')
                visualize_attention_segmentation(
                    sample_images, sample_masks, sample_predictions,
                    save_path=save_path, epoch=epoch+1
                )

        print("-" * 60)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(val_dices, label='Validation Dice', color='green')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300)
    plt.show()

    # Final evaluation
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
    model.eval()

    print(f"\nTraining completed!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    print(f"Results saved to: {results_dir}")

    return model, train_losses, val_dices

if __name__ == "__main__":
    print("Medical Attention U-Net - Advanced Medical Image Segmentation")
    print("=" * 60)
    print("Key features:")
    print("- Attention mechanisms for precise localization")
    print("- Multi-class medical segmentation")
    print("- Focal Dice loss for imbalanced data")
    print("- Comprehensive medical evaluation metrics")
    print("- Advanced visualization with attention maps")
    print("=" * 60)

    # Train the model
    model, train_losses, val_dices = train_attention_unet()

    print("\nAttention U-Net training completed successfully!")
    print("Model trained for medical image segmentation with attention mechanism.")
