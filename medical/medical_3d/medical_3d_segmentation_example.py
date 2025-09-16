"""
Medical 3D Segmentation Implementation

의료영상 3D 분할을 위한 구현으로, CT, MRI 등의 볼륨 데이터에서
장기나 병변을 3차원으로 분할합니다.

의료 특화 기능:
- 3D U-Net 아키텍처
- 볼륨 데이터 처리 최적화
- 다중 장기 동시 분할
- 3D 시각화 및 메쉬 생성
- 메모리 효율적 패치 기반 처리

핵심 특징:
1. 3D Convolutional Networks
2. Volumetric Data Processing
3. Multi-organ Segmentation
4. Memory-efficient Patch Processing
5. 3D Visualization and Mesh Generation

Reference:
- Çiçek, Ö., et al. (2016).
  "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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

from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

from medical.result_logger import create_logger_for_medical_3d

class Conv3DBlock(nn.Module):
    """3D Convolution block with batch norm and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DoubleConv3D(nn.Module):
    """Double 3D convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv3DBlock(in_channels, out_channels),
            Conv3DBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is NCDHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttentionGate3D(nn.Module):
    """3D Attention Gate for focusing on relevant features"""
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(inter_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Medical3DUNet(nn.Module):
    """3D U-Net for medical volume segmentation"""
    def __init__(self, n_channels=1, n_classes=4, trilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        # Encoder
        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        factor = 2 if trilinear else 1
        self.down4 = Down3D(512, 1024 // factor)

        # Attention gates
        self.att1 = AttentionGate3D(512, 1024 // factor, 256)
        self.att2 = AttentionGate3D(256, 512, 128)
        self.att3 = AttentionGate3D(128, 256, 64)
        self.att4 = AttentionGate3D(64, 128, 32)

        # Decoder
        self.up1 = Up3D(1024, 512 // factor, trilinear)
        self.up2 = Up3D(512, 256 // factor, trilinear)
        self.up3 = Up3D(256, 128 // factor, trilinear)
        self.up4 = Up3D(128, 64, trilinear)

        # Output
        self.outc = nn.Conv3d(64, n_classes, kernel_size=1)

        # Dropout for regularization
        self.dropout = nn.Dropout3d(p=0.1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Apply dropout in bottleneck
        x5 = self.dropout(x5)

        # Decoder with attention
        x4_att = self.att1(x4, x5)
        x = self.up1(x5, x4_att)

        x3_att = self.att2(x3, x)
        x = self.up2(x, x3_att)

        x2_att = self.att3(x2, x)
        x = self.up3(x, x2_att)

        x1_att = self.att4(x1, x)
        x = self.up4(x, x1_att)

        logits = self.outc(x)
        return logits

class DiceLoss3D(nn.Module):
    """3D Dice Loss for volumetric segmentation"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets, class_weights=None):
        # Flatten tensors
        inputs = F.softmax(inputs, dim=1)

        total_loss = 0
        num_classes = inputs.shape[1]

        for i in range(num_classes):
            input_flat = inputs[:, i].view(-1)
            target_flat = (targets == i).float().view(-1)

            intersection = (input_flat * target_flat).sum()
            dice = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)

            weight = class_weights[i] if class_weights is not None else 1.0
            total_loss += weight * (1 - dice)

        return total_loss / num_classes

class Medical3DDataset(Dataset):
    """Dataset for 3D medical volumes"""
    def __init__(self, volumes, masks, patch_size=(64, 64, 64), overlap=0.5):
        self.volumes = volumes
        self.masks = masks
        self.patch_size = patch_size
        self.overlap = overlap
        self.patches = self._extract_patches()

    def _extract_patches(self):
        """Extract overlapping patches from volumes"""
        patches = []

        for vol_idx, (volume, mask) in enumerate(zip(self.volumes, self.masks)):
            d, h, w = volume.shape
            pd, ph, pw = self.patch_size

            # Calculate step size based on overlap
            step_d = int(pd * (1 - self.overlap))
            step_h = int(ph * (1 - self.overlap))
            step_w = int(pw * (1 - self.overlap))

            for z in range(0, d - pd + 1, step_d):
                for y in range(0, h - ph + 1, step_h):
                    for x in range(0, w - pw + 1, step_w):
                        vol_patch = volume[z:z+pd, y:y+ph, x:x+pw]
                        mask_patch = mask[z:z+pd, y:y+ph, x:x+pw]

                        # Only keep patches with some foreground
                        if np.sum(mask_patch > 0) > 100:  # Threshold for meaningful content
                            patches.append((vol_patch, mask_patch))

        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        volume_patch, mask_patch = self.patches[idx]

        # Convert to tensors
        volume_tensor = torch.from_numpy(volume_patch).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_patch).long()

        return volume_tensor, mask_tensor

def create_synthetic_3d_data(dataset_type, num_volumes=50, volume_size=(128, 128, 128)):
    """Create synthetic 3D medical volumes"""
    print(f"Creating {num_volumes} synthetic 3D {dataset_type} volumes...")

    volumes = []
    masks = []

    for i in range(num_volumes):
        d, h, w = volume_size

        if dataset_type == 'ct_thorax':
            # Create thorax CT volume
            volume = np.random.randn(d, h, w) * 0.1 + 0.3

            # Create mask with multiple organs
            mask = np.zeros(volume_size, dtype=np.uint8)

            # Lungs (class 1)
            for lung_offset in [-w//6, w//6]:  # Left and right lungs
                lung_center_x = w//2 + lung_offset
                for z in range(d//4, 3*d//4):
                    for y in range(h//4, 3*h//4):
                        dist = np.sqrt((lung_center_x - w//2 - lung_offset)**2 + (y - h//2)**2)
                        if dist < w//8:
                            mask[z, y, lung_center_x] = 1
                            volume[z, y, lung_center_x] = -0.8  # Air-filled lungs

            # Heart (class 2)
            heart_center = (d//2, h//2, w//2)
            for z in range(max(0, heart_center[0] - d//8), min(d, heart_center[0] + d//8)):
                for y in range(max(0, heart_center[1] - h//12), min(h, heart_center[1] + h//8)):
                    for x in range(max(0, heart_center[2] - w//12), min(w, heart_center[2] + w//12)):
                        dist = np.sqrt((z - heart_center[0])**2 + (y - heart_center[1])**2 + (x - heart_center[2])**2)
                        if dist < d//12:
                            mask[z, y, x] = 2
                            volume[z, y, x] = 0.6  # Heart tissue

            # Liver (class 3) - lower part
            for z in range(2*d//3, d):
                for y in range(h//3, 2*h//3):
                    for x in range(w//4, 3*w//4):
                        if np.random.random() > 0.3:  # Make it somewhat irregular
                            mask[z, y, x] = 3
                            volume[z, y, x] = 0.4  # Liver tissue

        elif dataset_type == 'mri_brain':
            # Create brain MRI volume
            volume = np.random.randn(d, h, w) * 0.1 + 0.5

            mask = np.zeros(volume_size, dtype=np.uint8)

            # Brain tissue (class 1)
            brain_center = (d//2, h//2, w//2)
            brain_radius = min(d, h, w) // 3

            for z in range(d):
                for y in range(h):
                    for x in range(w):
                        dist = np.sqrt((z - brain_center[0])**2 + (y - brain_center[1])**2 + (x - brain_center[2])**2)
                        if dist < brain_radius:
                            mask[z, y, x] = 1
                            volume[z, y, x] = 0.7  # Brain tissue

            # Ventricles (class 2)
            ventricle_centers = [(d//2, h//2 - h//8, w//2), (d//2, h//2 + h//8, w//2)]
            for center in ventricle_centers:
                for z in range(max(0, center[0] - d//16), min(d, center[0] + d//16)):
                    for y in range(max(0, center[1] - h//16), min(h, center[1] + h//16)):
                        for x in range(max(0, center[2] - w//16), min(w, center[2] + w//16)):
                            dist = np.sqrt((z - center[0])**2 + (y - center[1])**2 + (x - center[2])**2)
                            if dist < d//20:
                                mask[z, y, x] = 2
                                volume[z, y, x] = 0.1  # CSF

            # Tumor (class 3) - random location
            if np.random.random() > 0.5:  # 50% chance of tumor
                tumor_center = (
                    np.random.randint(d//4, 3*d//4),
                    np.random.randint(h//4, 3*h//4),
                    np.random.randint(w//4, 3*w//4)
                )
                tumor_radius = np.random.randint(d//20, d//12)

                for z in range(max(0, tumor_center[0] - tumor_radius), min(d, tumor_center[0] + tumor_radius)):
                    for y in range(max(0, tumor_center[1] - tumor_radius), min(h, tumor_center[1] + tumor_radius)):
                        for x in range(max(0, tumor_center[2] - tumor_radius), min(w, tumor_center[2] + tumor_radius)):
                            dist = np.sqrt((z - tumor_center[0])**2 + (y - tumor_center[1])**2 + (x - tumor_center[2])**2)
                            if dist < tumor_radius:
                                mask[z, y, x] = 3
                                volume[z, y, x] = 0.9  # Tumor tissue

        # Normalize volume
        volume = np.clip(volume, -1, 1)

        volumes.append(volume)
        masks.append(mask)

    return np.array(volumes), np.array(masks)

def visualize_3d_volume(volume, mask, save_path, class_names):
    """Visualize 3D volume and segmentation"""
    fig = plt.figure(figsize=(20, 12))

    # Show axial slices
    slice_indices = [volume.shape[0]//4, volume.shape[0]//2, 3*volume.shape[0]//4]

    for i, slice_idx in enumerate(slice_indices):
        # Volume slice
        ax1 = plt.subplot(2, 6, i+1)
        plt.imshow(volume[slice_idx], cmap='gray')
        plt.title(f'Volume Slice {slice_idx}')
        plt.axis('off')

        # Mask slice
        ax2 = plt.subplot(2, 6, i+4)
        plt.imshow(mask[slice_idx], cmap='tab10', vmin=0, vmax=len(class_names)-1)
        plt.title(f'Mask Slice {slice_idx}')
        plt.axis('off')

    # 3D visualization of mask
    ax3d = fig.add_subplot(2, 6, (4, 6), projection='3d')

    # Create isosurface for each class
    colors = ['red', 'blue', 'green', 'yellow']
    for class_id in range(1, min(4, len(class_names))):
        if np.sum(mask == class_id) > 100:  # Only if enough voxels
            try:
                verts, faces, _, _ = measure.marching_cubes(mask == class_id, level=0.5)
                ax3d.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                                 color=colors[class_id-1], alpha=0.3, label=class_names[class_id])
            except:
                pass  # Skip if marching cubes fails

    ax3d.set_title('3D Segmentation')
    ax3d.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def calculate_3d_metrics(pred_mask, true_mask, num_classes):
    """Calculate 3D segmentation metrics"""
    metrics = {}

    for class_id in range(num_classes):
        pred_binary = (pred_mask == class_id).astype(np.uint8)
        true_binary = (true_mask == class_id).astype(np.uint8)

        # Dice coefficient
        intersection = np.sum(pred_binary * true_binary)
        dice = (2.0 * intersection) / (np.sum(pred_binary) + np.sum(true_binary) + 1e-8)

        # Jaccard index
        union = np.sum(pred_binary) + np.sum(true_binary) - intersection
        jaccard = intersection / (union + 1e-8)

        metrics[f'class_{class_id}'] = {'dice': dice, 'jaccard': jaccard}

    # Overall metrics
    overall_dice = np.mean([metrics[f'class_{i}']['dice'] for i in range(num_classes)])
    overall_jaccard = np.mean([metrics[f'class_{i}']['jaccard'] for i in range(num_classes)])

    metrics['overall'] = {'dice': overall_dice, 'jaccard': overall_jaccard}

    return metrics

def train_medical_3d_segmentation(dataset_type='ct_thorax', num_epochs=30, save_interval=5):
    """
    Medical 3D Segmentation 훈련 함수
    """
    # Create result logger
    logger = create_logger_for_medical_3d("3d_segmentation", dataset_type)

    # Check for quick test mode
    if os.getenv('QUICK_TEST') == '1':
        num_epochs = min(int(os.getenv('TEST_EPOCHS', 5)), num_epochs)

    # Configuration
    batch_size = 2  # Small batch size for 3D due to memory constraints
    volume_size = (128, 128, 128)
    patch_size = (64, 64, 64)
    learning_rate = 1e-4
    num_classes = 4  # background + 3 organs/structures

    class_names = {
        'ct_thorax': ['Background', 'Lungs', 'Heart', 'Liver'],
        'mri_brain': ['Background', 'Brain', 'Ventricles', 'Tumor']
    }[dataset_type]

    config = {
        'algorithm': 'Medical 3D Segmentation',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'volume_size': volume_size,
        'patch_size': patch_size,
        'learning_rate': learning_rate,
        'num_classes': num_classes,
        'class_names': class_names,
        'num_epochs': num_epochs
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Create 3D data
    logger.log(f"Creating 3D {dataset_type} volumes...")
    volumes, masks = create_synthetic_3d_data(dataset_type, num_volumes=40, volume_size=volume_size)

    # Split data
    split_idx = int(0.8 * len(volumes))
    train_volumes, val_volumes = volumes[:split_idx], volumes[split_idx:]
    train_masks, val_masks = masks[:split_idx], masks[split_idx:]

    logger.log(f"Training volumes: {len(train_volumes)}, Validation volumes: {len(val_volumes)}")

    # Create datasets
    train_dataset = Medical3DDataset(train_volumes, train_masks, patch_size, overlap=0.5)
    val_dataset = Medical3DDataset(val_volumes, val_masks, patch_size, overlap=0.25)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.log(f"Training patches: {len(train_dataset)}, Validation patches: {len(val_dataset)}")

    # Visualize sample data
    visualize_3d_volume(train_volumes[0], train_masks[0],
                       os.path.join(logger.dirs['images'], 'sample_3d_volume.png'),
                       class_names)

    # Initialize model
    model = Medical3DUNet(n_channels=1, n_classes=num_classes, trilinear=True).to(device)
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    class_weights = torch.tensor([0.1, 1.0, 2.0, 3.0]).to(device)  # Weight rare classes more
    criterion = DiceLoss3D()
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    # Training loop
    logger.log("Starting 3D segmentation training...")

    train_losses = []
    val_losses = []
    dice_scores = []

    best_dice = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Calculate combined loss
            dice_loss = criterion(output, target, class_weights)
            ce_loss = ce_criterion(output, target)
            total_loss = 0.7 * dice_loss + 0.3 * ce_loss

            total_loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += total_loss.item()

            if batch_idx % 5 == 0:
                logger.log(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {total_loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_metrics = []

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)

                dice_loss = criterion(output, target, class_weights)
                ce_loss = ce_criterion(output, target)
                total_loss = 0.7 * dice_loss + 0.3 * ce_loss

                val_loss += total_loss.item()

                # Calculate metrics for a subset of patches
                pred_masks = torch.argmax(output, dim=1).cpu().numpy()
                true_masks = target.cpu().numpy()

                for pred_mask, true_mask in zip(pred_masks, true_masks):
                    metrics = calculate_3d_metrics(pred_mask, true_mask, num_classes)
                    all_metrics.append(metrics['overall']['dice'])

        avg_val_loss = val_loss / len(val_loader)
        avg_dice = np.mean(all_metrics)

        val_losses.append(avg_val_loss)
        dice_scores.append(avg_dice)

        scheduler.step()

        logger.log(f'Epoch {epoch+1}/{num_epochs}:')
        logger.log(f'  Train Loss: {avg_train_loss:.4f}')
        logger.log(f'  Val Loss: {avg_val_loss:.4f}')
        logger.log(f'  Dice Score: {avg_dice:.4f}')
        logger.log(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save best model
        if avg_dice > best_dice:
            best_dice = avg_dice
            logger.save_model(model, "3d_segmentation_best", optimizer=optimizer,
                            metadata={'epoch': epoch+1, 'dice': avg_dice})

        # Save intermediate results
        if (epoch + 1) % save_interval == 0:
            # Generate sample predictions on full volumes
            with torch.no_grad():
                model.eval()
                sample_volume = val_volumes[0]
                sample_mask = val_masks[0]

                # Process volume in patches
                d, h, w = sample_volume.shape
                pd, ph, pw = patch_size

                pred_volume = np.zeros_like(sample_mask)

                # Simple non-overlapping prediction for visualization
                for z in range(0, d - pd + 1, pd):
                    for y in range(0, h - ph + 1, ph):
                        for x in range(0, w - pw + 1, pw):
                            patch = sample_volume[z:z+pd, y:y+ph, x:x+pw]
                            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                            pred_patch = model(patch_tensor)
                            pred_patch = torch.argmax(pred_patch, dim=1).squeeze().cpu().numpy()

                            pred_volume[z:z+pd, y:y+ph, x:x+pw] = pred_patch

                # Visualize results
                visualize_3d_volume(sample_volume, pred_volume,
                                   os.path.join(logger.dirs['images'], f'3d_prediction_epoch_{epoch+1}.png'),
                                   class_names)

        # Log metrics
        logger.log_metric("train_loss", avg_train_loss, epoch)
        logger.log_metric("val_loss", avg_val_loss, epoch)
        logger.log_metric("dice_score", avg_dice, epoch)

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('3D Segmentation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(dice_scores, label='Dice Score', color='green')
    plt.title('Validation Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    learning_rates = [lr * (0.5 ** (epoch // 15)) for epoch in range(num_epochs)]
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
    logger.save_model(model, "3d_segmentation_final", optimizer=optimizer,
                     metadata={'epochs': num_epochs, 'best_dice': best_dice})

    logger.log("3D segmentation training completed successfully!")
    logger.log(f"Best Dice Score: {best_dice:.4f}")
    logger.log(f"All results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

if __name__ == "__main__":
    print("Medical 3D Segmentation Implementation")
    print("=" * 45)
    print("Training 3D U-Net for volumetric medical image segmentation...")
    print("Key Features:")
    print("- 3D U-Net with attention gates")
    print("- Multi-organ volumetric segmentation")
    print("- Memory-efficient patch processing")
    print("- 3D visualization and mesh generation")
    print("- Class-weighted loss functions")

    model, results_dir = train_medical_3d_segmentation(
        dataset_type='ct_thorax',
        num_epochs=20,
        save_interval=5
    )

    print(f"\nTraining completed!")
    print(f"Results saved to: {results_dir}")
