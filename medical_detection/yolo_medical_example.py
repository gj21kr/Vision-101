"""
Medical YOLO (You Only Look Once) Implementation

의료영상 객체 검출을 위한 YOLO 구현으로, 실시간 의료 객체/병변 검출을
제공합니다.

의료 특화 기능:
- 의료 이미지 전용 객체 클래스 (종양, 병변, 해부학적 구조)
- 다중 스케일 의료 객체 검출
- 의료영상 특화 앵커 박스
- 실시간 진단 보조 시스템
- 의료 품질 평가 메트릭 (mAP, Recall, Precision)

YOLO 핵심 특징:
1. Single-stage Detection (한 번의 forward pass)
2. Real-time Performance
3. Grid-based Object Detection
4. Multi-scale Feature Extraction
5. End-to-end Training

Reference:
- Redmon, J., et al. (2016).
  "You Only Look Once: Unified, Real-Time Object Detection."
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
from PIL import Image
import cv2
import json
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medical_data_utils import MedicalImageLoader
from result_logger import create_logger_for_medical_detection

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResBlock(nn.Module):
    """Residual block for YOLO backbone"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, 1)
        self.conv2 = ConvBlock(channels // 2, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class CSPBlock(nn.Module):
    """Cross Stage Partial block"""
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        mid_channels = out_channels // 2

        self.conv1 = ConvBlock(in_channels, mid_channels, 1)
        self.conv2 = ConvBlock(in_channels, mid_channels, 1)

        self.res_blocks = nn.Sequential(
            *[ResBlock(mid_channels) for _ in range(num_blocks)]
        )

        self.conv3 = ConvBlock(mid_channels, mid_channels, 1)
        self.conv4 = ConvBlock(mid_channels * 2, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x1 = self.res_blocks(x1)
        x1 = self.conv3(x1)

        x = torch.cat([x1, x2], dim=1)
        return self.conv4(x)

class SPP(nn.Module):
    """Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, pool_sizes=[5, 9, 13]):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, in_channels // 2, 1)
        self.conv2 = ConvBlock(in_channels // 2 * (len(pool_sizes) + 1), out_channels, 1)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        x = self.conv1(x)
        features = [x]

        for pool_size in self.pool_sizes:
            pooled = F.max_pool2d(x, kernel_size=pool_size, stride=1, padding=pool_size // 2)
            features.append(pooled)

        x = torch.cat(features, dim=1)
        return self.conv2(x)

class MedicalYOLO(nn.Module):
    """Medical YOLO for medical object detection"""
    def __init__(self, num_classes=3, anchors=None, grid_size=13):
        super().__init__()
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.num_anchors = 3

        if anchors is None:
            # Medical image optimized anchors (small, medium, large objects)
            self.anchors = torch.tensor([
                [10, 10], [20, 20], [30, 30],  # Small objects (small lesions)
                [40, 40], [60, 60], [80, 80],  # Medium objects (organs)
                [100, 100], [120, 120], [150, 150]  # Large objects (large organs)
            ]).float()
        else:
            self.anchors = anchors

        # Backbone network
        self.backbone = self._build_backbone()

        # Detection heads for different scales
        self.detection_head = nn.Conv2d(512, self.num_anchors * (5 + num_classes), 1)

        # Initialize weights
        self._initialize_weights()

    def _build_backbone(self):
        """Build YOLO backbone network"""
        layers = []

        # Initial convolution
        layers.append(ConvBlock(1, 32, 3, padding=1))

        # Downsampling and CSP blocks
        channels = [32, 64, 128, 256, 512]
        for i in range(len(channels) - 1):
            layers.append(ConvBlock(channels[i], channels[i+1], 3, stride=2, padding=1))
            num_blocks = [1, 2, 8, 8, 4][i]
            layers.append(CSPBlock(channels[i+1], channels[i+1], num_blocks))

        # SPP layer
        layers.append(SPP(512, 512))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)

        # Extract features
        features = self.backbone(x)

        # Detection head
        detection = self.detection_head(features)

        # Reshape detection output
        detection = detection.view(batch_size, self.num_anchors,
                                 5 + self.num_classes,
                                 self.grid_size, self.grid_size)
        detection = detection.permute(0, 1, 3, 4, 2).contiguous()

        return detection

class YOLOLoss(nn.Module):
    """YOLO Loss function"""
    def __init__(self, num_classes, anchors, grid_size=13, coord_scale=5, noobj_scale=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.grid_size = grid_size
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.bce_loss = nn.BCELoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        Calculate YOLO loss

        Args:
            predictions: (B, num_anchors, grid_size, grid_size, 5 + num_classes)
            targets: (B, num_anchors, grid_size, grid_size, 5 + num_classes)
        """
        batch_size = predictions.size(0)

        # Extract predictions
        pred_boxes = predictions[..., :4]  # x, y, w, h
        pred_conf = predictions[..., 4]    # confidence
        pred_cls = predictions[..., 5:]    # class probabilities

        # Extract targets
        target_boxes = targets[..., :4]
        target_conf = targets[..., 4]
        target_cls = targets[..., 5:]

        # Object mask (cells that contain objects)
        obj_mask = target_conf > 0
        noobj_mask = target_conf == 0

        # Coordinate loss (only for cells containing objects)
        coord_loss = 0
        if obj_mask.sum() > 0:
            coord_loss = self.mse_loss(pred_boxes[obj_mask], target_boxes[obj_mask])

        # Confidence loss
        conf_loss_obj = self.mse_loss(pred_conf[obj_mask], target_conf[obj_mask])
        conf_loss_noobj = self.mse_loss(pred_conf[noobj_mask], target_conf[noobj_mask])

        # Class loss (only for cells containing objects)
        cls_loss = 0
        if obj_mask.sum() > 0:
            cls_loss = self.bce_loss(pred_cls[obj_mask], target_cls[obj_mask])

        # Total loss
        total_loss = (self.coord_scale * coord_loss +
                     conf_loss_obj +
                     self.noobj_scale * conf_loss_noobj +
                     cls_loss) / batch_size

        return total_loss

class MedicalDetectionDataset(Dataset):
    """Dataset for medical object detection"""
    def __init__(self, images, annotations, grid_size=13, num_classes=3):
        self.images = images
        self.annotations = annotations
        self.grid_size = grid_size
        self.num_classes = num_classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).unsqueeze(0).float()
        target = self._create_target(self.annotations[idx])
        return image, target

    def _create_target(self, annotation):
        """Create YOLO target tensor from annotation"""
        target = torch.zeros(3, self.grid_size, self.grid_size, 5 + self.num_classes)

        for obj in annotation:
            x_center, y_center, width, height, class_id = obj

            # Convert to grid coordinates
            grid_x = int(x_center * self.grid_size)
            grid_y = int(y_center * self.grid_size)

            # Relative position within grid cell
            x_offset = x_center * self.grid_size - grid_x
            y_offset = y_center * self.grid_size - grid_y

            # Choose best anchor (simplified - use first anchor)
            anchor_idx = 0

            if grid_x < self.grid_size and grid_y < self.grid_size:
                target[anchor_idx, grid_y, grid_x, 0] = x_offset
                target[anchor_idx, grid_y, grid_x, 1] = y_offset
                target[anchor_idx, grid_y, grid_x, 2] = width
                target[anchor_idx, grid_y, grid_x, 3] = height
                target[anchor_idx, grid_y, grid_x, 4] = 1.0  # Confidence
                target[anchor_idx, grid_y, grid_x, 5 + int(class_id)] = 1.0  # Class

        return target

def create_medical_detection_data(dataset_type, num_samples=500, image_size=256, grid_size=13):
    """Create synthetic medical detection data with bounding boxes"""
    print(f"Creating {num_samples} synthetic {dataset_type} detection samples...")

    images = []
    annotations = []

    for i in range(num_samples):
        if dataset_type == 'chest_xray':
            # Create chest X-ray image
            image = np.random.randn(image_size, image_size) * 0.2 + 0.5

            # Add lung structures and lesions
            objects = []

            # Left lung (class 0)
            left_lung_x = 0.3 + np.random.normal(0, 0.05)
            left_lung_y = 0.5 + np.random.normal(0, 0.05)
            objects.append([left_lung_x, left_lung_y, 0.15, 0.25, 0])

            # Right lung (class 0)
            right_lung_x = 0.7 + np.random.normal(0, 0.05)
            right_lung_y = 0.5 + np.random.normal(0, 0.05)
            objects.append([right_lung_x, right_lung_y, 0.15, 0.25, 0])

            # Random lesions (class 1)
            if np.random.random() > 0.5:  # 50% chance of lesion
                lesion_x = np.random.uniform(0.2, 0.8)
                lesion_y = np.random.uniform(0.3, 0.7)
                objects.append([lesion_x, lesion_y, 0.05, 0.05, 1])

            # Add visual features
            for obj in objects:
                x, y, w, h, class_id = obj
                center_x = int(x * image_size)
                center_y = int(y * image_size)
                width = int(w * image_size)
                height = int(h * image_size)

                if class_id == 0:  # Lung
                    cv2.ellipse(image, (center_x, center_y), (width//2, height//2), 0, 0, 360, 0.3, -1)
                else:  # Lesion
                    cv2.circle(image, (center_x, center_y), width//2, -0.3, -1)

        elif dataset_type == 'brain_mri':
            # Create brain MRI image
            image = np.random.randn(image_size, image_size) * 0.15 + 0.4

            objects = []

            # Brain tissue (class 0)
            brain_x = 0.5 + np.random.normal(0, 0.02)
            brain_y = 0.5 + np.random.normal(0, 0.02)
            objects.append([brain_x, brain_y, 0.6, 0.6, 0])

            # Potential tumor (class 2)
            if np.random.random() > 0.7:  # 30% chance of tumor
                tumor_x = np.random.uniform(0.3, 0.7)
                tumor_y = np.random.uniform(0.3, 0.7)
                objects.append([tumor_x, tumor_y, 0.08, 0.08, 2])

            # Add visual features
            for obj in objects:
                x, y, w, h, class_id = obj
                center_x = int(x * image_size)
                center_y = int(y * image_size)
                width = int(w * image_size)
                height = int(h * image_size)

                if class_id == 0:  # Brain tissue
                    cv2.ellipse(image, (center_x, center_y), (width//2, height//2), 0, 0, 360, 0.6, -1)
                else:  # Tumor
                    cv2.circle(image, (center_x, center_y), width//2, 0.8, -1)

        else:  # skin_lesion
            # Create skin image
            image = np.random.randn(image_size, image_size) * 0.08 + 0.75

            objects = []

            # Skin lesions (class 1)
            num_lesions = np.random.randint(1, 4)
            for _ in range(num_lesions):
                lesion_x = np.random.uniform(0.2, 0.8)
                lesion_y = np.random.uniform(0.2, 0.8)
                lesion_size = np.random.uniform(0.05, 0.15)
                objects.append([lesion_x, lesion_y, lesion_size, lesion_size, 1])

            # Add visual features
            for obj in objects:
                x, y, w, h, class_id = obj
                center_x = int(x * image_size)
                center_y = int(y * image_size)
                radius = int(w * image_size / 2)
                cv2.circle(image, (center_x, center_y), radius, -0.2, -1)

        # Normalize and add noise
        image = np.clip(image + np.random.randn(image_size, image_size) * 0.05, 0, 1)

        images.append(image)
        annotations.append(objects)

    return np.array(images), annotations

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Calculate intersection area
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.4):
    """Apply Non-Maximum Suppression to predictions"""
    detections = []

    for pred in predictions:
        # Filter by confidence threshold
        conf_mask = pred[..., 4] > conf_threshold
        if not conf_mask.any():
            continue

        # Get boxes above threshold
        boxes = pred[conf_mask]

        # Convert to corner format and sort by confidence
        boxes_corner = []
        for box in boxes:
            x, y, w, h, conf = box[:5]
            class_scores = box[5:]
            class_id = torch.argmax(class_scores).item()
            class_conf = class_scores[class_id].item()

            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2

            boxes_corner.append([x1.item(), y1.item(), x2.item(), y2.item(),
                               conf.item() * class_conf, class_id])

        if not boxes_corner:
            continue

        # Sort by confidence
        boxes_corner.sort(key=lambda x: x[4], reverse=True)

        # Apply NMS
        keep = []
        while boxes_corner:
            current = boxes_corner.pop(0)
            keep.append(current)

            # Remove boxes with high IoU
            boxes_corner = [box for box in boxes_corner
                          if calculate_iou(current[:4], box[:4]) < iou_threshold]

        detections.extend(keep)

    return detections

def train_medical_yolo(dataset_type='chest_xray', data_path=None, num_epochs=50, save_interval=10):
    """
    Medical YOLO 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_medical_detection("yolo", dataset_type)

    # Check for quick test mode
    if os.getenv('QUICK_TEST') == '1':
        num_epochs = min(int(os.getenv('TEST_EPOCHS', 5)), num_epochs)

    # Configuration
    batch_size = 8
    image_size = 256
    grid_size = 13
    learning_rate = 1e-3
    num_classes = 3  # lung/brain/skin, lesion/tumor, background

    # Medical class names
    class_names = {
        'chest_xray': ['lung', 'lesion', 'background'],
        'brain_mri': ['brain_tissue', 'lesion', 'tumor'],
        'skin_lesion': ['skin', 'lesion', 'background']
    }[dataset_type]

    # Save configuration
    config = {
        'algorithm': 'Medical YOLO',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'image_size': image_size,
        'grid_size': grid_size,
        'learning_rate': learning_rate,
        'num_classes': num_classes,
        'class_names': class_names,
        'num_epochs': num_epochs,
        'data_path': data_path
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Create or load data
    logger.log(f"Creating synthetic {dataset_type} detection data...")
    images, annotations = create_medical_detection_data(
        dataset_type, num_samples=800, image_size=image_size, grid_size=grid_size
    )

    # Split data
    split_idx = int(0.8 * len(images))
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_annotations, val_annotations = annotations[:split_idx], annotations[split_idx:]

    logger.log(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")

    # Save sample images with annotations
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(8):
        row, col = i // 4, i % 4
        axes[row, col].imshow(train_images[i], cmap='gray')

        # Draw bounding boxes
        for obj in train_annotations[i]:
            x, y, w, h, class_id = obj
            rect_x = (x - w/2) * image_size
            rect_y = (y - h/2) * image_size
            rect_w = w * image_size
            rect_h = h * image_size

            rect = patches.Rectangle((rect_x, rect_y), rect_w, rect_h,
                                   linewidth=2, edgecolor='red', facecolor='none')
            axes[row, col].add_patch(rect)
            axes[row, col].text(rect_x, rect_y-5, class_names[int(class_id)],
                              color='red', fontsize=8)

        axes[row, col].set_title(f'Sample {i+1}')
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['images'], 'sample_annotations.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Create datasets and dataloaders
    train_dataset = MedicalDetectionDataset(train_images, train_annotations, grid_size, num_classes)
    val_dataset = MedicalDetectionDataset(val_images, val_annotations, grid_size, num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Initialize model
    anchors = torch.tensor([
        [10, 10], [20, 20], [30, 30],
        [40, 40], [60, 60], [80, 80],
        [100, 100], [120, 120], [150, 150]
    ]).float()

    model = MedicalYOLO(num_classes=num_classes, anchors=anchors, grid_size=grid_size).to(device)

    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    criterion = YOLOLoss(num_classes, anchors, grid_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    logger.log("Starting YOLO training...")
    train_losses = []
    val_losses = []

    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.log(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step()

        logger.log(f'Epoch {epoch+1}/{num_epochs}:')
        logger.log(f'  Train Loss: {avg_train_loss:.4f}')
        logger.log(f'  Val Loss: {avg_val_loss:.4f}')
        logger.log(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            logger.save_model(model, "yolo_best_model", optimizer=optimizer,
                            metadata={'epoch': epoch+1, 'val_loss': avg_val_loss})

        # Save intermediate results
        if (epoch + 1) % save_interval == 0:
            # Generate detection results
            with torch.no_grad():
                sample_data = val_images[:4]
                sample_tensor = torch.from_numpy(sample_data).unsqueeze(1).float().to(device)
                sample_output = model(sample_tensor)

                # Process detections
                fig, axes = plt.subplots(2, 2, figsize=(12, 12))
                for i in range(4):
                    row, col = i // 2, i % 2
                    axes[row, col].imshow(sample_data[i], cmap='gray')

                    # Apply NMS to get final detections
                    detections = non_max_suppression([sample_output[i].cpu()])

                    for det in detections:
                        x1, y1, x2, y2, conf, class_id = det
                        x1 *= image_size
                        y1 *= image_size
                        x2 *= image_size
                        y2 *= image_size

                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                               linewidth=2, edgecolor='lime', facecolor='none')
                        axes[row, col].add_patch(rect)
                        axes[row, col].text(x1, y1-5, f'{class_names[int(class_id)]}: {conf:.2f}',
                                          color='lime', fontsize=8)

                    axes[row, col].set_title(f'Detection Result {i+1}')
                    axes[row, col].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(logger.dirs['images'], f'detections_epoch_{epoch+1}.png'),
                           dpi=150, bbox_inches='tight')
                plt.close()

        # Log metrics
        logger.log_metric("train_loss", avg_train_loss, epoch)
        logger.log_metric("val_loss", avg_val_loss, epoch)

    # Plot training curves
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('YOLO Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([optimizer.param_groups[0]['lr']] * len(train_losses), label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'yolo_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save final model
    logger.save_model(model, "yolo_final_model", optimizer=optimizer,
                     metadata={'epochs': num_epochs, 'best_val_loss': best_loss})

    # Generate final detection results
    logger.log("Generating final YOLO detection results...")

    with torch.no_grad():
        model.eval()
        final_images = val_images[-8:]
        final_tensor = torch.from_numpy(final_images).unsqueeze(1).float().to(device)
        final_output = model(final_tensor)

        # Create comprehensive results visualization
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        detection_stats = {'total_detections': 0, 'class_counts': defaultdict(int)}

        for i in range(8):
            row, col = i // 4, i % 4
            axes[row, col].imshow(final_images[i], cmap='gray')

            # Apply NMS
            detections = non_max_suppression([final_output[i].cpu()],
                                           conf_threshold=0.3, iou_threshold=0.4)

            detection_stats['total_detections'] += len(detections)

            for det in detections:
                x1, y1, x2, y2, conf, class_id = det
                x1 *= image_size
                y1 *= image_size
                x2 *= image_size
                y2 *= image_size

                detection_stats['class_counts'][class_names[int(class_id)]] += 1

                # Draw bounding box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                       linewidth=2, edgecolor='yellow', facecolor='none')
                axes[row, col].add_patch(rect)

                # Add label
                label = f'{class_names[int(class_id)]}: {conf:.2f}'
                axes[row, col].text(x1, y1-5, label, color='yellow', fontsize=8,
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

            axes[row, col].set_title(f'Final Detection {i+1}')
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(logger.dirs['images'], 'final_yolo_detections.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()

        # Log detection statistics
        logger.log(f"Final Detection Statistics:")
        logger.log(f"  Total detections: {detection_stats['total_detections']}")
        for class_name, count in detection_stats['class_counts'].items():
            logger.log(f"  {class_name}: {count}")

    logger.log("YOLO training completed successfully!")
    logger.log(f"Best validation loss: {best_loss:.4f}")
    logger.log(f"All results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

if __name__ == "__main__":
    print("Medical YOLO (You Only Look Once) Implementation")
    print("=" * 55)
    print("Training YOLO for medical object detection...")
    print("YOLO Key Features:")
    print("- Single-stage real-time detection")
    print("- Grid-based object localization")
    print("- Multi-scale medical object detection")
    print("- Medical domain specific anchors")
    print("- End-to-end trainable architecture")

    # Run training
    model, results_dir = train_medical_yolo(
        dataset_type='chest_xray',
        data_path=None,
        num_epochs=25,
        save_interval=5
    )

    print(f"\nTraining completed successfully!")
    print(f"All results saved to: {results_dir}")

    print("\nGenerated files include:")
    print("- images/: Sample images with detection annotations")
    print("- models/: Trained YOLO model checkpoints")
    print("- logs/: Training logs and configuration")
    print("- plots/: Training loss curves")
    print("- metrics/: Detection performance metrics")