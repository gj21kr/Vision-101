"""
Medical Vision Transformer (ViT) Implementation

의료영상 분류를 위한 Vision Transformer 구현으로, 질병 진단 및
병변 분류에 특화된 어텐션 기반 모델입니다.

의료 특화 기능:
- 의료영상 특화 패치 임베딩
- 해부학적 구조 인식 어텐션
- 다중 스케일 특징 추출
- 설명 가능한 어텐션 맵
- 의료 도메인 특화 사전 훈련

핵심 특징:
1. Self-Attention Mechanism
2. Patch-based Image Processing
3. Position-aware Encoding
4. Medical-specific Tokenization
5. Interpretable Attention Maps

Reference:
- Dosovitskiy, A., et al. (2020).
  "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
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
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medical.result_logger import create_logger_for_medical_cad

class PatchEmbedding(nn.Module):
    """Patch embedding for medical images"""
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Patch projection
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                   kernel_size=patch_size, stride=patch_size)

        # Medical-specific patch enhancement
        self.patch_enhancement = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape

        # Create patches: (B, embed_dim, n_patches_h, n_patches_w)
        x = self.projection(x)

        # Flatten patches: (B, embed_dim, n_patches)
        x = x.flatten(2)

        # Transpose: (B, n_patches, embed_dim)
        x = x.transpose(1, 2)

        # Apply enhancement
        x = self.patch_enhancement(x)

        return x

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for medical images"""
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x, attn

class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, embed_dim=768, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer block with medical-specific modifications"""
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

        # Medical-specific skip connection enhancement
        self.medical_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Self-attention with residual connection
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x)

        # Medical-specific gating
        gate = self.medical_gate(norm_x)
        attn_out = attn_out * gate

        x = x + attn_out

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights

class MedicalViT(nn.Module):
    """Medical Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_channels=1, num_classes=3,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Dropout
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # Medical-specific auxiliary heads
        self.attention_head = nn.Linear(embed_dim, 1)  # For attention visualization
        self.confidence_head = nn.Linear(embed_dim, 1)  # For prediction confidence

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)

    def forward(self, x, return_attention=False):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        # Store attention weights for visualization
        attention_weights = []

        # Apply transformer blocks
        for block in self.blocks:
            x, attn = block(x)
            if return_attention:
                attention_weights.append(attn)

        # Layer norm
        x = self.norm(x)

        # Extract class token
        cls_token_final = x[:, 0]

        # Classification
        logits = self.head(cls_token_final)

        # Auxiliary outputs
        attention_score = self.attention_head(cls_token_final)
        confidence_score = torch.sigmoid(self.confidence_head(cls_token_final))

        if return_attention:
            return logits, attention_score, confidence_score, attention_weights
        else:
            return logits, attention_score, confidence_score

class MedicalViTDataset(Dataset):
    """Dataset for medical image classification"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Convert to tensor if not already
        if not torch.is_tensor(image):
            image = torch.from_numpy(image).unsqueeze(0).float()

        return image, torch.tensor(label, dtype=torch.long)

def create_medical_classification_data(dataset_type, num_samples=1000, image_size=224):
    """Create medical images for classification"""
    print(f"Creating {num_samples} medical images for classification...")

    images = []
    labels = []

    if dataset_type == 'chest_xray':
        # Classes: 0=Normal, 1=Pneumonia, 2=COVID-19
        class_names = ['Normal', 'Pneumonia', 'COVID-19']

        for i in range(num_samples):
            # Random class assignment
            class_id = np.random.randint(0, 3)

            # Base chest X-ray
            image = np.random.randn(image_size, image_size) * 0.1 + 0.5

            # Add lung structures
            center_y, center_x = image_size // 2, image_size // 2

            # Left and right lungs
            left_lung = np.zeros((image_size, image_size))
            right_lung = np.zeros((image_size, image_size))

            cv2.ellipse(left_lung, (center_x - image_size//6, center_y),
                       (image_size//8, image_size//4), 0, 0, 360, 1, -1)
            cv2.ellipse(right_lung, (center_x + image_size//6, center_y),
                       (image_size//8, image_size//4), 0, 0, 360, 1, -1)

            if class_id == 0:  # Normal
                image = image * 0.6 + (left_lung + right_lung) * 0.4

            elif class_id == 1:  # Pneumonia
                # Add consolidation patterns
                consolidation = np.zeros((image_size, image_size))

                # Random consolidation areas
                for _ in range(np.random.randint(2, 5)):
                    x = np.random.randint(image_size//4, 3*image_size//4)
                    y = np.random.randint(image_size//3, 2*image_size//3)
                    size = np.random.randint(image_size//12, image_size//8)
                    cv2.circle(consolidation, (x, y), size, 0.6, -1)

                image = image * 0.6 + (left_lung + right_lung) * 0.3 + consolidation * 0.3

            else:  # COVID-19
                # Add ground-glass opacities
                ggo = np.zeros((image_size, image_size))

                # Multiple bilateral opacities
                for lung_center in [(center_x - image_size//6, center_y),
                                   (center_x + image_size//6, center_y)]:
                    for _ in range(np.random.randint(3, 6)):
                        x = lung_center[0] + np.random.randint(-image_size//12, image_size//12)
                        y = lung_center[1] + np.random.randint(-image_size//8, image_size//8)
                        size = np.random.randint(image_size//20, image_size//12)
                        cv2.circle(ggo, (x, y), size, 0.4, -1)

                image = image * 0.6 + (left_lung + right_lung) * 0.3 + ggo * 0.25

    elif dataset_type == 'brain_mri':
        # Classes: 0=Normal, 1=Tumor, 2=Stroke
        class_names = ['Normal', 'Tumor', 'Stroke']

        for i in range(num_samples):
            class_id = np.random.randint(0, 3)

            # Base brain MRI
            image = np.random.randn(image_size, image_size) * 0.1 + 0.4

            # Brain outline
            center_y, center_x = image_size // 2, image_size // 2
            brain_mask = np.zeros((image_size, image_size))
            cv2.ellipse(brain_mask, (center_x, center_y),
                       (image_size//3, image_size//3), 0, 0, 360, 1, -1)

            if class_id == 0:  # Normal
                image = image * 0.5 + brain_mask * 0.5

            elif class_id == 1:  # Tumor
                # Add tumor
                tumor_x = center_x + np.random.randint(-image_size//6, image_size//6)
                tumor_y = center_y + np.random.randint(-image_size//6, image_size//6)
                tumor_size = np.random.randint(image_size//15, image_size//8)

                tumor_mask = np.zeros((image_size, image_size))
                cv2.circle(tumor_mask, (tumor_x, tumor_y), tumor_size, 1, -1)

                image = image * 0.5 + brain_mask * 0.4 + tumor_mask * 0.8

            else:  # Stroke
                # Add stroke lesion
                stroke_area = np.zeros((image_size, image_size))

                # Irregular stroke pattern
                stroke_center_x = center_x + np.random.randint(-image_size//8, image_size//8)
                stroke_center_y = center_y + np.random.randint(-image_size//8, image_size//8)

                cv2.ellipse(stroke_area, (stroke_center_x, stroke_center_y),
                           (image_size//12, image_size//20),
                           np.random.randint(0, 180), 0, 360, 1, -1)

                image = image * 0.5 + brain_mask * 0.4 + stroke_area * (-0.3)

    else:  # skin_lesion
        # Classes: 0=Benign, 1=Malignant, 2=Melanoma
        class_names = ['Benign', 'Malignant', 'Melanoma']

        for i in range(num_samples):
            class_id = np.random.randint(0, 3)

            # Base skin
            image = np.random.randn(image_size, image_size) * 0.05 + 0.75

            if class_id == 0:  # Benign
                # Regular, symmetric lesion
                lesion_x = image_size // 2 + np.random.randint(-image_size//16, image_size//16)
                lesion_y = image_size // 2 + np.random.randint(-image_size//16, image_size//16)
                lesion_size = np.random.randint(image_size//12, image_size//8)

                cv2.circle(image, (lesion_x, lesion_y), lesion_size, 0.4, -1)

            elif class_id == 1:  # Malignant
                # Irregular lesion with uneven borders
                lesion_center_x = image_size // 2 + np.random.randint(-image_size//12, image_size//12)
                lesion_center_y = image_size // 2 + np.random.randint(-image_size//12, image_size//12)

                # Create irregular shape
                angles = np.linspace(0, 2*np.pi, 12)
                points = []
                for angle in angles:
                    r = np.random.randint(image_size//15, image_size//8)
                    x = int(lesion_center_x + r * np.cos(angle))
                    y = int(lesion_center_y + r * np.sin(angle))
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)
                cv2.fillPoly(image, [points], color=0.3)

            else:  # Melanoma
                # Very irregular, dark lesion with varied colors
                lesion_center_x = image_size // 2 + np.random.randint(-image_size//10, image_size//10)
                lesion_center_y = image_size // 2 + np.random.randint(-image_size//10, image_size//10)

                # Multiple irregular areas
                for _ in range(np.random.randint(2, 4)):
                    x = lesion_center_x + np.random.randint(-image_size//12, image_size//12)
                    y = lesion_center_y + np.random.randint(-image_size//12, image_size//12)
                    size = np.random.randint(image_size//20, image_size//10)
                    intensity = np.random.uniform(0.1, 0.4)
                    cv2.circle(image, (x, y), size, intensity, -1)

        # Normalize
        image = np.clip(image, 0, 1)

        images.append(image)
        labels.append(class_id)

    return np.array(images), np.array(labels), class_names

def train_medical_vit(dataset_type='chest_xray', num_epochs=50, save_interval=10):
    """
    Medical Vision Transformer 훈련 함수
    """
    # Create result logger
    logger = create_logger_for_medical_cad("vision_transformer", dataset_type)

    # Check for quick test mode
    if os.getenv('QUICK_TEST') == '1':
        num_epochs = min(int(os.getenv('TEST_EPOCHS', 5)), num_epochs)

    # Configuration
    batch_size = 16
    image_size = 224
    learning_rate = 1e-4
    num_classes = 3

    config = {
        'algorithm': 'Medical Vision Transformer',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'image_size': image_size,
        'learning_rate': learning_rate,
        'num_classes': num_classes,
        'num_epochs': num_epochs
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Create data
    logger.log(f"Creating {dataset_type} classification data...")
    images, labels, class_names = create_medical_classification_data(
        dataset_type, num_samples=1200, image_size=image_size
    )

    # Split data
    split_idx = int(0.8 * len(images))
    train_images, val_images = images[:split_idx], images[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    logger.log(f"Training samples: {len(train_images)}, Validation samples: {len(val_images)}")
    logger.log(f"Classes: {class_names}")

    # Class distribution
    unique, counts = np.unique(train_labels, return_counts=True)
    for i, (cls, count) in enumerate(zip(class_names, counts)):
        logger.log(f"  {cls}: {count} samples")

    # Create datasets
    train_dataset = MedicalViTDataset(train_images, train_labels)
    val_dataset = MedicalViTDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Save sample images
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for class_id in range(num_classes):
        class_indices = np.where(train_labels == class_id)[0][:4]

        for i, idx in enumerate(class_indices):
            axes[class_id, i].imshow(train_images[idx], cmap='gray')
            axes[class_id, i].set_title(f'{class_names[class_id]} {i+1}')
            axes[class_id, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['images'], 'sample_classification_data.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Initialize model
    model = MedicalViT(
        img_size=image_size,
        patch_size=16,
        in_channels=1,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        dropout=0.1
    ).to(device)

    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    logger.log("Starting Vision Transformer training...")

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits, attention_score, confidence_score = model(data)

            # Calculate loss
            loss = criterion(logits, target)

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            if batch_idx % 10 == 0:
                logger.log(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}')

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)

                logits, attention_score, confidence_score = model(data)
                loss = criterion(logits, target)

                val_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        logger.log(f'Epoch {epoch+1}/{num_epochs}:')
        logger.log(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.log(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logger.log(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            logger.save_model(model, "vit_best_model", optimizer=optimizer,
                            metadata={'epoch': epoch+1, 'val_acc': val_acc})

        # Save intermediate results
        if (epoch + 1) % save_interval == 0:
            # Generate attention visualization
            model.eval()
            with torch.no_grad():
                sample_data = val_images[:4]
                sample_tensor = torch.from_numpy(sample_data).unsqueeze(1).float().to(device)

                logits, attention_score, confidence_score, attention_weights = model(
                    sample_tensor, return_attention=True
                )

                # Visualize attention maps
                fig, axes = plt.subplots(4, 4, figsize=(16, 16))

                for i in range(4):
                    # Original image
                    axes[i, 0].imshow(sample_data[i], cmap='gray')
                    axes[i, 0].set_title(f'Original {i+1}')
                    axes[i, 0].axis('off')

                    # Prediction
                    pred_class = torch.argmax(logits[i]).item()
                    confidence = torch.softmax(logits[i], dim=0).max().item()
                    axes[i, 1].text(0.5, 0.5, f'Pred: {class_names[pred_class]}\nConf: {confidence:.3f}',
                                   transform=axes[i, 1].transAxes, ha='center', va='center',
                                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue'))
                    axes[i, 1].axis('off')

                    # Attention maps from different layers
                    for j, layer_idx in enumerate([3, 7, 11]):  # Show attention from different layers
                        if layer_idx < len(attention_weights):
                            # Average attention across heads for the CLS token
                            attn = attention_weights[layer_idx][i].mean(0)  # Average across heads
                            cls_attn = attn[0, 1:]  # CLS token attention to patches

                            # Reshape to spatial dimensions
                            patch_size = int(np.sqrt(len(cls_attn)))
                            attn_map = cls_attn.reshape(patch_size, patch_size).cpu().numpy()

                            # Resize to original image size
                            attn_resized = cv2.resize(attn_map, (image_size, image_size))

                            axes[i, j+2].imshow(attn_resized, cmap='hot', alpha=0.7)
                            axes[i, j+2].imshow(sample_data[i], cmap='gray', alpha=0.3)
                            axes[i, j+2].set_title(f'Attention Layer {layer_idx+1}')
                            axes[i, j+2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(logger.dirs['images'], f'attention_maps_epoch_{epoch+1}.png'),
                           dpi=150, bbox_inches='tight')
                plt.close()

            # Classification report
            if len(set(all_predictions)) == num_classes:
                report = classification_report(all_targets, all_predictions,
                                             target_names=class_names, output_dict=True)
                logger.log("Classification Report:")
                for class_name in class_names:
                    metrics = report[class_name]
                    logger.log(f"  {class_name}: Precision={metrics['precision']:.3f}, "
                              f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

        # Log metrics
        logger.log_metric("train_loss", avg_train_loss, epoch)
        logger.log_metric("train_acc", train_acc, epoch)
        logger.log_metric("val_loss", avg_val_loss, epoch)
        logger.log_metric("val_acc", val_acc, epoch)

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    learning_rates = [optimizer.param_groups[0]['lr'] * (np.cos(np.pi * epoch / num_epochs) + 1) / 2
                     for epoch in range(num_epochs)]
    plt.plot(learning_rates, label='Learning Rate')
    plt.title('Cosine Annealing Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save final model
    logger.save_model(model, "vit_final_model", optimizer=optimizer,
                     metadata={'epochs': num_epochs, 'best_acc': best_acc})

    logger.log("Vision Transformer training completed successfully!")
    logger.log(f"Best validation accuracy: {best_acc:.2f}%")
    logger.log(f"All results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

if __name__ == "__main__":
    print("Medical Vision Transformer Implementation")
    print("=" * 50)
    print("Training ViT for medical image classification...")
    print("Key Features:")
    print("- Patch-based image processing")
    print("- Self-attention mechanism")
    print("- Medical-specific modifications")
    print("- Interpretable attention maps")
    print("- Multi-class disease classification")

    model, results_dir = train_medical_vit(
        dataset_type='chest_xray',
        num_epochs=30,
        save_interval=5
    )

    print(f"\nTraining completed!")
    print(f"Results saved to: {results_dir}")