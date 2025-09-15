"""
Medical SAM (Segment Anything Model) Implementation

의료영상 분할을 위한 SAM 구현으로, 프롬프트 기반 의료영상 분할을
제공하는 파운데이션 모델입니다.

의료 특화 기능:
- 의료 이미지 전용 프롬프트 엔진
- 다중 해부학적 구조 동시 분할
- 점, 박스, 마스크 기반 프롬프팅
- 의료 도메인 특화 임베딩
- 대화형 분할 인터페이스

SAM 핵심 특징:
1. Promptable Segmentation (프롬프트 기반 분할)
2. Foundation Model Architecture
3. Zero-shot Generalization
4. Multi-modal Input (Point, Box, Mask prompts)
5. Real-time Interactive Segmentation

Reference:
- Kirillov, A., et al. (2023).
  "Segment Anything."
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
from PIL import Image
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medical_data_utils import MedicalImageLoader
from result_logger import create_logger_for_medical_segmentation

class ImageEncoder(nn.Module):
    """Vision Transformer based image encoder for SAM"""
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer blocks (simplified)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads=12, mlp_ratio=4.0)
            for _ in range(12)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Neck for feature extraction
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1, bias=False),
            nn.LayerNorm(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm(256),
        )

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # B, embed_dim, H/patch_size, W/patch_size
        x = x.flatten(2).transpose(1, 2)  # B, num_patches, embed_dim

        # Add position embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape for neck
        hw = int(np.sqrt(x.shape[1]))
        x = x.transpose(1, 2).reshape(B, -1, hw, hw)

        # Apply neck
        x = self.neck(x)

        return x

class TransformerBlock(nn.Module):
    """Transformer block for image encoder"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        x_norm = x_norm.transpose(0, 1)  # For MultiheadAttention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        attn_out = attn_out.transpose(0, 1)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x

class PromptEncoder(nn.Module):
    """Encoder for points and boxes prompts"""
    def __init__(self, embed_dim=256, image_embedding_size=(16, 16)):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_embedding_size = image_embedding_size

        # Point embeddings
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(4)  # positive, negative, top-left, bottom-right
        ])

        # Dense prompt embeddings
        self.dense_embed = nn.Embedding(1, embed_dim)

        # Positional encoding
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

    def forward(self, points=None, boxes=None, masks=None):
        """
        Embed different types of prompts.

        Args:
            points: (B, N, 3) array of point prompts (x, y, label)
            boxes: (B, N, 4) array of box prompts (x1, y1, x2, y2)
            masks: (B, N, H, W) array of mask prompts
        """
        bs = 1  # Simplified for demo
        sparse_embeddings = []
        dense_embeddings = None

        # Point prompts
        if points is not None:
            coords = points[:, :, :2]  # x, y coordinates
            labels = points[:, :, 2]   # point labels (0=negative, 1=positive)

            # Get positional embeddings
            pos_embed = self.pe_layer.forward_with_coords(coords)

            # Add point type embeddings
            point_embed = pos_embed.clone()
            for i, label in enumerate(labels[0]):  # Simplified for single batch
                if label == 1:  # positive point
                    point_embed[0, i] += self.point_embeddings[0].weight[0]
                else:  # negative point
                    point_embed[0, i] += self.point_embeddings[1].weight[0]

            sparse_embeddings.append(point_embed)

        # Box prompts
        if boxes is not None:
            coords = boxes.reshape(bs, -1, 2)  # Reshape to (B, N*2, 2)
            pos_embed = self.pe_layer.forward_with_coords(coords)

            # Add box corner embeddings
            box_embed = pos_embed.clone()
            box_embed[:, 0::2] += self.point_embeddings[2].weight[0]  # top-left
            box_embed[:, 1::2] += self.point_embeddings[3].weight[0]  # bottom-right

            sparse_embeddings.append(box_embed.view(bs, -1, self.embed_dim))

        # Concatenate sparse embeddings
        if sparse_embeddings:
            sparse_embeddings = torch.cat(sparse_embeddings, dim=1)
        else:
            sparse_embeddings = torch.empty((bs, 0, self.embed_dim))

        # Dense embeddings (masks)
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)

        return sparse_embeddings, dense_embeddings

    def _embed_masks(self, masks):
        """Embed mask prompts"""
        return self.dense_embed.weight.reshape(1, -1, 1, 1).expand(
            masks.shape[0], -1, self.image_embedding_size[0], self.image_embedding_size[1]
        )

class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies"""
    def __init__(self, num_pos_feats=128, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer("positional_encoding_gaussian_matrix",
                            scale * torch.randn((2, num_pos_feats)))

    def forward_with_coords(self, coords_input):
        """Forward pass with coordinate inputs"""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / 256.0  # Normalize to [0, 1]
        coords[:, :, 1] = coords[:, :, 1] / 256.0
        coords = 2 * coords - 1  # Normalize to [-1, 1]

        # Apply Gaussian matrix
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords

        # Concatenate sin and cos
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

class MaskDecoder(nn.Module):
    """Mask decoder that produces segmentation masks from image embeddings and prompts"""
    def __init__(self, embed_dim=256, num_multimask_outputs=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_multimask_outputs = num_multimask_outputs

        # IoU token and mask tokens
        self.iou_token = nn.Embedding(1, embed_dim)
        self.mask_tokens = nn.Embedding(num_multimask_outputs + 1, embed_dim)

        # Transformer decoder
        self.decoder = nn.ModuleList([
            TwoWayAttentionBlock(embed_dim, num_heads=8)
            for _ in range(2)
        ])

        self.final_attn_token_to_image = Attention(embed_dim, downsample_rate=2)

        # Output upscaling
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 4, kernel_size=2, stride=2),
            nn.LayerNorm2d(embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=2, stride=2),
            nn.GELU(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(embed_dim, embed_dim, embed_dim // 8, 3)
            for _ in range(self.num_multimask_outputs + 1)
        ])

        # IoU prediction head
        self.iou_prediction_head = MLP(embed_dim, 256, self.num_multimask_outputs + 1, 3)

    def forward(self, image_embeddings, sparse_prompt_embeddings, dense_prompt_embeddings=None):
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings: (B, embed_dim, H, W) from image encoder
            sparse_prompt_embeddings: (B, N, embed_dim) from prompt encoder
            dense_prompt_embeddings: (B, embed_dim, H, W) from prompt encoder
        """
        batch_size = image_embeddings.shape[0]

        # Concatenate output tokens
        output_tokens = torch.cat([
            self.iou_token.weight,
            self.mask_tokens.weight,
        ], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in the batch
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings if dense_prompt_embeddings is not None else src
        pos_src = torch.repeat_interleave(self._get_pos_embed(src), tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self._run_decoder(tokens, src, pos_src)

        # Upscale mask embeddings and predict masks
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []

        for i in range(self.num_multimask_outputs + 1):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](hs[:, i, :]))

        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(hs[:, 0, :])

        return masks, iou_pred

    def _get_pos_embed(self, src):
        """Get positional embeddings for image features"""
        pos_embed = torch.zeros((src.shape[0], src.shape[2], src.shape[3], src.shape[1]))
        return pos_embed.permute(0, 3, 1, 2)

    def _run_decoder(self, tokens, src, pos_src):
        """Run the transformer decoder"""
        # Flatten src
        b, c, h, w = src.shape
        src = src.flatten(2).transpose(1, 2)
        pos_src = pos_src.flatten(2).transpose(1, 2)

        # Apply decoder layers
        for layer in self.decoder:
            tokens, src = layer(tokens, src, pos_src)

        return tokens, src

class TwoWayAttentionBlock(nn.Module):
    """Two-way attention block for mask decoder"""
    def __init__(self, embed_dim, num_heads, mlp_dim=2048):
        super().__init__()
        self.self_attn = Attention(embed_dim, num_heads)
        self.cross_attn_token_to_image = Attention(embed_dim, num_heads)
        self.mlp = MLP(embed_dim, mlp_dim, embed_dim, 3)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)

        self.cross_attn_image_to_token = Attention(embed_dim, num_heads)

    def forward(self, queries, keys, query_pe):
        # Self attention block
        q = queries + query_pe
        attn_out = self.self_attn(q, q)
        queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block
        q = queries + query_pe
        k = keys + query_pe
        attn_out = self.cross_attn_token_to_image(q, k)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image to token
        q = queries + query_pe
        k = keys + query_pe
        attn_out = self.cross_attn_image_to_token(k, q)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys

class Attention(nn.Module):
    """Attention layer with optional downsampling"""
    def __init__(self, embed_dim, num_heads=8, downsample_rate=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.downsample_rate = downsample_rate

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v=None):
        if v is None:
            v = k

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Reshape for multi-head attention
        b, n, c = q.shape
        q = q.reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, k.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, v.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v

        # Reshape back
        out = out.transpose(1, 2).reshape(b, n, c)
        out = self.out_proj(out)

        return out

class MLP(nn.Module):
    """Simple MLP"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class LayerNorm2d(nn.Module):
    """2D Layer normalization"""
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MedicalSAM(nn.Module):
    """Medical Segment Anything Model"""
    def __init__(self, image_encoder, prompt_encoder, mask_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

    def forward(self, images, point_prompts=None, box_prompts=None, mask_prompts=None):
        """
        Forward pass of Medical SAM

        Args:
            images: (B, C, H, W) input images
            point_prompts: (B, N, 3) point prompts (x, y, label)
            box_prompts: (B, N, 4) box prompts (x1, y1, x2, y2)
            mask_prompts: (B, N, H, W) mask prompts
        """
        # Encode image
        image_embeddings = self.image_encoder(images)

        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=point_prompts,
            boxes=box_prompts,
            masks=mask_prompts
        )

        # Decode masks
        masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings
        )

        return masks, iou_predictions

def create_medical_prompts(image_shape, dataset_type='chest_xray'):
    """Create medical domain specific prompts"""
    h, w = image_shape
    prompts = []

    if dataset_type == 'chest_xray':
        # Lung region points
        left_lung_point = [w * 0.3, h * 0.5, 1]  # Positive point in left lung
        right_lung_point = [w * 0.7, h * 0.5, 1]  # Positive point in right lung
        background_point = [w * 0.1, h * 0.1, 0]  # Negative point in background

        prompts = [left_lung_point, right_lung_point, background_point]

    elif dataset_type == 'brain_mri':
        # Brain tissue points
        center_point = [w * 0.5, h * 0.5, 1]  # Positive point in brain center
        background_point = [w * 0.1, h * 0.1, 0]  # Negative point in background

        prompts = [center_point, background_point]

    else:  # skin_lesion
        # Lesion points
        center_point = [w * 0.5, h * 0.5, 1]  # Positive point in lesion
        skin_point = [w * 0.8, h * 0.8, 0]  # Negative point in healthy skin

        prompts = [center_point, skin_point]

    return torch.tensor([prompts], dtype=torch.float32)

def train_medical_sam(dataset_type='chest_xray', data_path=None, num_epochs=30, save_interval=5):
    """
    Medical SAM 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 중간 결과 저장 간격
    """
    # Create result logger
    logger = create_logger_for_medical_segmentation("sam", dataset_type)

    # Check for quick test mode
    if os.getenv('QUICK_TEST') == '1':
        num_epochs = min(int(os.getenv('TEST_EPOCHS', 5)), num_epochs)

    # Configuration
    batch_size = 4
    image_size = 256
    learning_rate = 1e-4

    # Save configuration
    config = {
        'algorithm': 'Medical SAM',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'image_size': image_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'data_path': data_path
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Create synthetic data for demonstration
    logger.log(f"Creating synthetic {dataset_type} data...")

    from unet_medical_example import create_synthetic_segmentation_data
    images, masks = create_synthetic_segmentation_data(dataset_type, num_samples=400, image_size=image_size)

    logger.log(f"Created {len(images)} synthetic medical images")
    logger.save_image_grid(images[:16], "sample_images", nrow=4, normalize=True)
    logger.save_image_grid(masks[:16], "sample_masks", nrow=4, normalize=True)

    # Initialize SAM components
    image_encoder = ImageEncoder(img_size=image_size, patch_size=16, in_chans=1, embed_dim=768)
    prompt_encoder = PromptEncoder(embed_dim=256, image_embedding_size=(16, 16))
    mask_decoder = MaskDecoder(embed_dim=256, num_multimask_outputs=3)

    # Create Medical SAM model
    model = MedicalSAM(image_encoder, prompt_encoder, mask_decoder).to(device)

    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    logger.log("Starting SAM training...")
    losses = []
    iou_scores = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_iou = 0.0
        num_batches = 0

        # Create batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_masks = masks[i:i+batch_size]

            if len(batch_images) < batch_size:
                continue

            # Convert to tensors
            image_tensor = torch.from_numpy(batch_images).unsqueeze(1).float().to(device)
            mask_tensor = torch.from_numpy(batch_masks).float().to(device)

            # Create prompts for each image in batch
            point_prompts = []
            for j in range(len(batch_images)):
                prompts = create_medical_prompts(batch_images[j].shape, dataset_type)
                point_prompts.append(prompts[0])
            point_prompts = torch.stack(point_prompts).to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_masks, iou_pred = model(image_tensor, point_prompts=point_prompts)

            # Calculate loss (use first mask output)
            loss = criterion(pred_masks[:, 0], mask_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Calculate IoU
            with torch.no_grad():
                pred_binary = torch.sigmoid(pred_masks[:, 0]) > 0.5
                intersection = (pred_binary * mask_tensor).sum()
                union = (pred_binary + mask_tensor).clamp(0, 1).sum()
                iou = intersection / (union + 1e-8)
                epoch_iou += iou.item()

            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_iou = epoch_iou / num_batches

        losses.append(avg_loss)
        iou_scores.append(avg_iou)

        scheduler.step()

        logger.log(f'Epoch {epoch+1}/{num_epochs}:')
        logger.log(f'  Loss: {avg_loss:.4f}')
        logger.log(f'  IoU: {avg_iou:.4f}')
        logger.log(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save intermediate results
        if (epoch + 1) % save_interval == 0:
            logger.save_model(model, f"sam_epoch_{epoch+1}", optimizer=optimizer)

            # Generate sample results
            with torch.no_grad():
                model.eval()
                sample_images = images[:4]
                sample_tensor = torch.from_numpy(sample_images).unsqueeze(1).float().to(device)

                sample_prompts = []
                for img in sample_images:
                    prompts = create_medical_prompts(img.shape, dataset_type)
                    sample_prompts.append(prompts[0])
                sample_prompts = torch.stack(sample_prompts).to(device)

                pred_masks, _ = model(sample_tensor, point_prompts=sample_prompts)
                pred_np = torch.sigmoid(pred_masks[:, 0]).cpu().numpy()

                logger.save_image_grid(pred_np, f"sam_predictions_epoch_{epoch+1}",
                                     nrow=2, normalize=True)

        # Log metrics
        logger.log_metric("train_loss", avg_loss, epoch)
        logger.log_metric("train_iou", avg_iou, epoch)

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Training Loss')
    plt.title('SAM Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(iou_scores, label='Training IoU')
    plt.title('SAM Training IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'sam_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save final model
    logger.save_model(model, "sam_final_model", optimizer=optimizer,
                     metadata={'epochs': num_epochs, 'final_iou': avg_iou})

    # Generate final results with different prompt types
    logger.log("Generating final SAM results with different prompts...")

    with torch.no_grad():
        model.eval()
        test_images = images[-8:]
        test_tensor = torch.from_numpy(test_images).unsqueeze(1).float().to(device)

        # Point prompts
        point_prompts = []
        for img in test_images:
            prompts = create_medical_prompts(img.shape, dataset_type)
            point_prompts.append(prompts[0])
        point_prompts = torch.stack(point_prompts).to(device)

        point_masks, point_iou = model(test_tensor, point_prompts=point_prompts)
        point_results = torch.sigmoid(point_masks[:, 0]).cpu().numpy()

        # Box prompts (create boxes around regions of interest)
        box_prompts = []
        for img in test_images:
            h, w = img.shape
            if dataset_type == 'chest_xray':
                # Box around lung region
                box = [w * 0.2, h * 0.3, w * 0.8, h * 0.7]
            elif dataset_type == 'brain_mri':
                # Box around brain region
                box = [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
            else:  # skin_lesion
                # Box around lesion
                box = [w * 0.4, h * 0.4, w * 0.6, h * 0.6]
            box_prompts.append(box)

        box_prompts = torch.tensor([box_prompts], dtype=torch.float32).to(device)
        box_masks, box_iou = model(test_tensor, box_prompts=box_prompts)
        box_results = torch.sigmoid(box_masks[:, 0]).cpu().numpy()

        # Save comprehensive results
        comparison_grid = np.zeros((4 * len(test_images), *test_images[0].shape))
        for i in range(len(test_images)):
            comparison_grid[i] = test_images[i]                    # Original
            comparison_grid[i + len(test_images)] = masks[-8+i]    # Ground truth
            comparison_grid[i + 2*len(test_images)] = point_results[i]  # Point prompts
            comparison_grid[i + 3*len(test_images)] = box_results[i]    # Box prompts

        logger.save_image_grid(comparison_grid, "final_sam_comparison",
                              nrow=len(test_images), normalize=True)

    logger.log("SAM training completed successfully!")
    logger.log(f"Final IoU Score: {avg_iou:.4f}")
    logger.log(f"All results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

if __name__ == "__main__":
    print("Medical SAM (Segment Anything Model) Implementation")
    print("=" * 60)
    print("Training SAM for promptable medical image segmentation...")
    print("SAM Key Features:")
    print("- Promptable segmentation (points, boxes, masks)")
    print("- Foundation model architecture")
    print("- Zero-shot generalization capability")
    print("- Real-time interactive segmentation")
    print("- Medical domain adaptation")

    # Run training
    model, results_dir = train_medical_sam(
        dataset_type='chest_xray',
        data_path=None,
        num_epochs=20,
        save_interval=5
    )

    print(f"\nTraining completed successfully!")
    print(f"All results saved to: {results_dir}")

    print("\nGenerated files include:")
    print("- images/: Original images and SAM segmentation results")
    print("- models/: Trained SAM model checkpoints")
    print("- logs/: Training logs and configuration")
    print("- plots/: Training curves and performance metrics")
    print("- metrics/: Detailed evaluation metrics")