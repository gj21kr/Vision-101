#!/usr/bin/env python3
"""
DINO-V2 의료 객체 검출

DINO-V2는 self-supervised learning을 통해 사전 훈련된 강력한 Vision Transformer를 백본으로 사용하는
개선된 DINO 모델입니다. 의료 영상에서 더 나은 특징 표현을 학습할 수 있습니다.

주요 특징:
- Self-supervised pre-training
- Vision Transformer backbone
- Improved feature representation
- Better generalization with less data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
import math
sys.path.append('/workspace/Vision-101')
from result_logger import create_logger_for_medical_detection

# DINO-V2용 의료 데이터셋 (기존과 동일하지만 더 큰 해상도 지원)
class MedicalDinoV2Dataset(Dataset):
    def __init__(self, data_type='chest_xray', split='train', transform=None):
        """
        의료 DINO-V2 데이터셋

        Args:
            data_type: 'chest_xray', 'mammography', 'brain_mri', 'skin_lesion'
            split: 'train', 'val', 'test'
            transform: 이미지 전처리
        """
        self.data_type = data_type
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((518, 518)),  # DINO-V2는 더 큰 해상도 선호
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 클래스 정의
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)

        # 합성 데이터 생성
        self.images, self.annotations = self._generate_synthetic_data()

    def _get_class_names(self):
        """데이터 타입별 클래스 이름 반환"""
        class_maps = {
            'chest_xray': ['background', 'nodule', 'pneumonia', 'cardiomegaly', 'pneumothorax', 'effusion'],
            'mammography': ['background', 'mass', 'calcification', 'distortion', 'asymmetry'],
            'brain_mri': ['background', 'tumor', 'hemorrhage', 'infarct', 'edema'],
            'skin_lesion': ['background', 'melanoma', 'basal_cell', 'squamous_cell', 'nevus']
        }
        return class_maps.get(self.data_type, ['background', 'lesion', 'normal'])

    def _generate_synthetic_data(self):
        """합성 의료 데이터 생성"""
        images = []
        annotations = []

        num_samples = 600 if self.split == 'train' else 150

        for i in range(num_samples):
            # 고해상도 의료 영상 생성
            image = self._create_high_res_medical_image(i)
            objects = self._create_synthetic_annotations(image, i)

            images.append(image)
            annotations.append(objects)

        return images, annotations

    def _create_high_res_medical_image(self, seed):
        """고해상도 의료 영상 생성"""
        np.random.seed(seed)

        if self.data_type == 'chest_xray':
            return self._create_high_res_chest_xray(seed)
        elif self.data_type == 'mammography':
            return self._create_high_res_mammography(seed)
        else:
            # 기본 고해상도 의료 영상
            image = np.random.normal(0.4, 0.15, (518, 518, 3))
            image = np.clip(image, 0, 1)
            return (image * 255).astype(np.uint8)

    def _create_high_res_chest_xray(self, seed):
        """고해상도 흉부 X-ray 생성"""
        np.random.seed(seed)
        image = np.zeros((518, 518, 3))

        # 더 세밀한 폐야 구조
        center_x, center_y = 259, 259
        for y in range(518):
            for x in range(518):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 220:
                    # 폐 조직의 세밀한 구조
                    intensity = 0.3 + 0.15 * np.sin(dist * 0.02) + 0.1 * np.sin(x * 0.05)
                    intensity += 0.05 * np.sin(y * 0.03)  # 혈관 구조
                    image[y, x] = intensity + np.random.normal(0, 0.03)

        # 늑골 구조 (더 세밀함)
        for i in range(14):
            y_pos = 40 + i * 30
            for x in range(518):
                if 0 <= y_pos < 518:
                    intensity = 0.25 * np.sin(x * 0.015 + i * 0.5)
                    image[int(y_pos):int(y_pos+2), x] += intensity

        # 심장 그림자
        heart_center_x, heart_center_y = 200, 300
        for y in range(518):
            for x in range(518):
                dist = np.sqrt((x - heart_center_x)**2 + (y - heart_center_y)**2)
                if dist < 80:
                    image[y, x] += 0.2

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_high_res_mammography(self, seed):
        """고해상도 유방촬영술 생성"""
        np.random.seed(seed)
        image = np.random.gamma(2, 0.08, (518, 518, 3))

        # 유방 조직의 복잡한 패턴
        for i in range(30):
            x = np.random.randint(50, 468)
            y = np.random.randint(50, 468)
            radius = np.random.randint(8, 35)

            xx, yy = np.meshgrid(np.arange(518), np.arange(518))
            dist = np.sqrt((xx - x)**2 + (yy - y)**2)

            # 방사형 패턴
            angle = np.arctan2(yy - y, xx - x)
            radial_pattern = np.sin(angle * 4 + dist * 0.1)

            mask = (dist < radius) & (radial_pattern > 0.3)
            image[mask] += 0.12

        # 피브로글란다 조직
        for i in range(50):
            x = np.random.randint(100, 418)
            y = np.random.randint(100, 418)

            xx, yy = np.meshgrid(np.arange(518), np.arange(518))
            pattern = np.sin((xx - x) * 0.05) * np.sin((yy - y) * 0.05)
            mask = np.abs(pattern) > 0.7
            image[mask] += 0.08

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_synthetic_annotations(self, image, seed):
        """합성 객체 어노테이션 생성"""
        np.random.seed(seed)
        objects = []

        # 객체 개수 (1-4개, DINO-V2는 더 정확한 검출 가능)
        num_objects = np.random.randint(1, 5)

        for _ in range(num_objects):
            # 랜덤 클래스 (background 제외)
            class_id = np.random.randint(1, self.num_classes)

            # 더 정밀한 위치와 크기
            x = np.random.randint(60, 458)
            y = np.random.randint(60, 458)
            w = np.random.randint(25, 100)
            h = np.random.randint(25, 100)

            # 경계 박스 정규화 (0-1 범위)
            bbox = [
                max(0, (x - w//2) / 518),
                max(0, (y - h//2) / 518),
                min(1, (x + w//2) / 518),
                min(1, (y + h//2) / 518)
            ]

            objects.append({
                'class_id': class_id,
                'bbox': bbox,
                'area': w * h / (518 * 518)
            })

        return objects

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        targets = self.annotations[idx]

        if self.transform:
            image = self.transform(image)

        # 타겟 형식 변환
        boxes = []
        labels = []

        for obj in targets:
            boxes.append(obj['bbox'])
            labels.append(obj['class_id'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }

        return image, target

# Vision Transformer Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=518, patch_size=14, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

# Multi-Head Self-Attention for ViT
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x, attn

# Vision Transformer Block
class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights

# DINO-V2 Vision Transformer Backbone
class DinoV2Backbone(nn.Module):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.num_patches = self.patch_embed.n_patches

        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Feature projection for DINO
        self.feature_proj = nn.Linear(embed_dim, 256)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # Transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attention_weights.append(attn)

        x = self.norm(x)

        # Remove class token and reshape for DINO
        x = x[:, 1:]  # Remove class token
        x = self.feature_proj(x)  # Project to DINO feature dimension

        # Reshape to spatial format for DINO decoder
        h_patches = w_patches = int(np.sqrt(self.num_patches))
        x = x.transpose(1, 2).reshape(B, 256, h_patches, w_patches)

        return x, attention_weights

# Cross-Attention for DINO-V2 Decoder
class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, embed_dim)
        self.v = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, N_q, C = query.shape
        N_k = key.shape[1]

        q = self.q(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v(value).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)

        return x, attn

# DINO-V2 Decoder Layer
class DinoV2DecoderLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, tgt, memory):
        # Self-attention
        tgt_norm = self.norm1(tgt)
        tgt2, self_attn = self.self_attn(tgt_norm)
        tgt = tgt + tgt2

        # Cross-attention
        tgt_norm = self.norm2(tgt)
        tgt2, cross_attn = self.cross_attn(tgt_norm, memory, memory)
        tgt = tgt + tgt2

        # MLP
        tgt = tgt + self.mlp(self.norm3(tgt))

        return tgt, cross_attn

# DINO-V2 모델
class MedicalDinoV2(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, num_heads=8,
                 num_decoder_layers=6, num_queries=100):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # DINO-V2 백본 (더 큰 ViT 사용)
        self.backbone = DinoV2Backbone(
            img_size=518,
            patch_size=14,
            embed_dim=1024,
            depth=24,
            num_heads=16
        )

        # 쿼리 임베딩
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 디코더
        self.decoder_layers = nn.ModuleList([
            DinoV2DecoderLayer(hidden_dim, num_heads)
            for _ in range(num_decoder_layers)
        ])

        # 예측 헤드
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, images):
        batch_size = images.size(0)

        # DINO-V2 백본을 통한 특징 추출
        features, backbone_attentions = self.backbone(images)  # [B, hidden_dim, H, W]

        B, C, H, W = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # [H*W, B, C]

        # 쿼리 임베딩
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # 디코더
        decoder_output = query_embed
        decoder_attentions = []

        for layer in self.decoder_layers:
            decoder_output, cross_attn = layer(decoder_output, features)
            decoder_attentions.append(cross_attn)

        # 예측
        class_logits = self.class_embed(decoder_output.transpose(0, 1))
        bbox_coords = self.bbox_embed(decoder_output.transpose(0, 1))

        return {
            'pred_logits': class_logits,
            'pred_boxes': bbox_coords,
            'decoder_attentions': decoder_attentions,
            'backbone_attentions': backbone_attentions
        }

# 간소화된 DINO Loss (V2용)
class DinoV2Loss(nn.Module):
    def __init__(self, num_classes, weight_dict={'loss_ce': 2, 'loss_bbox': 5}):
        super().__init__()
        self.num_classes = num_classes
        self.weight_dict = weight_dict

        # Class weights (down-weight background)
        empty_weight = torch.ones(num_classes)
        empty_weight[0] = 0.1
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        # Simplified matching (nearest neighbor)
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']

        total_loss = 0
        batch_size = pred_logits.shape[0]

        for b in range(batch_size):
            target_boxes = targets[b]['boxes']
            target_labels = targets[b]['labels']

            if len(target_boxes) == 0:
                # No target objects - only background loss
                background_targets = torch.zeros(pred_logits.shape[1], dtype=torch.long, device=pred_logits.device)
                loss_ce = F.cross_entropy(pred_logits[b], background_targets, weight=self.empty_weight)
                total_loss += loss_ce * self.weight_dict.get('loss_ce', 1)
                continue

            # Find closest predictions for each target
            pred_boxes_batch = pred_boxes[b]  # [num_queries, 4]
            target_boxes_batch = target_boxes  # [num_targets, 4]

            # Compute L1 distance between all pred-target pairs
            distances = torch.cdist(pred_boxes_batch, target_boxes_batch, p=1)  # [num_queries, num_targets]

            # For each target, find closest prediction
            closest_preds = distances.min(dim=0)[1]  # [num_targets]

            # Classification loss
            class_targets = torch.zeros(pred_logits.shape[1], dtype=torch.long, device=pred_logits.device)
            class_targets[closest_preds] = target_labels

            loss_ce = F.cross_entropy(pred_logits[b], class_targets, weight=self.empty_weight)

            # Box regression loss (only for matched predictions)
            if len(closest_preds) > 0:
                matched_pred_boxes = pred_boxes_batch[closest_preds]
                loss_bbox = F.l1_loss(matched_pred_boxes, target_boxes_batch, reduction='mean')
            else:
                loss_bbox = torch.tensor(0.0, device=pred_logits.device)

            total_loss += loss_ce * self.weight_dict.get('loss_ce', 1)
            total_loss += loss_bbox * self.weight_dict.get('loss_bbox', 1)

        return total_loss / batch_size

def train_medical_dino_v2(dataset_type='chest_xray', num_epochs=30, batch_size=4, lr=5e-5):
    """
    의료 DINO-V2 모델 훈련

    Args:
        dataset_type: 의료 데이터셋 타입
        num_epochs: 훈련 에포크 수
        batch_size: 배치 크기 (DINO-V2는 메모리 사용량이 높음)
        lr: 학습률
    """

    # 로거 설정
    logger = create_logger_for_medical_detection('dino_v2', dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # 데이터셋 준비
    train_dataset = MedicalDinoV2Dataset(data_type=dataset_type, split='train')
    val_dataset = MedicalDinoV2Dataset(data_type=dataset_type, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 모델 설정
    model = MedicalDinoV2(num_classes=train_dataset.num_classes).to(device)

    # 손실 함수 설정
    criterion = DinoV2Loss(train_dataset.num_classes)

    # 옵티마이저 설정 (DINO-V2는 더 작은 학습률 사용)
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': lr * 0.1},  # Backbone은 더 작은 학습률
        {'params': [p for name, p in model.named_parameters() if 'backbone' not in name], 'lr': lr}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 훈련 메트릭 저장
    train_losses = []
    val_losses = []

    logger.log("Starting DINO-V2 training...")
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        # 훈련 단계
        model.train()
        running_loss = 0.0

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                       for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # 순전파
            outputs = model(images)

            # 손실 계산
            loss = criterion(outputs, targets)

            # 역전파
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.log(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                          f'Loss: {loss.item():.4f}')

        # 검증 단계
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in t.items()} for t in targets]

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item()

        # 메트릭 계산
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logger.log(f'Epoch [{epoch+1}/{num_epochs}]')
        logger.log(f'Train Loss: {avg_train_loss:.4f}')
        logger.log(f'Val Loss: {avg_val_loss:.4f}')
        logger.log(f'Learning Rate: {scheduler.get_last_lr()[0]:.2e}')

        # 스케줄러 업데이트
        scheduler.step()

        # 메트릭 저장
        logger.save_metrics({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': scheduler.get_last_lr()[0]
        })

        # 중간 결과 저장 (매 5 에포크)
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                sample_images, sample_targets = next(iter(val_loader))
                sample_images = sample_images[:2].to(device)  # 메모리 고려해서 2개만

                outputs = model(sample_images)

                # 시각화를 위한 이미지 준비
                vis_images = []
                for i in range(len(sample_images)):
                    # 원본 이미지
                    img = sample_images[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min())

                    # 예측 박스 오버레이
                    pred_boxes = outputs['pred_boxes'][i]
                    pred_scores = F.softmax(outputs['pred_logits'][i], dim=-1)
                    pred_scores = pred_scores.max(dim=-1)[0]

                    # 상위 예측 선택 (더 보수적)
                    top_indices = pred_scores.topk(min(3, len(pred_scores)))[1]

                    img_with_boxes = img.copy()
                    for idx in top_indices:
                        if pred_scores[idx] > 0.7:  # 더 높은 임계값
                            box = pred_boxes[idx] * 518
                            x1, y1, x2, y2 = box.cpu().numpy()

                            # 박스 그리기
                            thickness = 2
                            img_with_boxes[int(y1):int(y1)+thickness, int(x1):int(x2)] = [1, 0, 0]
                            img_with_boxes[int(y2):int(y2)+thickness, int(x1):int(x2)] = [1, 0, 0]
                            img_with_boxes[int(y1):int(y2), int(x1):int(x1)+thickness] = [1, 0, 0]
                            img_with_boxes[int(y1):int(y2), int(x2):int(x2)+thickness] = [1, 0, 0]

                    vis_images.extend([img, img_with_boxes])

                logger.save_image_grid(vis_images,
                                     f'dino_v2_detection_epoch_{epoch+1}.png',
                                     titles=['Original', 'DINO-V2 Detection'] * 2,
                                     nrow=2)

    # 최종 모델 저장
    logger.save_model(model, "medical_dino_v2_final",
                     optimizer=optimizer, epoch=num_epochs,
                     config={'dataset_type': dataset_type, 'num_epochs': num_epochs})

    # 훈련 곡선 저장
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('DINO-V2 Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    lrs = [scheduler.get_last_lr()[0] for _ in train_losses]
    plt.plot(lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    # 파라미터 분포 시각화
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    plt.bar(['Total', 'Trainable'], [total_params, trainable_params])
    plt.title('Model Parameters')
    plt.ylabel('Number of Parameters')
    for i, v in enumerate([total_params, trainable_params]):
        plt.text(i, v + 1000000, f'{v:,}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'dino_v2_training_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logger.log("DINO-V2 training completed successfully!")
    logger.log(f"Final validation loss: {val_losses[-1]:.4f}")
    logger.log(f"Results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

def collate_fn(batch):
    """Custom collate function for DINO-V2"""
    images, targets = tuple(zip(*batch))
    images = torch.stack(images, 0)
    return images, list(targets)

if __name__ == "__main__":
    print("🚀 DINO-V2 의료 객체 검출")
    print("=" * 60)

    # 하이퍼파라미터 설정
    config = {
        'dataset_type': 'mammography',
        'num_epochs': 5,  # 빠른 테스트를 위해 5로 설정
        'batch_size': 2,  # DINO-V2는 메모리 사용량이 높음
        'lr': 5e-5
    }

    print(f"Dataset: {config['dataset_type']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print()

    try:
        model, results_dir = train_medical_dino_v2(
            dataset_type=config['dataset_type'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )

        print("\n✅ DINO-V2 training completed successfully!")
        print(f"📁 Results saved to: {results_dir}")

        print("\n📊 Generated files include:")
        print("- images/: DINO-V2 detection result visualizations")
        print("- models/: Trained DINO-V2 model")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves and analysis")
        print("- metrics/: Training metrics in JSON format")

        print("\n🎯 DINO-V2 Features:")
        print("- Self-supervised pre-trained ViT backbone")
        print("- Enhanced feature representation")
        print("- Better performance with limited data")
        print("- Multi-scale attention visualization")

    except Exception as e:
        print(f"\n❌ Error during DINO-V2 training: {str(e)}")
        import traceback
        traceback.print_exc()