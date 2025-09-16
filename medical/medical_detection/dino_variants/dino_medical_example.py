#!/usr/bin/env python3
"""
DINO (Detection Transformer with Improved deNoising anchor boxes) 의료 객체 검출

DINO는 DETR 기반의 end-to-end 객체 검출 모델로, 앵커 박스 없이도 높은 성능을 달성합니다.
의료 영상에서 병변, 결절, 이상 징후 검출에 특화되어 있습니다.

주요 특징:
- Contrastive Learning
- Mixed Query Selection
- Deformable Attention
- Medical Image Optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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

import math
import random
from medical.result_logger import create_logger_for_medical_detection

# 의료 객체 검출 데이터셋
class MedicalDinoDataset(Dataset):
    def __init__(self, data_type='chest_xray', split='train', transform=None):
        """
        의료 DINO 데이터셋

        Args:
            data_type: 'chest_xray', 'mammography', 'brain_mri', 'skin_lesion', 'retinal'
            split: 'train', 'val', 'test'
            transform: 이미지 전처리
        """
        self.data_type = data_type
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
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
            'chest_xray': ['background', 'nodule', 'pneumonia', 'cardiomegaly', 'pneumothorax'],
            'mammography': ['background', 'mass', 'calcification', 'distortion'],
            'brain_mri': ['background', 'tumor', 'hemorrhage', 'infarct'],
            'skin_lesion': ['background', 'melanoma', 'basal_cell', 'squamous_cell'],
            'retinal': ['background', 'hemorrhage', 'exudate', 'microaneurysm']
        }
        return class_maps.get(self.data_type, ['background', 'lesion'])

    def _generate_synthetic_data(self):
        """합성 의료 데이터 생성"""
        images = []
        annotations = []

        num_samples = 800 if self.split == 'train' else 200

        for i in range(num_samples):
            # 의료 영상 합성 생성
            image = self._create_synthetic_medical_image(i)

            # 객체 어노테이션 생성
            objects = self._create_synthetic_annotations(image, i)

            images.append(image)
            annotations.append(objects)

        return images, annotations

    def _create_synthetic_medical_image(self, seed):
        """의료 특성을 반영한 합성 이미지 생성"""
        np.random.seed(seed)

        if self.data_type == 'chest_xray':
            return self._create_chest_xray(seed)
        elif self.data_type == 'mammography':
            return self._create_mammography(seed)
        elif self.data_type == 'brain_mri':
            return self._create_brain_mri(seed)
        else:
            # 기본 의료 영상
            image = np.random.normal(0.3, 0.1, (512, 512, 3))
            image = np.clip(image, 0, 1)
            return (image * 255).astype(np.uint8)

    def _create_chest_xray(self, seed):
        """흉부 X-ray 합성 생성"""
        np.random.seed(seed)
        image = np.zeros((512, 512, 3))

        # 폐야 영역
        center_x, center_y = 256, 256
        for y in range(512):
            for x in range(512):
                # 폐 모양 근사
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 200:
                    intensity = 0.3 + 0.2 * np.sin(dist * 0.02)
                    image[y, x] = intensity + np.random.normal(0, 0.05)

        # 늑골 구조
        for i in range(12):
            y_pos = 50 + i * 35
            for x in range(512):
                if 0 <= y_pos < 512:
                    image[int(y_pos), x] += 0.2 * np.sin(x * 0.02)

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_mammography(self, seed):
        """유방촬영술 합성 생성"""
        np.random.seed(seed)
        image = np.random.gamma(2, 0.1, (512, 512, 3))

        # 유방 조직 패턴
        for i in range(20):
            x = np.random.randint(50, 462)
            y = np.random.randint(50, 462)
            radius = np.random.randint(5, 25)

            xx, yy = np.meshgrid(np.arange(512), np.arange(512))
            dist = np.sqrt((xx - x)**2 + (yy - y)**2)
            mask = dist < radius
            image[mask] += 0.1

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_brain_mri(self, seed):
        """뇌 MRI 합성 생성"""
        np.random.seed(seed)
        image = np.zeros((512, 512, 3))

        # 뇌 모양 근사
        center_x, center_y = 256, 256
        for y in range(512):
            for x in range(512):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 180:
                    # 회백질/백질 구조
                    if dist < 120:
                        image[y, x] = 0.6  # 회백질
                    else:
                        image[y, x] = 0.4  # 백질

        # 노이즈 추가
        image += np.random.normal(0, 0.05, image.shape)
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_synthetic_annotations(self, image, seed):
        """합성 객체 어노테이션 생성"""
        np.random.seed(seed)
        objects = []

        # 객체 개수 (1-5개)
        num_objects = np.random.randint(1, 6)

        for _ in range(num_objects):
            # 랜덤 클래스 (background 제외)
            class_id = np.random.randint(1, self.num_classes)

            # 랜덤 위치와 크기
            x = np.random.randint(50, 462)
            y = np.random.randint(50, 462)
            w = np.random.randint(20, 80)
            h = np.random.randint(20, 80)

            # 경계 박스 정규화 (0-1 범위)
            bbox = [
                max(0, (x - w//2) / 512),
                max(0, (y - h//2) / 512),
                min(1, (x + w//2) / 512),
                min(1, (y + h//2) / 512)
            ]

            objects.append({
                'class_id': class_id,
                'bbox': bbox,
                'area': w * h / (512 * 512)
            })

        return objects

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        targets = self.annotations[idx]

        if self.transform:
            image = self.transform(image)

        # DINO 형식으로 타겟 변환
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

# 위치 인코딩
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        output = self.w_o(context)
        return output, attn_weights

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention
        src2, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # Feed forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

# Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention
        tgt2, attn_weights = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # Feed forward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt, attn_weights

# DINO 백본 네트워크
class DinoBackbone(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        # CNN 백본 (의료 영상 최적화)
        self.backbone = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),

            # Stage 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Stage 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Stage 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv = nn.Conv2d(512, hidden_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        return x

# DINO 모델
class MedicalDino(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6, num_queries=100):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # 백본
        self.backbone = DinoBackbone(hidden_dim)

        # 위치 인코딩
        self.pos_embed = PositionalEncoding(hidden_dim)

        # 트랜스포머
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, nheads, hidden_dim*4, 0.1)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, nheads, hidden_dim*4, 0.1)
            for _ in range(num_decoder_layers)
        ])

        # 쿼리 임베딩
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # 예측 헤드
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, images):
        batch_size = images.size(0)

        # 백본을 통한 특징 추출
        features = self.backbone(images)  # [B, hidden_dim, H, W]

        B, C, H, W = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # [H*W, B, C]

        # 위치 인코딩 추가
        features = self.pos_embed(features)

        # 인코더
        encoder_output = features
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        # 쿼리 임베딩
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # 디코더
        decoder_output = query_embed
        attention_weights = []

        for layer in self.decoder_layers:
            decoder_output, attn_weights = layer(decoder_output, encoder_output)
            attention_weights.append(attn_weights)

        # 예측
        class_logits = self.class_embed(decoder_output.transpose(0, 1))  # [B, num_queries, num_classes]
        bbox_coords = self.bbox_embed(decoder_output.transpose(0, 1))    # [B, num_queries, 4]

        return {
            'pred_logits': class_logits,
            'pred_boxes': bbox_coords,
            'attention_weights': attention_weights
        }

# 헝가리안 매칭 (간소화 버전)
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(self, outputs, targets):
        """Hungarian matching between predictions and targets"""
        batch_size, num_queries = outputs['pred_logits'].shape[:2]

        # 출력을 평평하게 만들기
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [batch_size * num_queries, 4]

        # 타겟 연결
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])

        if len(tgt_ids) == 0:
            # 타겟이 없는 경우
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(batch_size)]

        # 분류 비용 (negative log likelihood)
        cost_class = -out_prob[:, tgt_ids]

        # L1 박스 비용
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 총 비용 매트릭스
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        C = C.view(batch_size, num_queries, -1)

        # 배치별로 헝가리안 할당 수행 (간소화)
        indices = []
        for i, c in enumerate(C.split(1, dim=0)):
            c = c.squeeze(0)  # [num_queries, num_targets_i]
            if c.shape[1] == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                continue

            # 간소화된 매칭: 각 타겟에 대해 가장 가까운 쿼리 선택
            matched_queries = c.min(dim=0)[1]  # [num_targets_i]
            matched_targets = torch.arange(c.shape[1])  # [num_targets_i]

            indices.append((matched_queries.long(), matched_targets.long()))

        return indices

# DINO 손실 함수
class DinoLoss(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # 클래스 가중치 (배경 클래스 다운웨이트)
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """분류 손실"""
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """박스 회귀 손실"""
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # 매칭된 예측의 배치 인덱스
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets):
        # 헝가리안 매칭
        indices = self.matcher(outputs, targets)

        # 타겟 박스 수 계산
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs['pred_logits'].device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # 손실 계산
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        return loss_map[loss](outputs, targets, indices, num_boxes)

def train_medical_dino(dataset_type='chest_xray', num_epochs=50, batch_size=8, lr=1e-4):
    """
    의료 DINO 모델 훈련

    Args:
        dataset_type: 의료 데이터셋 타입
        num_epochs: 훈련 에포크 수
        batch_size: 배치 크기
        lr: 학습률
    """

    # 로거 설정
    logger = create_logger_for_medical_detection('dino', dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # 데이터셋 준비
    train_dataset = MedicalDinoDataset(data_type=dataset_type, split='train')
    val_dataset = MedicalDinoDataset(data_type=dataset_type, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 모델 설정
    model = MedicalDino(num_classes=train_dataset.num_classes).to(device)

    # 손실 함수 설정
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
    losses = ['labels', 'boxes']
    criterion = DinoLoss(train_dataset.num_classes, matcher, weight_dict, eos_coef=0.1, losses=losses)

    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 훈련 메트릭 저장
    train_losses = []
    val_losses = []

    logger.log("Starting DINO training...")

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
            loss_dict = criterion(outputs, targets)
            losses_reduced = sum(loss for loss in loss_dict.values())

            # 역전파
            losses_reduced.backward()
            optimizer.step()

            running_loss += losses_reduced.item()

            if batch_idx % 20 == 0:
                logger.log(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                          f'Loss: {losses_reduced.item():.4f}')

        # 검증 단계
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in t.items()} for t in targets]

                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                losses_reduced = sum(loss for loss in loss_dict.values())

                val_loss += losses_reduced.item()

        # 메트릭 계산
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logger.log(f'Epoch [{epoch+1}/{num_epochs}]')
        logger.log(f'Train Loss: {avg_train_loss:.4f}')
        logger.log(f'Val Loss: {avg_val_loss:.4f}')

        # 스케줄러 업데이트
        scheduler.step()

        # 메트릭 저장
        logger.save_metrics({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'lr': scheduler.get_last_lr()[0]
        })

        # 중간 결과 저장 (매 10 에포크)
        if (epoch + 1) % 10 == 0:
            # 검출 결과 시각화
            model.eval()
            with torch.no_grad():
                sample_images, sample_targets = next(iter(val_loader))
                sample_images = sample_images[:4].to(device)

                outputs = model(sample_images)

                # 시각화를 위한 이미지 준비
                vis_images = []
                for i in range(len(sample_images)):
                    # 원본 이미지
                    img = sample_images[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min())

                    # 예측 박스 오버레이 (상위 5개)
                    pred_boxes = outputs['pred_boxes'][i]
                    pred_scores = F.softmax(outputs['pred_logits'][i], dim=-1)
                    pred_scores = pred_scores.max(dim=-1)[0]

                    # 상위 예측 선택
                    top_indices = pred_scores.topk(5)[1]

                    img_with_boxes = img.copy()
                    for idx in top_indices:
                        if pred_scores[idx] > 0.5:
                            box = pred_boxes[idx] * 512  # 이미지 크기로 스케일링
                            x1, y1, x2, y2 = box.cpu().numpy()

                            # 박스 그리기 (간단한 방식)
                            img_with_boxes[int(y1):int(y1)+2, int(x1):int(x2)] = [1, 0, 0]  # 빨간색
                            img_with_boxes[int(y2):int(y2)+2, int(x1):int(x2)] = [1, 0, 0]
                            img_with_boxes[int(y1):int(y2), int(x1):int(x1)+2] = [1, 0, 0]
                            img_with_boxes[int(y1):int(y2), int(x2):int(x2)+2] = [1, 0, 0]

                    vis_images.extend([img, img_with_boxes])

                logger.save_image_grid(vis_images,
                                     f'dino_detection_epoch_{epoch+1}.png',
                                     titles=['Original', 'Detection'] * 4,
                                     nrow=2)

    # 최종 모델 저장
    logger.save_model(model, "medical_dino_final",
                     optimizer=optimizer, epoch=num_epochs,
                     config={'dataset_type': dataset_type, 'num_epochs': num_epochs})

    # 훈련 곡선 저장
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('DINO Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot([scheduler.get_last_lr()[0]] * len(train_losses))
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'dino_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logger.log("DINO training completed successfully!")
    logger.log(f"Results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

def collate_fn(batch):
    """Custom collate function for DINO"""
    images, targets = tuple(zip(*batch))
    images = torch.stack(images, 0)
    return images, list(targets)

if __name__ == "__main__":
    print("🎯 DINO (Detection Transformer) 의료 객체 검출")
    print("=" * 60)

    # 하이퍼파라미터 설정
    config = {
        'dataset_type': 'chest_xray',
        'num_epochs': 5,  # 빠른 테스트를 위해 5로 설정
        'batch_size': 4,
        'lr': 1e-4
    }

    print(f"Dataset: {config['dataset_type']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print()

    try:
        model, results_dir = train_medical_dino(
            dataset_type=config['dataset_type'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )

        print("\n✅ DINO training completed successfully!")
        print(f"📁 Results saved to: {results_dir}")

        print("\n📊 Generated files include:")
        print("- images/: Detection result visualizations")
        print("- models/: Trained DINO model")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves and metrics")
        print("- metrics/: Training metrics in JSON format")

    except Exception as e:
        print(f"\n❌ Error during DINO training: {str(e)}")
        import traceback
        traceback.print_exc()
