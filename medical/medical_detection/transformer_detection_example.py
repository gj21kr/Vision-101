#!/usr/bin/env python3
"""
Transformer 기반 의료 객체 검출 (Medical Object Detection with Transformers)

Vision Transformer를 활용한 의료 영상에서의 객체 검출 시스템입니다.
DETR (DEtection TRansformer) 아키텍처를 의료 영상에 특화하여 구현하였습니다.

주요 기능:
- Vision Transformer 백본
- Set-based Object Detection
- Medical-specific Data Augmentation
- Multi-scale Feature Processing
- Hungarian Matching Algorithm
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
from scipy.optimize import linear_sum_assignment
from medical.result_logger import create_logger_for_medical_detection

class MedicalTransformerDetectionDataset(Dataset):
    def __init__(self, data_type='chest_xray', split='train', transform=None):
        """
        Transformer 검출용 의료 데이터셋

        Args:
            data_type: 'chest_xray', 'mammography', 'brain_mri', 'ct_scan'
            split: 'train', 'val', 'test'
            transform: 이미지 전처리
        """
        self.data_type = data_type
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((384, 384)),  # Transformer는 더 큰 해상도 선호
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 클래스 정의
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)

        # 합성 데이터 생성
        self.images, self.annotations = self._generate_medical_detection_data()

    def _get_class_names(self):
        """의료 영상별 클래스 정의"""
        class_maps = {
            'chest_xray': ['background', 'nodule', 'mass', 'pneumonia', 'cardiomegaly', 'effusion'],
            'mammography': ['background', 'mass', 'calcification', 'architectural_distortion', 'asymmetry'],
            'brain_mri': ['background', 'tumor', 'hemorrhage', 'infarct', 'lesion'],
            'ct_scan': ['background', 'nodule', 'mass', 'consolidation', 'ground_glass']
        }
        return class_maps.get(self.data_type, ['background', 'lesion'])

    def _generate_medical_detection_data(self):
        """의료 검출 데이터 생성"""
        images = []
        annotations = []

        num_samples = 600 if self.split == 'train' else 150

        for i in range(num_samples):
            # 의료 영상 생성
            image = self._create_medical_image(i)

            # 객체 어노테이션 생성
            objects = self._create_object_annotations(image, i)

            images.append(image)
            annotations.append(objects)

        return images, annotations

    def _create_medical_image(self, seed):
        """의료 영상 생성"""
        np.random.seed(seed)

        if self.data_type == 'chest_xray':
            return self._create_chest_xray_image(seed)
        elif self.data_type == 'mammography':
            return self._create_mammography_image(seed)
        elif self.data_type == 'brain_mri':
            return self._create_brain_mri_image(seed)
        else:  # ct_scan
            return self._create_ct_scan_image(seed)

    def _create_chest_xray_image(self, seed):
        """흉부 X-ray 영상 생성"""
        np.random.seed(seed)
        image = np.zeros((384, 384, 3))

        # 폐야 구조
        center_x, center_y = 192, 200
        for y in range(384):
            for x in range(384):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < 150:
                    intensity = 0.25 + 0.1 * np.sin(dist * 0.03)
                    image[y, x] = intensity + np.random.normal(0, 0.03)

        # 심장 그림자
        heart_x, heart_y = 170, 230
        for y in range(200, 280):
            for x in range(140, 200):
                dist = np.sqrt((x - heart_x)**2 + (y - heart_y)**2)
                if dist < 35:
                    image[y, x] += 0.4

        # 늑골 구조
        for rib in range(12):
            y_pos = 80 + rib * 22
            for x in range(384):
                if 0 <= y_pos < 384:
                    rib_intensity = 0.15 * np.sin(x * 0.02 + rib * 0.5)
                    image[int(y_pos):int(y_pos)+2, x] += rib_intensity

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_mammography_image(self, seed):
        """유방촬영술 영상 생성"""
        np.random.seed(seed)
        image = np.random.gamma(2, 0.08, (384, 384, 3))

        # 유방 조직 패턴
        for i in range(40):
            x = np.random.randint(50, 334)
            y = np.random.randint(50, 334)
            radius = np.random.randint(8, 30)

            xx, yy = np.meshgrid(np.arange(384), np.arange(384))
            dist = np.sqrt((xx - x)**2 + (yy - y)**2)

            # 방사형 패턴
            angle = np.arctan2(yy - y, xx - x)
            radial_pattern = np.sin(angle * 3 + dist * 0.1)

            mask = (dist < radius) & (radial_pattern > 0.2)
            image[mask] += 0.1

        # 피브로글란다 조직
        for i in range(60):
            x = np.random.randint(80, 304)
            y = np.random.randint(80, 304)

            xx, yy = np.meshgrid(np.arange(384), np.arange(384))
            pattern = np.sin((xx - x) * 0.05) * np.sin((yy - y) * 0.05)
            mask = np.abs(pattern) > 0.6
            image[mask] += 0.06

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_brain_mri_image(self, seed):
        """뇌 MRI 영상 생성"""
        np.random.seed(seed)
        image = np.zeros((384, 384, 3))

        # 뇌 윤곽
        center_x, center_y = 192, 192
        brain_radius = 140

        for y in range(384):
            for x in range(384):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist < brain_radius:
                    if dist < 100:
                        image[y, x] = 0.6 + np.random.normal(0, 0.05)  # 회백질
                    else:
                        image[y, x] = 0.4 + np.random.normal(0, 0.05)  # 백질

        # 뇌실 구조
        ventricle_centers = [(170, 170), (214, 170), (192, 140), (192, 200)]
        for vx, vy in ventricle_centers:
            for y in range(max(0, vy-15), min(384, vy+15)):
                for x in range(max(0, vx-15), min(384, vx+15)):
                    dist = np.sqrt((x - vx)**2 + (y - vy)**2)
                    if dist < 12:
                        image[y, x] = 0.1

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_ct_scan_image(self, seed):
        """CT 스캔 영상 생성"""
        np.random.seed(seed)
        image = np.random.normal(0.35, 0.08, (384, 384, 3))

        # 장기 구조 (폐, 간, 심장 등)
        # 폐야
        for lung_side in [(-60, 0), (60, 0)]:
            lung_x = 192 + lung_side[0]
            lung_y = 180

            for y in range(100, 260):
                for x in range(max(0, lung_x-50), min(384, lung_x+50)):
                    dist = np.sqrt((x - lung_x)**2 + (y - lung_y)**2)
                    if dist < 45:
                        image[y, x] = 0.15 + np.random.normal(0, 0.02)

        # 심장/종격동
        for y in range(150, 220):
            for x in range(160, 224):
                image[y, x] = 0.5 + np.random.normal(0, 0.05)

        # 간
        for y in range(220, 320):
            for x in range(120, 280):
                if np.random.random() > 0.3:
                    image[y, x] = 0.45 + np.random.normal(0, 0.03)

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_object_annotations(self, image, seed):
        """객체 어노테이션 생성"""
        np.random.seed(seed)
        objects = []

        # 객체 개수 (1-4개)
        num_objects = np.random.randint(1, 5)

        for _ in range(num_objects):
            # 클래스 (background 제외)
            class_id = np.random.randint(1, self.num_classes)

            # 위치와 크기 (의료 객체 특성 고려)
            if self.data_type == 'chest_xray':
                # 폐야 영역에 주로 위치
                x = np.random.randint(80, 304)
                y = np.random.randint(100, 280)
                w = np.random.randint(20, 60)
                h = np.random.randint(20, 60)
            elif self.data_type == 'mammography':
                # 유방 조직 영역
                x = np.random.randint(100, 284)
                y = np.random.randint(80, 304)
                w = np.random.randint(15, 45)
                h = np.random.randint(15, 45)
            elif self.data_type == 'brain_mri':
                # 뇌 영역 내부
                x = np.random.randint(120, 264)
                y = np.random.randint(120, 264)
                w = np.random.randint(15, 40)
                h = np.random.randint(15, 40)
            else:  # ct_scan
                x = np.random.randint(80, 304)
                y = np.random.randint(100, 284)
                w = np.random.randint(20, 50)
                h = np.random.randint(20, 50)

            # 경계 박스 정규화 (0-1 범위)
            bbox = [
                max(0, (x - w//2) / 384),
                max(0, (y - h//2) / 384),
                min(1, (x + w//2) / 384),
                min(1, (y + h//2) / 384)
            ]

            # 면적 계산
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            objects.append({
                'class_id': class_id,
                'bbox': bbox,
                'area': area
            })

        return objects

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        targets = self.annotations[idx]

        if self.transform:
            image = self.transform(image)

        # DETR 형식으로 타겟 변환
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

# Vision Transformer 패치 임베딩
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, embed_dim=768):
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

# 다중 헤드 어텐션
class MultiHeadAttention(nn.Module):
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

# Vision Transformer 블록
class VisionTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
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
        # 셀프 어텐션
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights

# Vision Transformer 백본
class VisionTransformerBackbone(nn.Module):
    def __init__(self, img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        self.num_patches = self.patch_embed.n_patches

        # 위치 임베딩
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer 블록들
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 초기화
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # 클래스 토큰 추가
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # Transformer 블록 통과
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attention_weights.append(attn)

        x = self.norm(x)

        # 클래스 토큰 제거하고 공간 구조 복원
        x = x[:, 1:]  # 클래스 토큰 제거
        h_patches = w_patches = int(np.sqrt(self.num_patches))
        x = x.transpose(1, 2).reshape(B, -1, h_patches, w_patches)

        return x, attention_weights

# Transformer 디코더 레이어
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 셀프 어텐션
        tgt2, self_attn_weights = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)

        # 크로스 어텐션
        tgt2, cross_attn_weights = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)

        return tgt, self_attn_weights, cross_attn_weights

# Medical DETR 모델
class MedicalDETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6,
                 num_decoder_layers=6, num_queries=100):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # Vision Transformer 백본
        self.backbone = VisionTransformerBackbone(
            img_size=384,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12
        )

        # 특징 차원 맞추기
        self.input_proj = nn.Conv2d(768, hidden_dim, kernel_size=1)

        # 쿼리 임베딩
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # Transformer 디코더
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, nheads)
            for _ in range(num_decoder_layers)
        ])

        # 예측 헤드
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no object
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()
        )

        # 의료 영상 특화 초기화
        self._init_medical_weights()

    def _init_medical_weights(self):
        """의료 영상 특화 가중치 초기화"""
        # 클래스 임베딩 편향 조정 (의료 영상은 배경이 많음)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes + 1) * bias_value
        self.class_embed.bias.data[-1] = -bias_value  # no object class

    def forward(self, images):
        batch_size = images.size(0)

        # 백본을 통한 특징 추출
        features, backbone_attentions = self.backbone(images)  # [B, C, H, W]

        # 특징 차원 조정
        features = self.input_proj(features)  # [B, hidden_dim, H, W]

        B, C, H, W = features.shape
        features = features.flatten(2).permute(2, 0, 1)  # [H*W, B, C]

        # 쿼리 임베딩
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)

        # 디코더
        decoder_output = query_embed
        decoder_attentions = []

        for layer in self.decoder_layers:
            decoder_output, self_attn, cross_attn = layer(
                decoder_output.transpose(0, 1),
                features.transpose(0, 1)
            )
            decoder_output = decoder_output.transpose(0, 1)
            decoder_attentions.append({'self_attn': self_attn, 'cross_attn': cross_attn})

        # 예측
        class_logits = self.class_embed(decoder_output.transpose(0, 1))  # [B, num_queries, num_classes+1]
        bbox_coords = self.bbox_embed(decoder_output.transpose(0, 1))    # [B, num_queries, 4]

        return {
            'pred_logits': class_logits,
            'pred_boxes': bbox_coords,
            'backbone_attentions': backbone_attentions,
            'decoder_attentions': decoder_attentions
        }

# 헝가리안 매처 (간소화 버전)
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def forward(self, outputs, targets):
        """Hungarian matching"""
        batch_size, num_queries = outputs['pred_logits'].shape[:2]

        # 출력 평탄화
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs['pred_boxes'].flatten(0, 1)  # [batch_size * num_queries, 4]

        # 타겟 연결
        tgt_ids = torch.cat([v['labels'] for v in targets])
        tgt_bbox = torch.cat([v['boxes'] for v in targets])

        if len(tgt_ids) == 0:
            return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in range(batch_size)]

        # 분류 비용
        cost_class = -out_prob[:, tgt_ids]

        # L1 박스 비용
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # 총 비용 매트릭스
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        C = C.view(batch_size, num_queries, -1).cpu()

        # 헝가리안 할당
        indices = []
        for i, c in enumerate(C.split(1, dim=0)):
            c = c.squeeze(0)
            if c.shape[1] == 0:
                indices.append((torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)))
                continue

            # 선형 할당 문제 해결
            row_ind, col_ind = linear_sum_assignment(c.numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), torch.as_tensor(col_ind, dtype=torch.int64)))

        return indices

# DETR 손실 함수
class DETRLoss(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        # 클래스 가중치 (no object 클래스 다운웨이트)
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """분류 손실"""
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                  dtype=torch.int64, device=src_logits.device)
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

def collate_fn(batch):
    """Custom collate function"""
    images, targets = tuple(zip(*batch))
    images = torch.stack(images, 0)
    return images, list(targets)

def train_medical_transformer_detection(dataset_type='chest_xray', num_epochs=50, batch_size=4, lr=1e-4):
    """
    Transformer 기반 의료 객체 검출 훈련

    Args:
        dataset_type: 의료 데이터셋 타입
        num_epochs: 훈련 에포크 수
        batch_size: 배치 크기 (Transformer는 메모리 많이 사용)
        lr: 학습률
    """

    # 로거 설정
    logger = create_logger_for_medical_detection('transformer_detection', dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # 데이터셋 준비
    train_dataset = MedicalTransformerDetectionDataset(data_type=dataset_type, split='train')
    val_dataset = MedicalTransformerDetectionDataset(data_type=dataset_type, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, drop_last=True)

    # 모델 설정
    model = MedicalDETR(num_classes=train_dataset.num_classes).to(device)

    # 손실 함수 설정
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5}
    losses = ['labels', 'boxes']
    criterion = DETRLoss(train_dataset.num_classes, matcher, weight_dict, eos_coef=0.1, losses=losses)

    # 옵티마이저 설정 (Transformer는 다른 학습률 적용)
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr * 0.1,  # 백본은 더 작은 학습률
        },
    ]
    optimizer = optim.AdamW(param_dicts, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # 훈련 메트릭 저장
    train_losses = []
    val_losses = []

    logger.log("Starting Medical Transformer Detection training...")
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
            loss_dict = criterion(outputs, targets)
            losses_reduced = sum(loss for loss in loss_dict.values())

            # 역전파
            losses_reduced.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += losses_reduced.item()

            if batch_idx % 10 == 0:
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

        # 중간 결과 저장 (매 10 에포크)
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                sample_images, sample_targets = next(iter(val_loader))
                sample_images = sample_images[:2].to(device)  # 메모리 고려

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

                    # no object 클래스 제외하고 최대 확률
                    pred_scores = pred_scores[:, :-1].max(dim=-1)[0]

                    # 상위 예측 선택
                    top_indices = pred_scores.topk(min(3, len(pred_scores)))[1]

                    img_with_boxes = img.copy()
                    for idx in top_indices:
                        if pred_scores[idx] > 0.5:
                            box = pred_boxes[idx] * 384  # 이미지 크기로 스케일링
                            x1, y1, x2, y2 = box.cpu().numpy()

                            # 박스 그리기
                            thickness = 2
                            img_with_boxes[int(y1):int(y1)+thickness, int(x1):int(x2)] = [1, 0, 0]
                            img_with_boxes[int(y2):int(y2)+thickness, int(x1):int(x2)] = [1, 0, 0]
                            img_with_boxes[int(y1):int(y2), int(x1):int(x1)+thickness] = [1, 0, 0]
                            img_with_boxes[int(y1):int(y2), int(x2):int(x2)+thickness] = [1, 0, 0]

                    vis_images.extend([img, img_with_boxes])

                logger.save_image_grid(vis_images,
                                     f'transformer_detection_epoch_{epoch+1}.png',
                                     titles=['Original', 'Transformer Detection'] * 2,
                                     nrow=2)

    # 최종 모델 저장
    logger.save_model(model, "medical_transformer_detection_final",
                     optimizer=optimizer, epoch=num_epochs,
                     config={'dataset_type': dataset_type, 'num_epochs': num_epochs})

    # 훈련 곡선 저장
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Medical Transformer Detection Training Loss')
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
    # 어텐션 맵 분석 (마지막 배치)
    model.eval()
    with torch.no_grad():
        sample_images, _ = next(iter(val_loader))
        sample_images = sample_images[:1].to(device)

        outputs = model(sample_images)

        # 백본 어텐션 가중치
        if outputs['backbone_attentions']:
            # 마지막 레이어의 어텐션 사용
            last_attn = outputs['backbone_attentions'][-1][0, 0]  # [num_patches, num_patches]

            # 클래스 토큰과 패치들 간의 어텐션 (첫 번째 행)
            cls_attn = last_attn[0, 1:].cpu().numpy()  # 클래스 토큰 제외

            # 패치 격자로 재구성
            patch_size = int(np.sqrt(len(cls_attn)))
            attn_map = cls_attn.reshape(patch_size, patch_size)

            plt.imshow(attn_map, cmap='hot', interpolation='nearest')
            plt.title('Vision Transformer Attention Map')
            plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'transformer_detection_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logger.log("Medical Transformer Detection training completed successfully!")
    logger.log(f"Final validation loss: {val_losses[-1]:.4f}")
    logger.log(f"Results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

if __name__ == "__main__":
    print("🤖 Transformer 기반 의료 객체 검출 (Medical Object Detection with Transformers)")
    print("=" * 80)

    # 하이퍼파라미터 설정
    config = {
        'dataset_type': 'chest_xray',
        'num_epochs': 5,  # 빠른 테스트를 위해 5로 설정
        'batch_size': 2,  # Transformer는 메모리 사용량이 높음
        'lr': 1e-4
    }

    print(f"Dataset: {config['dataset_type']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print()

    try:
        model, results_dir = train_medical_transformer_detection(
            dataset_type=config['dataset_type'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )

        print("\n✅ Medical Transformer Detection training completed successfully!")
        print(f"📁 Results saved to: {results_dir}")

        print("\n📊 Generated files include:")
        print("- images/: Transformer detection result visualizations")
        print("- models/: Trained Medical DETR model")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves and attention analysis")
        print("- metrics/: Training metrics in JSON format")

        print("\n🎯 Transformer Detection Features:")
        print("- Vision Transformer backbone")
        print("- DETR-style set-based detection")
        print("- Hungarian matching algorithm")
        print("- Medical-specific initialization")
        print("- Multi-head attention visualization")

    except Exception as e:
        print(f"\n❌ Error during Transformer detection training: {str(e)}")
        import traceback
        traceback.print_exc()
