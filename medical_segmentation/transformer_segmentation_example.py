#!/usr/bin/env python3
"""
Transformer 기반 의료 영상 분할 (Medical Image Segmentation with Transformers)

Vision Transformer를 활용한 의료 영상 분할 시스템입니다.
SETR (SEgmentation TRansformer)와 Segmenter 아키텍처를 의료 영상에 특화하여 구현하였습니다.

주요 기능:
- Vision Transformer 인코더
- Multi-scale Feature Processing
- Patch-based Segmentation
- Medical-specific Loss Functions
- Self-attention Visualization
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
from sklearn.metrics import jaccard_score, f1_score
sys.path.append('/workspace/Vision-101')
from medical.result_logger import create_logger_for_medical_segmentation

class MedicalTransformerSegmentationDataset(Dataset):
    def __init__(self, data_type='brain_mri', split='train', transform=None):
        """
        Transformer 분할용 의료 데이터셋

        Args:
            data_type: 'brain_mri', 'cardiac_mri', 'liver_ct', 'lung_ct', 'retinal_oct'
            split: 'train', 'val', 'test'
            transform: 이미지 전처리
        """
        self.data_type = data_type
        self.split = split
        self.transform = transform or transforms.Compose([
            transforms.Resize((384, 384)),  # Transformer는 큰 해상도 선호
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 세그멘테이션 클래스 정의
        self.class_names, self.num_classes = self._get_segmentation_classes()

        # 합성 데이터 생성
        self.images, self.masks = self._generate_segmentation_data()

    def _get_segmentation_classes(self):
        """의료 영상별 분할 클래스 정의"""
        class_maps = {
            'brain_mri': (['background', 'gray_matter', 'white_matter', 'csf', 'tumor'], 5),
            'cardiac_mri': (['background', 'lv_cavity', 'lv_myocardium', 'rv_cavity'], 4),
            'liver_ct': (['background', 'liver', 'tumor', 'vessel'], 4),
            'lung_ct': (['background', 'lung', 'vessel', 'airway', 'lesion'], 5),
            'retinal_oct': (['background', 'retina', 'choroid', 'drusen', 'fluid'], 5)
        }
        names, num = class_maps.get(self.data_type, (['background', 'organ', 'lesion'], 3))
        return names, num

    def _generate_segmentation_data(self):
        """분할 데이터 생성"""
        images = []
        masks = []

        num_samples = 400 if self.split == 'train' else 100

        for i in range(num_samples):
            # 의료 영상과 마스크 생성
            image, mask = self._create_medical_segmentation_pair(i)

            images.append(image)
            masks.append(mask)

        return images, masks

    def _create_medical_segmentation_pair(self, seed):
        """의료 영상-마스크 쌍 생성"""
        np.random.seed(seed)

        if self.data_type == 'brain_mri':
            return self._create_brain_mri_segmentation(seed)
        elif self.data_type == 'cardiac_mri':
            return self._create_cardiac_mri_segmentation(seed)
        elif self.data_type == 'liver_ct':
            return self._create_liver_ct_segmentation(seed)
        elif self.data_type == 'lung_ct':
            return self._create_lung_ct_segmentation(seed)
        else:  # retinal_oct
            return self._create_retinal_oct_segmentation(seed)

    def _create_brain_mri_segmentation(self, seed):
        """뇌 MRI 분할 데이터 생성"""
        np.random.seed(seed)
        image = np.zeros((384, 384, 3))
        mask = np.zeros((384, 384), dtype=np.uint8)

        # 뇌 윤곽
        center_x, center_y = 192, 192
        brain_radius = 140

        for y in range(384):
            for x in range(384):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= brain_radius:
                    # 회백질 영역
                    if dist <= 90:
                        image[y, x] = 0.6 + np.random.normal(0, 0.05)
                        mask[y, x] = 1  # gray_matter
                    # 백질 영역
                    else:
                        image[y, x] = 0.4 + np.random.normal(0, 0.05)
                        mask[y, x] = 2  # white_matter

        # 뇌척수액 (CSF) 영역
        ventricle_centers = [(160, 160), (224, 160), (160, 224), (224, 224)]
        for vx, vy in ventricle_centers:
            for y in range(max(0, vy-20), min(384, vy+20)):
                for x in range(max(0, vx-20), min(384, vx+20)):
                    dist = np.sqrt((x - vx)**2 + (y - vy)**2)
                    if dist < 15:
                        image[y, x] = 0.1 + np.random.normal(0, 0.02)
                        mask[y, x] = 3  # csf

        # 종양 영역 (일부 케이스에서만)
        if np.random.random() > 0.6:
            tumor_x = np.random.randint(120, 264)
            tumor_y = np.random.randint(120, 264)
            tumor_size = np.random.randint(15, 35)

            for y in range(max(0, tumor_y-tumor_size), min(384, tumor_y+tumor_size)):
                for x in range(max(0, tumor_x-tumor_size), min(384, tumor_x+tumor_size)):
                    dist = np.sqrt((x - tumor_x)**2 + (y - tumor_y)**2)
                    if dist < tumor_size:
                        image[y, x] = 0.8 + np.random.normal(0, 0.1)
                        mask[y, x] = 4  # tumor

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8), mask

    def _create_cardiac_mri_segmentation(self, seed):
        """심장 MRI 분할 데이터 생성"""
        np.random.seed(seed)
        image = np.zeros((384, 384, 3))
        mask = np.zeros((384, 384), dtype=np.uint8)

        # 좌심실
        lv_center_x, lv_center_y = 200, 200
        lv_outer_radius = 45
        lv_inner_radius = 25

        for y in range(384):
            for x in range(384):
                dist = np.sqrt((x - lv_center_x)**2 + (y - lv_center_y)**2)

                if dist <= lv_inner_radius:
                    # 좌심실 내강
                    image[y, x] = 0.2 + np.random.normal(0, 0.03)
                    mask[y, x] = 1  # lv_cavity
                elif dist <= lv_outer_radius:
                    # 좌심실 심근
                    image[y, x] = 0.7 + np.random.normal(0, 0.05)
                    mask[y, x] = 2  # lv_myocardium

        # 우심실
        rv_center_x, rv_center_y = 160, 180
        rv_radius = 30

        for y in range(384):
            for x in range(384):
                dist = np.sqrt((x - rv_center_x)**2 + (y - rv_center_y)**2)
                if dist <= rv_radius:
                    image[y, x] = 0.25 + np.random.normal(0, 0.03)
                    mask[y, x] = 3  # rv_cavity

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8), mask

    def _create_liver_ct_segmentation(self, seed):
        """간 CT 분할 데이터 생성"""
        np.random.seed(seed)
        image = np.random.normal(0.35, 0.08, (384, 384, 3))
        mask = np.zeros((384, 384), dtype=np.uint8)

        # 간 영역 (불규칙한 모양)
        liver_center_x, liver_center_y = 200, 220

        for y in range(150, 320):
            for x in range(120, 300):
                # 불규칙한 간 경계
                boundary_noise = 20 * np.sin((x + y) * 0.02) * np.cos(x * 0.03)
                dist = np.sqrt((x - liver_center_x)**2 + (y - liver_center_y)**2)

                if dist < 80 + boundary_noise:
                    image[y, x] = 0.5 + np.random.normal(0, 0.04)
                    mask[y, x] = 1  # liver

        # 간 종양 (일부 케이스)
        if np.random.random() > 0.5:
            tumor_x = np.random.randint(150, 250)
            tumor_y = np.random.randint(180, 280)
            tumor_size = np.random.randint(15, 30)

            for y in range(max(0, tumor_y-tumor_size), min(384, tumor_y+tumor_size)):
                for x in range(max(0, tumor_x-tumor_size), min(384, tumor_x+tumor_size)):
                    dist = np.sqrt((x - tumor_x)**2 + (y - tumor_y)**2)
                    if dist < tumor_size:
                        image[y, x] = 0.3 + np.random.normal(0, 0.05)
                        mask[y, x] = 2  # tumor

        # 혈관 구조
        for vessel in range(5):
            vx = np.random.randint(140, 260)
            vy_start = np.random.randint(160, 200)

            for y in range(vy_start, min(384, vy_start + 80)):
                vessel_width = 3 + np.sin(y * 0.1)
                for dx in range(-int(vessel_width), int(vessel_width)):
                    if 0 <= vx + dx < 384:
                        image[y, vx + dx] = 0.8 + np.random.normal(0, 0.03)
                        mask[y, vx + dx] = 3  # vessel

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8), mask

    def _create_lung_ct_segmentation(self, seed):
        """폐 CT 분할 데이터 생성"""
        np.random.seed(seed)
        image = np.random.normal(0.2, 0.05, (384, 384, 3))
        mask = np.zeros((384, 384), dtype=np.uint8)

        # 양측 폐야
        for lung_side in [(-80, 0), (80, 0)]:
            lung_x = 192 + lung_side[0]
            lung_y = 180

            for y in range(80, 280):
                for x in range(max(0, lung_x-60), min(384, lung_x+60)):
                    dist = np.sqrt((x - lung_x)**2 + (y - lung_y)**2)
                    if dist < 55:
                        image[y, x] = 0.1 + np.random.normal(0, 0.02)
                        mask[y, x] = 1  # lung

        # 혈관 구조
        for vessel in range(15):
            vx = np.random.randint(80, 304)
            vy = np.random.randint(100, 260)

            # 혈관이 폐야 내부에 있는지 확인
            if mask[vy, vx] == 1:
                vessel_length = np.random.randint(20, 50)
                direction = np.random.uniform(0, 2 * np.pi)

                for t in range(vessel_length):
                    x = int(vx + t * np.cos(direction))
                    y = int(vy + t * np.sin(direction))

                    if 0 <= x < 384 and 0 <= y < 384 and mask[y, x] == 1:
                        image[y, x] = 0.6 + np.random.normal(0, 0.03)
                        mask[y, x] = 2  # vessel

        # 기도
        for airway in range(8):
            ax = np.random.randint(120, 264)
            ay = np.random.randint(120, 240)

            if mask[ay, ax] == 1:
                airway_size = np.random.randint(3, 8)
                for y in range(max(0, ay-airway_size), min(384, ay+airway_size)):
                    for x in range(max(0, ax-airway_size), min(384, ax+airway_size)):
                        if mask[y, x] == 1:
                            dist = np.sqrt((x - ax)**2 + (y - ay)**2)
                            if dist < airway_size:
                                image[y, x] = 0.05
                                mask[y, x] = 3  # airway

        # 병변 (일부 케이스)
        if np.random.random() > 0.6:
            lesion_count = np.random.randint(1, 4)
            for _ in range(lesion_count):
                lx = np.random.randint(100, 284)
                ly = np.random.randint(120, 240)

                if mask[ly, lx] == 1:
                    lesion_size = np.random.randint(8, 20)
                    for y in range(max(0, ly-lesion_size), min(384, ly+lesion_size)):
                        for x in range(max(0, lx-lesion_size), min(384, lx+lesion_size)):
                            dist = np.sqrt((x - lx)**2 + (y - ly)**2)
                            if dist < lesion_size and mask[y, x] == 1:
                                image[y, x] = 0.5 + np.random.normal(0, 0.05)
                                mask[y, x] = 4  # lesion

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8), mask

    def _create_retinal_oct_segmentation(self, seed):
        """망막 OCT 분할 데이터 생성"""
        np.random.seed(seed)
        image = np.random.exponential(0.2, (384, 384, 3))
        mask = np.zeros((384, 384), dtype=np.uint8)

        # 망막 층 구조
        for layer in range(8):  # 8개 망막층
            y_start = 100 + layer * 20
            layer_thickness = np.random.randint(8, 15)

            for y in range(y_start, min(384, y_start + layer_thickness)):
                for x in range(384):
                    # 층별 강도 변화
                    intensity = 0.3 + layer * 0.05 + np.random.normal(0, 0.02)
                    image[y, x] = intensity
                    mask[y, x] = 1  # retina

        # 맥락막
        choroid_start = 280
        for y in range(choroid_start, 350):
            for x in range(384):
                image[y, x] = 0.4 + np.random.normal(0, 0.08)
                mask[y, x] = 2  # choroid

        # 드루젠 (일부 케이스)
        if np.random.random() > 0.7:
            drusen_count = np.random.randint(3, 8)
            for _ in range(drusen_count):
                dx = np.random.randint(50, 334)
                dy = np.random.randint(250, 280)
                drusen_size = np.random.randint(5, 15)

                for y in range(max(0, dy-drusen_size), min(384, dy+drusen_size)):
                    for x in range(max(0, dx-drusen_size), min(384, dx+drusen_size)):
                        dist = np.sqrt((x - dx)**2 + (y - dy)**2)
                        if dist < drusen_size:
                            image[y, x] = 0.8 + np.random.normal(0, 0.05)
                            mask[y, x] = 3  # drusen

        # 액체 축적 (일부 케이스)
        if np.random.random() > 0.8:
            fluid_x = np.random.randint(100, 284)
            fluid_y = np.random.randint(150, 220)
            fluid_size = np.random.randint(15, 30)

            for y in range(max(0, fluid_y-fluid_size), min(384, fluid_y+fluid_size)):
                for x in range(max(0, fluid_x-fluid_size), min(384, fluid_x+fluid_size)):
                    dist = np.sqrt((x - fluid_x)**2 + (y - fluid_y)**2)
                    if dist < fluid_size:
                        image[y, x] = 0.1
                        mask[y, x] = 4  # fluid

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8), mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)

        # 마스크를 텐서로 변환 (클래스 인덱스)
        mask = torch.from_numpy(mask).long()

        return image, mask

# Vision Transformer 패치 임베딩 (Segmentation용)
class SegmentationPatchEmbedding(nn.Module):
    def __init__(self, img_size=384, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H', W')

        # 공간 차원 보존을 위해 reshape
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

# 위치 인코딩 (2D용)
class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim, grid_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # 2D 위치 인코딩 생성
        pos_embed = self._get_2d_sincos_pos_embed(embed_dim, grid_size)
        self.register_buffer('pos_embed', pos_embed)

    def _get_2d_sincos_pos_embed(self, embed_dim, grid_size):
        """2D sin-cos 위치 인코딩"""
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # 여기서 w, h 순서로
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return torch.from_numpy(pos_embed).float()

    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim, grid):
        assert embed_dim % 2 == 0

        # x, y 좌표 분리
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim, pos):
        """1D sin-cos 위치 인코딩"""
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def forward(self, x):
        return x + self.pos_embed

# Transformer 인코더 블록 (Segmentation용)
class SegmentationTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
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
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x, attn_weights

# Medical Segmentation Transformer
class MedicalSegmentationTransformer(nn.Module):
    def __init__(self, num_classes, img_size=384, patch_size=16, embed_dim=768,
                 depth=12, num_heads=12, decoder_depth=2):
        super().__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.n_patches = self.grid_size ** 2

        # 패치 임베딩
        self.patch_embed = SegmentationPatchEmbedding(img_size, patch_size, 3, embed_dim)

        # 위치 인코딩
        self.pos_embed = PositionalEncoding2D(embed_dim, self.grid_size)

        # Transformer 인코더
        self.encoder_blocks = nn.ModuleList([
            SegmentationTransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.encoder_norm = nn.LayerNorm(embed_dim)

        # 분할 디코더
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
                nn.BatchNorm2d(embed_dim // 2),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) for _ in range(decoder_depth)
        ])

        # 최종 분류 헤드
        final_dim = embed_dim // (2 ** decoder_depth)
        self.seg_head = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, 3, padding=1),
            nn.BatchNorm2d(final_dim),
            nn.ReLU(),
            nn.Conv2d(final_dim, num_classes, 1)
        )

        # 의료 영상 특화 초기화
        self._init_medical_weights()

    def _init_medical_weights(self):
        """의료 영상 특화 가중치 초기화"""
        # 패치 임베딩 초기화
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # 클래스 헤드 초기화 (배경 클래스 편향)
        nn.init.constant_(self.seg_head[-1].bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # 패치 임베딩
        x = self.patch_embed(x)  # [B, n_patches, embed_dim]

        # 위치 인코딩
        x = self.pos_embed(x)

        # Transformer 인코더
        attention_weights = []
        for block in self.encoder_blocks:
            x, attn = block(x)
            attention_weights.append(attn)

        x = self.encoder_norm(x)

        # 공간 차원 복원
        x = x.transpose(1, 2).reshape(B, -1, self.grid_size, self.grid_size)

        # 분할 디코더
        for decoder_layer in self.decoder:
            x = decoder_layer(x)

        # 최종 업샘플링으로 원본 크기 복원
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        # 분할 예측
        seg_logits = self.seg_head(x)

        return {
            'seg_logits': seg_logits,
            'attention_weights': attention_weights
        }

# 다중 스케일 Transformer (개선된 버전)
class MultiScaleSegmentationTransformer(nn.Module):
    def __init__(self, num_classes, img_size=384, embed_dim=768):
        super().__init__()

        self.num_classes = num_classes

        # 다중 스케일 패치 임베딩
        self.patch_embeds = nn.ModuleList([
            SegmentationPatchEmbedding(img_size, 16, 3, embed_dim),  # 16x16 패치
            SegmentationPatchEmbedding(img_size, 32, 3, embed_dim),  # 32x32 패치
        ])

        # 각 스케일별 위치 인코딩
        self.pos_embeds = nn.ModuleList([
            PositionalEncoding2D(embed_dim, img_size // 16),
            PositionalEncoding2D(embed_dim, img_size // 32),
        ])

        # 스케일 융합
        self.scale_fusion = nn.MultiheadAttention(embed_dim, 8, batch_first=True)

        # Transformer 인코더
        self.encoder_blocks = nn.ModuleList([
            SegmentationTransformerBlock(embed_dim, 12)
            for _ in range(8)
        ])

        # 점진적 업샘플링 디코더
        self.decoder = nn.Sequential(
            # 24x24 -> 48x48
            nn.ConvTranspose2d(embed_dim, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            # 48x48 -> 96x96
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # 96x96 -> 192x192
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 192x192 -> 384x384
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 최종 분류
            nn.Conv2d(64, num_classes, 3, padding=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 다중 스케일 특징 추출
        multi_scale_features = []
        for patch_embed, pos_embed in zip(self.patch_embeds, self.pos_embeds):
            features = patch_embed(x)
            features = pos_embed(features)
            multi_scale_features.append(features)

        # 스케일 간 어텐션으로 융합
        # 더 세밀한 스케일(16x16)을 쿼리로 사용
        fine_features = multi_scale_features[0]
        coarse_features = multi_scale_features[1]

        # 크기 맞추기 (coarse를 fine 크기로 interpolate)
        if coarse_features.shape[1] != fine_features.shape[1]:
            # coarse features를 fine features 크기로 확장
            coarse_upsampled = F.interpolate(
                coarse_features.transpose(1, 2).reshape(B, -1, 12, 12),
                size=(24, 24), mode='bilinear', align_corners=False
            ).flatten(2).transpose(1, 2)

            fused_features, _ = self.scale_fusion(fine_features, coarse_upsampled, coarse_upsampled)
        else:
            fused_features, _ = self.scale_fusion(fine_features, coarse_features, coarse_features)

        # Transformer 인코더
        x = fused_features
        attention_weights = []
        for block in self.encoder_blocks:
            x, attn = block(x)
            attention_weights.append(attn)

        # 공간 차원 복원 (24x24 그리드)
        x = x.transpose(1, 2).reshape(B, -1, 24, 24)

        # 디코더로 업샘플링
        seg_logits = self.decoder(x)

        return {
            'seg_logits': seg_logits,
            'attention_weights': attention_weights
        }

# Dice Loss + Cross Entropy
class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, dice_weight=0.5, ce_weight=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def dice_loss(self, pred, target):
        """Dice Loss 계산"""
        smooth = 1e-5

        # 소프트맥스 적용
        pred = F.softmax(pred, dim=1)

        dice_losses = []
        for c in range(self.num_classes):
            pred_c = pred[:, c]
            target_c = (target == c).float()

            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))

            dice = (2 * intersection + smooth) / (union + smooth)
            dice_losses.append(1 - dice.mean())

        return sum(dice_losses) / len(dice_losses)

    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)

        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss

        return total_loss, {'ce_loss': ce_loss.item(), 'dice_loss': dice_loss.item()}

def calculate_metrics(pred, target, num_classes):
    """분할 메트릭 계산"""
    pred = torch.argmax(pred, dim=1).cpu().numpy()
    target = target.cpu().numpy()

    # Flatten
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    # IoU 계산 (클래스별)
    ious = []
    for c in range(num_classes):
        pred_c = (pred_flat == c)
        target_c = (target_flat == c)

        intersection = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()

        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0 if intersection == 0 else 0.0

        ious.append(iou)

    # mIoU
    miou = np.mean(ious)

    # 전체 정확도
    accuracy = (pred_flat == target_flat).mean()

    return {
        'miou': miou,
        'accuracy': accuracy,
        'class_ious': ious
    }

def train_medical_transformer_segmentation(dataset_type='brain_mri', num_epochs=50, batch_size=4, lr=1e-4):
    """
    Transformer 기반 의료 영상 분할 훈련

    Args:
        dataset_type: 의료 데이터셋 타입
        num_epochs: 훈련 에포크 수
        batch_size: 배치 크기
        lr: 학습률
    """

    # 로거 설정
    logger = create_logger_for_medical_segmentation('transformer_segmentation', dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # 데이터셋 준비
    train_dataset = MedicalTransformerSegmentationDataset(data_type=dataset_type, split='train')
    val_dataset = MedicalTransformerSegmentationDataset(data_type=dataset_type, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 모델 설정 (두 가지 모델 비교)
    models = {
        'basic': MedicalSegmentationTransformer(num_classes=train_dataset.num_classes).to(device),
        'multiscale': MultiScaleSegmentationTransformer(num_classes=train_dataset.num_classes).to(device)
    }

    # 손실 함수
    criterion = DiceCrossEntropyLoss(num_classes=train_dataset.num_classes)

    # 옵티마이저 (각 모델별)
    optimizers = {
        name: optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        for name, model in models.items()
    }

    schedulers = {
        name: optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        for name, opt in optimizers.items()
    }

    # 훈련 메트릭 저장
    train_losses = {name: [] for name in models.keys()}
    val_losses = {name: [] for name in models.keys()}
    val_mious = {name: [] for name in models.keys()}

    logger.log("Starting Medical Transformer Segmentation training...")
    for name, model in models.items():
        logger.log(f"{name.upper()} model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(num_epochs):
        # 각 모델별로 훈련
        for model_name, model in models.items():
            model.train()
            running_loss = 0.0

            for batch_idx, (images, masks) in enumerate(train_loader):
                images, masks = images.to(device), masks.to(device)

                optimizers[model_name].zero_grad()

                # 순전파
                outputs = model(images)

                # 손실 계산
                loss, loss_dict = criterion(outputs['seg_logits'], masks)

                # 역전파
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizers[model_name].step()

                running_loss += loss.item()

                if batch_idx % 10 == 0 and model_name == 'basic':  # 첫 번째 모델만 로그
                    logger.log(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                              f'Loss: {loss.item():.4f} (CE: {loss_dict["ce_loss"]:.4f}, '
                              f'Dice: {loss_dict["dice_loss"]:.4f})')

            train_losses[model_name].append(running_loss / len(train_loader))

            # 검증 단계
            model.eval()
            val_loss = 0.0
            val_metrics = []

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)

                    outputs = model(images)
                    loss, _ = criterion(outputs['seg_logits'], masks)

                    val_loss += loss.item()

                    # 메트릭 계산
                    metrics = calculate_metrics(outputs['seg_logits'], masks, train_dataset.num_classes)
                    val_metrics.append(metrics)

            avg_val_loss = val_loss / len(val_loader)
            avg_val_miou = np.mean([m['miou'] for m in val_metrics])

            val_losses[model_name].append(avg_val_loss)
            val_mious[model_name].append(avg_val_miou)

            schedulers[model_name].step()

        # 에포크별 로깅
        logger.log(f'Epoch [{epoch+1}/{num_epochs}]')
        for model_name in models.keys():
            logger.log(f'{model_name.upper()} - Train Loss: {train_losses[model_name][-1]:.4f}, '
                      f'Val Loss: {val_losses[model_name][-1]:.4f}, '
                      f'Val mIoU: {val_mious[model_name][-1]:.4f}')

        # 메트릭 저장
        logger.save_metrics({
            'epoch': epoch + 1,
            **{f'{name}_train_loss': train_losses[name][-1] for name in models.keys()},
            **{f'{name}_val_loss': val_losses[name][-1] for name in models.keys()},
            **{f'{name}_val_miou': val_mious[name][-1] for name in models.keys()},
        })

        # 중간 결과 저장 (매 10 에포크)
        if (epoch + 1) % 10 == 0:
            for model_name, model in models.items():
                model.eval()
                with torch.no_grad():
                    sample_images, sample_masks = next(iter(val_loader))
                    sample_images = sample_images[:2].to(device)
                    sample_masks = sample_masks[:2]

                    outputs = model(sample_images)
                    pred_masks = torch.argmax(outputs['seg_logits'], dim=1)

                    # 시각화를 위한 이미지 준비
                    vis_images = []
                    for i in range(len(sample_images)):
                        # 원본 이미지
                        img = sample_images[i].cpu().numpy().transpose(1, 2, 0)
                        img = (img - img.min()) / (img.max() - img.min())

                        # 실제 마스크
                        true_mask = sample_masks[i].numpy() / train_dataset.num_classes

                        # 예측 마스크
                        pred_mask = pred_masks[i].cpu().numpy() / train_dataset.num_classes

                        vis_images.extend([img, true_mask, pred_mask])

                    titles = ['Original', 'True Mask', f'{model_name.upper()} Pred'] * 2

                    logger.save_image_grid(vis_images,
                                         f'transformer_seg_{model_name}_epoch_{epoch+1}.png',
                                         titles=titles,
                                         nrow=3)

    # 최종 모델들 저장
    for model_name, model in models.items():
        logger.save_model(model, f"medical_transformer_seg_{model_name}_final",
                         optimizer=optimizers[model_name], epoch=num_epochs,
                         config={'dataset_type': dataset_type, 'model_type': model_name})

    # 훈련 곡선 저장
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    for model_name in models.keys():
        plt.plot(train_losses[model_name], label=f'{model_name.upper()} Train')
        plt.plot(val_losses[model_name], label=f'{model_name.upper()} Val')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 2)
    for model_name in models.keys():
        plt.plot(val_mious[model_name], label=f'{model_name.upper()}')
    plt.title('Validation mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 3)
    # 최종 성능 비교
    final_mious = [val_mious[name][-1] for name in models.keys()]
    model_names = [name.upper() for name in models.keys()]

    plt.bar(model_names, final_mious)
    plt.title('Final Model Comparison (mIoU)')
    plt.ylabel('mIoU')
    plt.ylim(0, 1)
    for i, v in enumerate(final_mious):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

    plt.subplot(1, 4, 4)
    # 어텐션 맵 시각화 (Basic 모델)
    models['basic'].eval()
    with torch.no_grad():
        sample_images, _ = next(iter(val_loader))
        sample_images = sample_images[:1].to(device)

        outputs = models['basic'](sample_images)

        if outputs['attention_weights']:
            # 마지막 레이어의 어텐션 사용
            last_attn = outputs['attention_weights'][-1][0, 0]  # [seq_len, seq_len]

            # 평균 어텐션 (첫 번째 토큰 제외)
            attn_map = last_attn.mean(dim=0).cpu().numpy()

            # 패치 격자로 재구성
            grid_size = int(np.sqrt(len(attn_map)))
            attn_map = attn_map.reshape(grid_size, grid_size)

            plt.imshow(attn_map, cmap='hot', interpolation='nearest')
            plt.title('Transformer Attention Map')
            plt.colorbar()

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'transformer_segmentation_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logger.log("Medical Transformer Segmentation training completed successfully!")
    for model_name in models.keys():
        logger.log(f"Final {model_name.upper()} mIoU: {val_mious[model_name][-1]:.4f}")
    logger.log(f"Results saved in: {logger.dirs['base']}")

    return models, logger.dirs['base']

if __name__ == "__main__":
    print("🧠 Transformer 기반 의료 영상 분할 (Medical Image Segmentation with Transformers)")
    print("=" * 80)

    # 하이퍼파라미터 설정
    config = {
        'dataset_type': 'brain_mri',
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
        models, results_dir = train_medical_transformer_segmentation(
            dataset_type=config['dataset_type'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )

        print("\n✅ Medical Transformer Segmentation training completed successfully!")
        print(f"📁 Results saved to: {results_dir}")

        print("\n📊 Generated files include:")
        print("- images/: Transformer segmentation visualizations")
        print("- models/: Trained Basic and Multi-scale Transformer models")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves and attention analysis")
        print("- metrics/: Training metrics in JSON format")

        print("\n🎯 Transformer Segmentation Features:")
        print("- Vision Transformer encoder for medical images")
        print("- Multi-scale patch processing")
        print("- 2D positional encoding")
        print("- Dice + Cross-Entropy loss")
        print("- Self-attention visualization")
        print("- Basic vs Multi-scale model comparison")

    except Exception as e:
        print(f"\n❌ Error during Transformer segmentation training: {str(e)}")
        import traceback
        traceback.print_exc()