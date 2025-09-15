#!/usr/bin/env python3
"""
다중 모달 의료 AI 융합 시스템 (Multi-modal Medical AI Fusion)

다중 모달 의료 AI는 영상, 텍스트, 수치 데이터, 유전자 정보 등 다양한 형태의
의료 데이터를 통합하여 보다 정확하고 포괄적인 진단 및 예후 예측을 수행합니다.

주요 기능:
- 영상-텍스트 융합 (Vision-Language Fusion)
- 다중 영상 모달리티 융합 (Multi-imaging Fusion)
- 임상 데이터 통합 분석
- 크로스 모달리티 어텐션 메커니즘
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
import re
import json
from collections import Counter
sys.path.append('/workspace/Vision-101')
from result_logger import create_logger_for_multimodal_medical

class MultimodalMedicalDataset(Dataset):
    def __init__(self, data_type='chest_radiology', transform=None):
        """
        다중 모달 의료 데이터셋

        Args:
            data_type: 'chest_radiology', 'pathology', 'cardiology', 'oncology'
            transform: 이미지 전처리
        """
        self.data_type = data_type
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 어휘집 구축
        self.vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)

        # 다중 모달 사례 생성
        self.cases = self._generate_multimodal_cases()

    def _build_vocabulary(self):
        """의료 용어 어휘집 구축"""
        medical_terms = [
            '<PAD>', '<UNK>', '<START>', '<END>',
            # 해부학적 구조
            'lung', 'heart', 'liver', 'kidney', 'brain', 'chest', 'abdomen', 'pelvis',
            'left', 'right', 'upper', 'lower', 'anterior', 'posterior', 'medial', 'lateral',
            # 병리학적 소견
            'normal', 'abnormal', 'mass', 'lesion', 'nodule', 'opacity', 'consolidation',
            'infiltrate', 'effusion', 'pneumothorax', 'cardiomegaly', 'atelectasis',
            'tumor', 'cancer', 'benign', 'malignant', 'metastasis', 'inflammation',
            # 영상 소견
            'hyperdense', 'hypodense', 'isodense', 'hyperintense', 'hypointense',
            'enhancement', 'contrast', 'calcification', 'necrosis', 'hemorrhage',
            # 크기/정도
            'small', 'large', 'mild', 'moderate', 'severe', 'extensive', 'focal', 'diffuse',
            # 임상 용어
            'patient', 'findings', 'impression', 'diagnosis', 'recommendation',
            'follow', 'up', 'compare', 'previous', 'stable', 'improved', 'worsened',
            # 수치
            'mm', 'cm', 'size', 'diameter', 'length', 'width', 'height', 'volume'
        ]

        vocab = {term: idx for idx, term in enumerate(medical_terms)}
        return vocab

    def _generate_multimodal_cases(self):
        """다중 모달 의료 사례 생성"""
        cases = []

        for i in range(1000):
            case = self._create_multimodal_case(i)
            cases.append(case)

        return cases

    def _create_multimodal_case(self, seed):
        """다중 모달 사례 생성"""
        np.random.seed(seed)

        if self.data_type == 'chest_radiology':
            return self._create_chest_radiology_case(seed)
        elif self.data_type == 'pathology':
            return self._create_pathology_case(seed)
        elif self.data_type == 'cardiology':
            return self._create_cardiology_case(seed)
        else:
            return self._create_oncology_case(seed)

    def _create_chest_radiology_case(self, seed):
        """흉부 영상의학 다중 모달 사례"""
        np.random.seed(seed)

        # 영상 생성
        chest_xray = self._generate_chest_xray(seed)
        chest_ct = self._generate_chest_ct(seed)

        # 임상 정보
        age = np.random.randint(20, 90)
        gender = np.random.choice(['M', 'F'])
        symptom_list = [
            ['cough', 'fever'],
            ['shortness of breath'],
            ['chest pain'],
            ['fatigue'],
            ['cough', 'sputum'],
            ['no symptoms']
        ]
        symptoms = symptom_list[np.random.randint(len(symptom_list))]

        # 영상 판독 보고서 생성
        findings = self._generate_radiology_report(seed)
        report_text = ' '.join(findings)

        # 진단 레이블
        diagnosis_labels = ['normal', 'pneumonia', 'mass', 'nodule', 'effusion']
        diagnosis = np.random.choice(range(len(diagnosis_labels)))

        # 중증도 평가
        severity = np.random.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])

        return {
            'images': {
                'chest_xray': chest_xray,
                'chest_ct': chest_ct
            },
            'text': {
                'report': report_text,
                'symptoms': ' '.join(symptoms),
                'clinical_history': f"{age}yo {gender} with {symptoms[0]}"
            },
            'clinical_data': {
                'age': age,
                'gender': 1 if gender == 'M' else 0,
                'temperature': np.random.normal(37.0, 1.0),
                'wbc': np.random.lognormal(2.0, 0.3),
                'crp': np.random.exponential(10)
            },
            'diagnosis': diagnosis,
            'severity': severity
        }

    def _create_pathology_case(self, seed):
        """병리학 다중 모달 사례"""
        np.random.seed(seed)

        # 병리 이미지들
        h_e_stain = self._generate_pathology_image(seed, 'H&E')
        ihc_stain = self._generate_pathology_image(seed, 'IHC')

        # 병리 보고서
        pathology_report = self._generate_pathology_report(seed)

        # 분자 마커
        molecular_markers = {
            'ki67': np.random.uniform(0, 100),
            'p53': np.random.choice([0, 1]),
            'her2': np.random.choice([0, 1, 2, 3]),
            'er': np.random.uniform(0, 100),
            'pr': np.random.uniform(0, 100)
        }

        # 진단
        pathology_diagnoses = ['benign', 'dysplasia', 'carcinoma_in_situ', 'invasive_carcinoma']
        diagnosis = np.random.choice(range(len(pathology_diagnoses)))

        return {
            'images': {
                'h_e': h_e_stain,
                'ihc': ihc_stain
            },
            'text': {
                'report': pathology_report,
                'diagnosis_text': pathology_diagnoses[diagnosis]
            },
            'molecular_data': molecular_markers,
            'diagnosis': diagnosis,
            'severity': min(3, diagnosis)
        }

    def _generate_chest_xray(self, seed):
        """흉부 X-ray 이미지 생성"""
        np.random.seed(seed)
        image = np.zeros((224, 224, 3))

        # 기본 흉부 구조
        # 폐야
        for y in range(50, 180):
            for x in range(30, 194):
                image[y, x] = 0.3 + np.random.normal(0, 0.05)

        # 심장 그림자
        heart_center = (110, 140)
        for y in range(120, 180):
            for x in range(90, 130):
                dist = np.sqrt((x - heart_center[0])**2 + (y - heart_center[1])**2)
                if dist < 25:
                    image[y, x] = 0.7

        # 랜덤 병변 추가
        if np.random.random() > 0.6:
            lesion_x, lesion_y = np.random.randint(60, 164, 2)
            lesion_size = np.random.randint(8, 25)
            for y in range(max(0, lesion_y - lesion_size), min(224, lesion_y + lesion_size)):
                for x in range(max(0, lesion_x - lesion_size), min(224, lesion_x + lesion_size)):
                    image[y, x] = 0.8

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _generate_chest_ct(self, seed):
        """흉부 CT 이미지 생성"""
        np.random.seed(seed + 1000)  # 다른 시드 사용
        image = np.random.normal(0.4, 0.1, (224, 224, 3))

        # CT 특성 (더 세밀한 구조)
        # 폐 조직
        for y in range(40, 184):
            for x in range(20, 204):
                if np.random.random() > 0.7:
                    image[y, x] = 0.2  # 공기 포함 조직

        # 혈관 구조
        for i in range(5):
            vessel_x = 50 + i * 30
            for y in range(60, 160):
                thickness = 2 + np.sin(y * 0.1)
                image[y:y+int(thickness), vessel_x] = 0.8

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _generate_pathology_image(self, seed, stain_type):
        """병리 이미지 생성"""
        np.random.seed(seed + (100 if stain_type == 'IHC' else 0))

        if stain_type == 'H&E':
            # H&E 염색 (핑크-보라색 계열)
            image = np.random.uniform(0.6, 0.9, (224, 224, 3))
            # 세포핵 (보라색)
            for _ in range(300):
                nucleus_x, nucleus_y = np.random.randint(5, 219, 2)
                image[nucleus_y:nucleus_y+3, nucleus_x:nucleus_x+3, 0] *= 0.5
                image[nucleus_y:nucleus_y+3, nucleus_x:nucleus_x+3, 2] *= 1.2

        else:  # IHC
            # IHC 염색 (갈색 계열)
            image = np.random.uniform(0.7, 1.0, (224, 224, 3))
            # 양성 염색 영역 (갈색)
            for _ in range(150):
                pos_x, pos_y = np.random.randint(10, 214, 2)
                size = np.random.randint(5, 15)
                image[pos_y:pos_y+size, pos_x:pos_x+size, :2] *= 0.4

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _generate_radiology_report(self, seed):
        """영상의학 판독 보고서 생성"""
        np.random.seed(seed)

        report_templates = [
            "The lungs are clear without focal consolidation or effusion",
            "There is a small nodule in the right upper lobe measuring 8 mm",
            "Cardiomegaly is present with normal lung fields",
            "Bilateral lower lobe opacities consistent with pneumonia",
            "No acute cardiopulmonary abnormality",
            "Left pleural effusion with associated atelectasis",
            "Multiple pulmonary nodules concerning for metastases"
        ]

        findings = np.random.choice(report_templates, size=np.random.randint(1, 4), replace=False)
        return findings.tolist()

    def _generate_pathology_report(self, seed):
        """병리 보고서 생성"""
        np.random.seed(seed)

        reports = [
            "Sections show normal tissue architecture with no malignancy",
            "Moderate dysplasia with increased mitotic activity",
            "Invasive ductal carcinoma with lymphovascular invasion",
            "High grade sarcoma with extensive necrosis"
        ]

        return np.random.choice(reports)

    def _text_to_indices(self, text, max_length=50):
        """텍스트를 인덱스 시퀀스로 변환"""
        # 간단한 토큰화
        tokens = re.findall(r'\w+', text.lower())
        indices = [self.vocab.get('<START>')]

        for token in tokens[:max_length-2]:
            indices.append(self.vocab.get(token, self.vocab.get('<UNK>')))

        indices.append(self.vocab.get('<END>'))

        # 패딩
        while len(indices) < max_length:
            indices.append(self.vocab.get('<PAD>'))

        return indices[:max_length]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]

        # 이미지 처리
        processed_images = {}
        for img_type, img_data in case['images'].items():
            img = Image.fromarray(img_data)
            if self.transform:
                img = self.transform(img)
            processed_images[img_type] = img

        # 텍스트 처리
        processed_texts = {}
        for text_type, text_data in case['text'].items():
            text_indices = self._text_to_indices(text_data)
            processed_texts[text_type] = torch.tensor(text_indices, dtype=torch.long)

        # 임상 데이터 처리
        clinical_features = []
        if 'clinical_data' in case:
            for key, value in case['clinical_data'].items():
                clinical_features.append(float(value))
        if 'molecular_data' in case:
            for key, value in case['molecular_data'].items():
                clinical_features.append(float(value))

        clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32) if clinical_features else torch.tensor([0.0])

        return {
            'images': processed_images,
            'texts': processed_texts,
            'clinical': clinical_tensor,
            'diagnosis': torch.tensor(case['diagnosis'], dtype=torch.long),
            'severity': torch.tensor(case['severity'], dtype=torch.long)
        }

# 크로스 모달 어텐션 메커니즘
class CrossModalAttention(nn.Module):
    def __init__(self, vision_dim, text_dim, hidden_dim):
        super(CrossModalAttention, self).__init__()

        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)

    def forward(self, vision_features, text_features):
        # 프로젝션
        vision_proj = self.vision_proj(vision_features)  # [B, N_v, hidden_dim]
        text_proj = self.text_proj(text_features)        # [B, N_t, hidden_dim]

        # 크로스 어텐션: 텍스트 쿼리로 비전 정보 참조
        attended_vision, attn_weights = self.attention(text_proj, vision_proj, vision_proj)

        return attended_vision, attn_weights

# 다중 모달 융합 네트워크
class MultimodalFusionNet(nn.Module):
    def __init__(self, vocab_size, num_classes=5, clinical_dim=5):
        super(MultimodalFusionNet, self).__init__()

        # 비전 인코더 (여러 이미지 타입 지원)
        self.vision_encoders = nn.ModuleDict({
            'chest_xray': self._create_vision_encoder(),
            'chest_ct': self._create_vision_encoder(),
            'h_e': self._create_vision_encoder(),
            'ihc': self._create_vision_encoder()
        })

        # 텍스트 인코더
        self.text_encoder = nn.LSTM(
            input_size=128,  # 임베딩 차원
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.text_embedding = nn.Embedding(vocab_size, 128, padding_idx=0)

        # 임상 데이터 인코더
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # 크로스 모달 어텐션
        self.cross_attention = CrossModalAttention(
            vision_dim=512,
            text_dim=512,  # bidirectional LSTM: 256*2
            hidden_dim=512
        )

        # 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(512 + 512 + 512, 512),  # vision + text + clinical
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 분류기들
        self.diagnosis_classifier = nn.Linear(256, num_classes)
        self.severity_classifier = nn.Linear(256, 4)  # 0-3 중증도

        # 모달리티 중요도 학습
        self.modality_attention = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # vision, text, clinical
            nn.Softmax(dim=1)
        )

    def _create_vision_encoder(self):
        return nn.Sequential(
            # CNN 백본
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),

            # ResNet-like blocks
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2) if stride > 1 else nn.Identity()
        )

    def forward(self, images, texts, clinical_data):
        batch_size = clinical_data.size(0)

        # 1. 비전 특징 추출
        vision_features = []
        for img_type, img_tensor in images.items():
            if img_type in self.vision_encoders and img_tensor.size(0) > 0:
                features = self.vision_encoders[img_type](img_tensor)
                vision_features.append(features)

        if vision_features:
            # 여러 이미지의 특징을 결합
            vision_combined = torch.stack(vision_features, dim=1)  # [B, num_images, 512]
            vision_pooled = vision_combined.mean(dim=1)  # [B, 512]
        else:
            vision_pooled = torch.zeros(batch_size, 512, device=clinical_data.device)

        # 2. 텍스트 특징 추출
        text_features_list = []
        for text_type, text_tensor in texts.items():
            if text_tensor.size(0) > 0:
                # 임베딩 및 LSTM
                embedded = self.text_embedding(text_tensor)  # [B, seq_len, 128]
                lstm_out, (hidden, cell) = self.text_encoder(embedded)

                # 마지막 레이어의 양방향 히든 스테이트를 결합
                # hidden shape: (num_layers * 2, B, hidden_size)
                # 마지막 레이어의 forward (hidden[-2])와 backward (hidden[-1])를 결합
                text_feature = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1) # [B, 512]
                text_features_list.append(text_feature)

        if text_features_list:
            text_combined = torch.stack(text_features_list, dim=1).mean(dim=1)  # [B, 512]
        else:
            text_combined = torch.zeros(batch_size, 512, device=clinical_data.device)

        # 3. 임상 데이터 특징 추출
        if clinical_data.size(1) > 1:
            clinical_features = self.clinical_encoder(clinical_data)  # [B, 512]
        else:
            clinical_features = torch.zeros(batch_size, 512, device=clinical_data.device)

        # 4. 크로스 모달 어텐션 (선택적)
        if vision_features and text_features_list:
            vision_reshaped = vision_combined.view(batch_size, -1, 512)
            text_reshaped = text_combined.unsqueeze(1)  # [B, 1, 512]

            attended_vision, attention_weights = self.cross_attention(
                vision_reshaped, text_reshaped
            )
            vision_attended = attended_vision.squeeze(1)
        else:
            vision_attended = vision_pooled
            attention_weights = None

        # 5. 특징 융합
        fused_features = torch.cat([
            vision_attended,
            text_combined,
            clinical_features
        ], dim=1)

        fused_features = self.fusion_layer(fused_features)

        # 6. 모달리티 중요도 학습
        modality_weights = self.modality_attention(fused_features)

        # 가중 평균 적용
        weighted_features = (
            modality_weights[:, 0:1] * vision_attended +
            modality_weights[:, 1:2] * text_combined +
            modality_weights[:, 2:3] * clinical_features
        )

        # 최종 분류
        diagnosis_logits = self.diagnosis_classifier(fused_features)
        severity_logits = self.severity_classifier(fused_features)

        return {
            'diagnosis_logits': diagnosis_logits,
            'severity_logits': severity_logits,
            'modality_weights': modality_weights,
            'attention_weights': attention_weights,
            'features': fused_features
        }

def collate_multimodal(batch):
    """다중 모달 배치 처리"""
    # 모든 이미지 타입 수집
    all_image_types = set()
    for sample in batch:
        all_image_types.update(sample['images'].keys())

    # 배치 구성
    batched_images = {}
    for img_type in all_image_types:
        imgs = []
        for sample in batch:
            if img_type in sample['images']:
                imgs.append(sample['images'][img_type])
        if imgs:
            batched_images[img_type] = torch.stack(imgs)

    # 텍스트 배치
    all_text_types = set()
    for sample in batch:
        all_text_types.update(sample['texts'].keys())

    batched_texts = {}
    for text_type in all_text_types:
        texts = []
        for sample in batch:
            if text_type in sample['texts']:
                texts.append(sample['texts'][text_type])
        if texts:
            batched_texts[text_type] = torch.stack(texts)

    # 임상 데이터 배치
    clinical_data = torch.stack([sample['clinical'] for sample in batch])
    diagnoses = torch.stack([sample['diagnosis'] for sample in batch])
    severities = torch.stack([sample['severity'] for sample in batch])

    return {
        'images': batched_images,
        'texts': batched_texts,
        'clinical': clinical_data,
        'diagnosis': diagnoses,
        'severity': severities
    }

def train_multimodal_fusion(dataset_type='chest_radiology', num_epochs=50, batch_size=8, lr=0.001):
    """
    다중 모달 융합 시스템 훈련

    Args:
        dataset_type: 데이터셋 타입
        num_epochs: 훈련 에포크 수
        batch_size: 배치 크기
        lr: 학습률
    """

    # 로거 설정
    logger = create_logger_for_multimodal_medical('fusion', dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # 데이터셋 준비
    dataset = MultimodalMedicalDataset(data_type=dataset_type)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_multimodal, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_multimodal, drop_last=True)

    # 모델 설정
    sample = dataset[0]
    clinical_dim = sample['clinical'].shape[0]

    model = MultimodalFusionNet(
        vocab_size=dataset.vocab_size,
        num_classes=5,
        clinical_dim=clinical_dim
    ).to(device)

    # 손실 함수
    criterion_diagnosis = nn.CrossEntropyLoss()
    criterion_severity = nn.CrossEntropyLoss()

    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)

    # 훈련 메트릭 저장
    train_losses = []
    val_losses = []
    diagnosis_accuracies = []
    severity_accuracies = []

    logger.log("Starting multimodal fusion training...")

    for epoch in range(num_epochs):
        # 훈련 단계
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            # 배치를 device로 이동
            images = {k: v.to(device) for k, v in batch['images'].items()}
            texts = {k: v.to(device) for k, v in batch['texts'].items()}
            clinical = batch['clinical'].to(device)
            diagnosis = batch['diagnosis'].to(device)
            severity = batch['severity'].to(device)

            optimizer.zero_grad()

            # 순전파
            outputs = model(images, texts, clinical)

            # 손실 계산
            loss_diagnosis = criterion_diagnosis(outputs['diagnosis_logits'], diagnosis)
            loss_severity = criterion_severity(outputs['severity_logits'], severity)

            total_loss = loss_diagnosis + 0.5 * loss_severity

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if batch_idx % 20 == 0:
                logger.log(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                          f'Loss: {total_loss.item():.4f}')

        # 검증 단계
        model.eval()
        val_loss = 0.0
        correct_diagnosis = 0
        correct_severity = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                images = {k: v.to(device) for k, v in batch['images'].items()}
                texts = {k: v.to(device) for k, v in batch['texts'].items()}
                clinical = batch['clinical'].to(device)
                diagnosis = batch['diagnosis'].to(device)
                severity = batch['severity'].to(device)

                outputs = model(images, texts, clinical)

                # 손실 계산
                loss_diagnosis = criterion_diagnosis(outputs['diagnosis_logits'], diagnosis)
                loss_severity = criterion_severity(outputs['severity_logits'], severity)
                total_loss = loss_diagnosis + 0.5 * loss_severity

                val_loss += total_loss.item()

                # 정확도 계산
                pred_diagnosis = torch.argmax(outputs['diagnosis_logits'], dim=1)
                pred_severity = torch.argmax(outputs['severity_logits'], dim=1)

                correct_diagnosis += (pred_diagnosis == diagnosis).sum().item()
                correct_severity += (pred_severity == severity).sum().item()
                total_samples += diagnosis.size(0)

        # 메트릭 계산
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        diagnosis_accuracy = correct_diagnosis / total_samples
        severity_accuracy = correct_severity / total_samples

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        diagnosis_accuracies.append(diagnosis_accuracy)
        severity_accuracies.append(severity_accuracy)

        logger.log(f'Epoch [{epoch+1}/{num_epochs}]')
        logger.log(f'Train Loss: {avg_train_loss:.4f}')
        logger.log(f'Val Loss: {avg_val_loss:.4f}')
        logger.log(f'Diagnosis Accuracy: {diagnosis_accuracy:.4f}')
        logger.log(f'Severity Accuracy: {severity_accuracy:.4f}')

        # 스케줄러 업데이트
        scheduler.step(avg_val_loss)

        # 메트릭 저장
        logger.save_metrics({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'diagnosis_accuracy': diagnosis_accuracy,
            'severity_accuracy': severity_accuracy,
        })

        # 중간 결과 저장 (매 10 에포크)
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 샘플 배치 분석
                sample_batch = next(iter(val_loader))
                images = {k: v.to(device) for k, v in sample_batch['images'].items()}
                texts = {k: v.to(device) for k, v in sample_batch['texts'].items()}
                clinical = sample_batch['clinical'].to(device)

                outputs = model(images, texts, clinical)

                # 모달리티 가중치 분석
                modality_weights = outputs['modality_weights'].cpu().numpy()

                # 시각화를 위한 이미지 준비
                vis_images = []
                if 'chest_xray' in images:
                    for i in range(min(4, images['chest_xray'].size(0))):
                        img = images['chest_xray'][i].cpu().numpy().transpose(1, 2, 0)
                        img = (img - img.min()) / (img.max() - img.min())
                        vis_images.append(img)

                if vis_images:
                    # 모달리티 가중치를 제목에 포함
                    titles = []
                    for i in range(len(vis_images)):
                        weights = modality_weights[i]
                        titles.append(f'V:{weights[0]:.2f} T:{weights[1]:.2f} C:{weights[2]:.2f}')

                    logger.save_image_grid(vis_images,
                                         f'multimodal_fusion_epoch_{epoch+1}.png',
                                         titles=titles,
                                         nrow=2)

    # 최종 모델 저장
    logger.save_model(model, "multimodal_fusion_final",
                     optimizer=optimizer, epoch=num_epochs,
                     config={'dataset_type': dataset_type, 'vocab_size': dataset.vocab_size})

    # 훈련 곡선 저장
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(diagnosis_accuracies, label='Diagnosis')
    plt.plot(severity_accuracies, label='Severity')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 3)
    # 모달리티 가중치 분포
    with torch.no_grad():
        model.eval()
        all_weights = []
        for batch in val_loader:
            images = {k: v.to(device) for k, v in batch['images'].items()}
            texts = {k: v.to(device) for k, v in batch['texts'].items()}
            clinical = batch['clinical'].to(device)

            outputs = model(images, texts, clinical)
            all_weights.append(outputs['modality_weights'].cpu().numpy())
            break  # 첫 번째 배치만

        weights = np.concatenate(all_weights, axis=0)
        vision_weights = weights[:, 0]
        text_weights = weights[:, 1]
        clinical_weights = weights[:, 2]

        plt.hist(vision_weights, alpha=0.5, label='Vision', bins=20)
        plt.hist(text_weights, alpha=0.5, label='Text', bins=20)
        plt.hist(clinical_weights, alpha=0.5, label='Clinical', bins=20)
        plt.title('Modality Weight Distribution')
        plt.xlabel('Weight')
        plt.ylabel('Frequency')
        plt.legend()

    plt.subplot(1, 4, 4)
    # 모달리티별 평균 가중치
    mean_weights = weights.mean(axis=0)
    modalities = ['Vision', 'Text', 'Clinical']
    plt.bar(modalities, mean_weights)
    plt.title('Average Modality Weights')
    plt.ylabel('Weight')
    for i, v in enumerate(mean_weights):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'multimodal_fusion_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logger.log("Multimodal fusion training completed successfully!")
    logger.log(f"Final diagnosis accuracy: {diagnosis_accuracies[-1]:.4f}")
    logger.log(f"Final severity accuracy: {severity_accuracies[-1]:.4f}")
    logger.log(f"Results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

if __name__ == "__main__":
    print("🔗 다중 모달 의료 AI 융합 (Multimodal Medical AI Fusion)")
    print("=" * 60)

    # 하이퍼파라미터 설정
    config = {
        'dataset_type': 'chest_radiology',
        'num_epochs': 5,  # 빠른 테스트를 위해 5로 설정
        'batch_size': 4,  # 다중 모달은 메모리 사용량이 높음
        'lr': 0.001
    }

    print(f"Dataset: {config['dataset_type']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print()

    try:
        model, results_dir = train_multimodal_fusion(
            dataset_type=config['dataset_type'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )

        print("\n✅ Multimodal fusion training completed successfully!")
        print(f"📁 Results saved to: {results_dir}")

        print("\n📊 Generated files include:")
        print("- images/: Multimodal fusion visualizations with modality weights")
        print("- models/: Trained multimodal fusion model")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves and modality analysis")
        print("- metrics/: Training metrics in JSON format")

        print("\n🎯 Multimodal Fusion Features:")
        print("- Vision-Language-Clinical data integration")
        print("- Cross-modal attention mechanisms")
        print("- Adaptive modality weighting")
        print("- Multi-task learning (diagnosis + severity)")

    except Exception as e:
        print(f"\n❌ Error during multimodal fusion training: {str(e)}")
        import traceback
        traceback.print_exc()