#!/usr/bin/env python3
"""
임상 의사결정 지원 시스템 (Clinical Decision Support System)

임상 의사결정 지원 시스템은 의료진이 진단, 치료 계획 수립, 예후 예측 등에서
보다 정확하고 신속한 의사결정을 내릴 수 있도록 지원하는 AI 시스템입니다.

주요 기능:
- 다중 모달리티 진단 지원
- 치료 계획 추천 시스템
- 위험도 층화 (Risk Stratification)
- 예후 예측 (Prognosis Prediction)
- 임상 가이드라인 준수 검증
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
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
sys.path.append('/workspace/Vision-101')
from result_logger import create_logger_for_clinical_ai

class ClinicalDataset(Dataset):
    def __init__(self, data_type='cardiology', transform=None):
        """
        임상 데이터셋

        Args:
            data_type: 'cardiology', 'oncology', 'radiology', 'emergency'
            transform: 이미지 전처리
        """
        self.data_type = data_type
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 임상 데이터 생성
        self.cases = self._generate_clinical_cases()

    def _generate_clinical_cases(self):
        """임상 사례 데이터 생성"""
        cases = []

        for i in range(800):
            case = self._create_clinical_case(i)
            cases.append(case)

        return cases

    def _create_clinical_case(self, seed):
        """임상 사례 생성"""
        np.random.seed(seed)

        if self.data_type == 'cardiology':
            return self._create_cardiology_case(seed)
        elif self.data_type == 'oncology':
            return self._create_oncology_case(seed)
        elif self.data_type == 'radiology':
            return self._create_radiology_case(seed)
        else:
            return self._create_emergency_case(seed)

    def _create_cardiology_case(self, seed):
        """심장내과 사례 생성"""
        np.random.seed(seed)

        # 환자 기본 정보
        age = np.random.randint(30, 90)
        gender = np.random.choice([0, 1])  # 0: 여성, 1: 남성
        bmi = np.random.normal(26, 5)

        # 임상 검사 수치
        total_cholesterol = np.random.normal(200, 40)
        ldl_cholesterol = np.random.normal(120, 30)
        hdl_cholesterol = np.random.normal(50, 15)
        triglycerides = np.random.normal(150, 50)
        systolic_bp = np.random.normal(130, 20)
        diastolic_bp = np.random.normal(80, 10)
        heart_rate = np.random.normal(75, 15)

        # 심전도 특징 (시뮬레이션)
        ecg_features = {
            'pr_interval': np.random.normal(160, 20),
            'qrs_duration': np.random.normal(90, 15),
            'qt_interval': np.random.normal(400, 30),
            'st_elevation': np.random.choice([0, 1], p=[0.9, 0.1]),
            'st_depression': np.random.choice([0, 1], p=[0.85, 0.15]),
            't_wave_inversion': np.random.choice([0, 1], p=[0.8, 0.2])
        }

        # 심장 초음파 소견
        echo_features = {
            'ejection_fraction': np.random.normal(55, 10),
            'lv_mass_index': np.random.normal(95, 20),
            'e_prime': np.random.normal(8, 3),
            'mitral_regurgitation': np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.15, 0.05])
        }

        # 위험 인자
        risk_factors = {
            'diabetes': np.random.choice([0, 1], p=[0.8, 0.2]),
            'hypertension': np.random.choice([0, 1], p=[0.6, 0.4]),
            'smoking': np.random.choice([0, 1], p=[0.7, 0.3]),
            'family_history': np.random.choice([0, 1], p=[0.85, 0.15])
        }

        # 진단 (0: 정상, 1: 관상동맥질환, 2: 심부전, 3: 부정맥)
        # 위험 인자에 따른 확률적 진단
        risk_score = (
            (age > 65) * 0.2 +
            gender * 0.1 +
            (bmi > 30) * 0.1 +
            risk_factors['diabetes'] * 0.3 +
            risk_factors['hypertension'] * 0.2 +
            risk_factors['smoking'] * 0.2 +
            (systolic_bp > 140) * 0.2 +
            (total_cholesterol > 240) * 0.1 +
            ecg_features['st_elevation'] * 0.4 +
            (echo_features['ejection_fraction'] < 45) * 0.3
        )

        if risk_score > 0.8:
            diagnosis = 1  # 관상동맥질환
        elif risk_score > 0.6:
            diagnosis = 2  # 심부전
        elif risk_score > 0.4:
            diagnosis = 3  # 부정맥
        else:
            diagnosis = 0  # 정상

        # 치료 권고 생성
        treatment_recommendations = self._generate_cardiology_treatment(
            diagnosis, risk_factors, echo_features['ejection_fraction']
        )

        # 예후 예측 (1년 내 주요 심혈관 사건 위험도)
        prognosis_risk = min(1.0, risk_score * 0.3 + np.random.normal(0, 0.1))

        # ECG 이미지 시뮬레이션
        ecg_image = self._generate_ecg_image(seed, ecg_features)

        return {
            'image': ecg_image,
            'demographics': [age, gender, bmi],
            'lab_values': [total_cholesterol, ldl_cholesterol, hdl_cholesterol,
                          triglycerides, systolic_bp, diastolic_bp, heart_rate],
            'ecg_features': list(ecg_features.values()),
            'echo_features': list(echo_features.values()),
            'risk_factors': list(risk_factors.values()),
            'diagnosis': diagnosis,
            'treatment_recommendations': treatment_recommendations,
            'prognosis_risk': prognosis_risk
        }

    def _create_oncology_case(self, seed):
        """종양내과 사례 생성"""
        np.random.seed(seed)

        # 환자 기본 정보
        age = np.random.randint(40, 85)
        gender = np.random.choice([0, 1])

        # 종양 마커
        cea = np.random.lognormal(1, 1)  # CEA
        ca_125 = np.random.lognormal(2, 0.5)  # CA-125
        psa = np.random.lognormal(0.5, 0.8) if gender == 1 else 0  # PSA (남성만)

        # 영상 소견
        tumor_size = np.random.exponential(3)  # cm
        lymph_nodes = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05])
        metastasis = np.random.choice([0, 1], p=[0.7, 0.3])

        # TNM 병기 계산
        if tumor_size < 2:
            t_stage = 1
        elif tumor_size < 5:
            t_stage = 2
        else:
            t_stage = 3

        n_stage = lymph_nodes
        m_stage = metastasis

        # 전체 병기 (0-IV)
        if m_stage == 1:
            stage = 4
        elif t_stage >= 3 or n_stage >= 2:
            stage = 3
        elif t_stage == 2 or n_stage == 1:
            stage = 2
        else:
            stage = 1

        # 치료 반응 예측
        performance_status = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
        treatment_response = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])  # 0: 진행, 1: 안정, 2: 반응

        # CT 이미지 시뮬레이션
        ct_image = self._generate_ct_image(seed, tumor_size, metastasis)

        return {
            'image': ct_image,
            'demographics': [age, gender],
            'tumor_markers': [cea, ca_125, psa],
            'imaging_features': [tumor_size, lymph_nodes, metastasis],
            'tnm_stage': [t_stage, n_stage, m_stage],
            'overall_stage': stage,
            'performance_status': performance_status,
            'diagnosis': min(3, stage),  # 0-3으로 매핑
            'treatment_response': treatment_response,
            'prognosis_risk': min(1.0, stage * 0.2 + (4 - performance_status) * 0.1)
        }

    def _create_radiology_case(self, seed):
        """영상의학과 사례 생성"""
        np.random.seed(seed)

        # 영상 유형
        imaging_modality = np.random.choice(['ct', 'mri', 'xray', 'ultrasound'])

        # 소견
        findings = {
            'mass_lesion': np.random.choice([0, 1], p=[0.7, 0.3]),
            'calcification': np.random.choice([0, 1], p=[0.8, 0.2]),
            'enhancement': np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2]),  # 0: none, 1: mild, 2: strong
            'necrosis': np.random.choice([0, 1], p=[0.85, 0.15]),
            'hemorrhage': np.random.choice([0, 1], p=[0.9, 0.1]),
            'edema': np.random.choice([0, 1], p=[0.75, 0.25])
        }

        # BIRADS 또는 유사한 분류 시스템
        risk_score = (
            findings['mass_lesion'] * 2 +
            findings['calcification'] * 1 +
            findings['enhancement'] * 1.5 +
            findings['necrosis'] * 1 +
            findings['hemorrhage'] * 0.5
        )

        if risk_score >= 4:
            diagnosis = 3  # 고위험
        elif risk_score >= 2:
            diagnosis = 2  # 중등도 위험
        elif risk_score >= 1:
            diagnosis = 1  # 저위험
        else:
            diagnosis = 0  # 정상/양성

        # 영상 시뮬레이션
        image = self._generate_radiology_image(seed, imaging_modality, findings)

        return {
            'image': image,
            'imaging_modality': ['ct', 'mri', 'xray', 'ultrasound'].index(imaging_modality),
            'findings': list(findings.values()),
            'diagnosis': diagnosis,
            'prognosis_risk': diagnosis * 0.25
        }

    def _create_emergency_case(self, seed):
        """응급의학과 사례 생성"""
        np.random.seed(seed)

        # 환자 기본 정보
        age = np.random.randint(18, 95)
        gender = np.random.choice([0, 1])

        # 활력징후
        systolic_bp = np.random.normal(120, 30)
        diastolic_bp = np.random.normal(80, 15)
        heart_rate = np.random.normal(80, 25)
        respiratory_rate = np.random.normal(18, 5)
        temperature = np.random.normal(36.5, 1)
        oxygen_saturation = np.random.normal(98, 3)

        # 의식 수준 (GCS)
        gcs_eye = np.random.choice([1, 2, 3, 4], p=[0.05, 0.1, 0.15, 0.7])
        gcs_verbal = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.1, 0.15, 0.6])
        gcs_motor = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.05, 0.05, 0.1, 0.1, 0.2, 0.5])
        total_gcs = gcs_eye + gcs_verbal + gcs_motor

        # 검사실 수치
        hemoglobin = np.random.normal(12, 2)
        white_cell_count = np.random.lognormal(2, 0.5)
        creatinine = np.random.lognormal(0, 0.3)
        lactate = np.random.exponential(2)

        # 중증도 점수 계산 (modified NEWS 또는 유사)
        severity_score = (
            (systolic_bp < 90 or systolic_bp > 180) * 2 +
            (heart_rate < 50 or heart_rate > 120) * 1 +
            (respiratory_rate > 24) * 2 +
            (temperature < 35 or temperature > 38.5) * 1 +
            (oxygen_saturation < 95) * 2 +
            (total_gcs < 13) * 3 +
            (lactate > 4) * 2
        )

        # 진단 (0: 경증, 1: 중등증, 2: 중증, 3: 최중증)
        if severity_score >= 8:
            diagnosis = 3
        elif severity_score >= 5:
            diagnosis = 2
        elif severity_score >= 3:
            diagnosis = 1
        else:
            diagnosis = 0

        # 응급 영상 시뮬레이션 (흉부 X-ray)
        emergency_image = self._generate_emergency_image(seed, severity_score)

        return {
            'image': emergency_image,
            'demographics': [age, gender],
            'vital_signs': [systolic_bp, diastolic_bp, heart_rate,
                           respiratory_rate, temperature, oxygen_saturation],
            'gcs': [gcs_eye, gcs_verbal, gcs_motor],
            'lab_values': [hemoglobin, white_cell_count, creatinine, lactate],
            'severity_score': severity_score,
            'diagnosis': diagnosis,
            'prognosis_risk': diagnosis * 0.3
        }

    def _generate_cardiology_treatment(self, diagnosis, risk_factors, ef):
        """심장내과 치료 권고 생성"""
        treatments = []

        if diagnosis == 1:  # 관상동맥질환
            treatments.extend([
                "Aspirin 100mg daily",
                "Statin therapy",
                "ACE inhibitor or ARB"
            ])
            if risk_factors['diabetes']:
                treatments.append("Diabetes management")

        elif diagnosis == 2:  # 심부전
            treatments.extend([
                "ACE inhibitor or ARB",
                "Beta-blocker",
                "Diuretics if needed"
            ])
            if ef < 35:
                treatments.append("Consider ICD/CRT")

        elif diagnosis == 3:  # 부정맥
            treatments.extend([
                "Rate/rhythm control",
                "Anticoagulation if indicated"
            ])

        return treatments

    def _generate_ecg_image(self, seed, ecg_features):
        """ECG 이미지 시뮬레이션"""
        np.random.seed(seed)
        image = np.zeros((224, 224, 3))

        # ECG 파형 시뮬레이션
        for lead in range(12):
            y_offset = 20 + lead * 15
            if y_offset < 220:
                # 기본 심전도 파형
                for x in range(220):
                    if ecg_features['st_elevation'] and lead < 6:
                        # ST 분절 상승
                        y = y_offset + 5 + np.sin(x * 0.1) * 2
                    elif ecg_features['st_depression'] and lead >= 6:
                        # ST 분절 하강
                        y = y_offset - 3 + np.sin(x * 0.1) * 2
                    else:
                        y = y_offset + np.sin(x * 0.2) * 3

                    if 0 <= int(y) < 224:
                        image[int(y), x] = [1, 1, 1]

        return (image * 255).astype(np.uint8)

    def _generate_ct_image(self, seed, tumor_size, metastasis):
        """CT 이미지 시뮬레이션"""
        np.random.seed(seed)
        image = np.random.normal(0.3, 0.1, (224, 224, 3))

        # 종양 시뮬레이션
        if tumor_size > 0:
            tumor_x = np.random.randint(50, 174)
            tumor_y = np.random.randint(50, 174)
            tumor_radius = int(tumor_size * 5)  # 픽셀 단위로 변환

            for y in range(max(0, tumor_y - tumor_radius),
                          min(224, tumor_y + tumor_radius)):
                for x in range(max(0, tumor_x - tumor_radius),
                              min(224, tumor_x + tumor_radius)):
                    dist = np.sqrt((x - tumor_x)**2 + (y - tumor_y)**2)
                    if dist < tumor_radius:
                        image[y, x] = 0.7  # 고밀도 병변

        # 전이 시뮬레이션
        if metastasis:
            num_mets = np.random.randint(2, 6)
            for _ in range(num_mets):
                met_x = np.random.randint(20, 204)
                met_y = np.random.randint(20, 204)
                met_size = np.random.randint(3, 10)

                for y in range(max(0, met_y - met_size),
                              min(224, met_y + met_size)):
                    for x in range(max(0, met_x - met_size),
                                  min(224, met_x + met_size)):
                        image[y, x] = 0.6

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _generate_radiology_image(self, seed, modality, findings):
        """영상의학 이미지 시뮬레이션"""
        np.random.seed(seed)

        if modality == 'ct':
            base_intensity = 0.3
        elif modality == 'mri':
            base_intensity = 0.5
        elif modality == 'xray':
            base_intensity = 0.4
        else:  # ultrasound
            base_intensity = 0.2

        image = np.random.normal(base_intensity, 0.1, (224, 224, 3))

        # 소견에 따른 병변 추가
        if findings['mass_lesion']:
            mass_x = np.random.randint(60, 164)
            mass_y = np.random.randint(60, 164)
            mass_size = np.random.randint(10, 30)

            for y in range(max(0, mass_y - mass_size),
                          min(224, mass_y + mass_size)):
                for x in range(max(0, mass_x - mass_size),
                              min(224, mass_x + mass_size)):
                    dist = np.sqrt((x - mass_x)**2 + (y - mass_y)**2)
                    if dist < mass_size:
                        intensity_mod = 0.3 if findings['enhancement'] else -0.1
                        image[y, x] += intensity_mod

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _generate_emergency_image(self, seed, severity):
        """응급 영상 시뮬레이션"""
        np.random.seed(seed)
        image = np.zeros((224, 224, 3))

        # 기본 흉부 구조
        # 폐야
        for y in range(50, 180):
            for x in range(30, 194):
                image[y, x] = 0.3 + np.random.normal(0, 0.05)

        # 중증도에 따른 이상 소견
        if severity > 5:
            # 기흉 시뮬레이션
            for y in range(60, 120):
                for x in range(40, 80):
                    image[y, x] = 0.1  # 공기층

        if severity > 3:
            # 폐부종 시뮬레이션
            for y in range(120, 170):
                for x in range(50, 174):
                    if np.random.random() > 0.7:
                        image[y, x] = 0.6

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]

        # 이미지 처리
        image = Image.fromarray(case['image'])
        if self.transform:
            image = self.transform(image)

        # 모든 수치적 특징을 하나의 벡터로 결합
        features = []

        if 'demographics' in case:
            features.extend(case['demographics'])
        if 'lab_values' in case:
            features.extend(case['lab_values'])
        if 'vital_signs' in case:
            features.extend(case['vital_signs'])
        if 'ecg_features' in case:
            features.extend(case['ecg_features'])
        if 'echo_features' in case:
            features.extend(case['echo_features'])
        if 'risk_factors' in case:
            features.extend(case['risk_factors'])
        if 'tumor_markers' in case:
            features.extend(case['tumor_markers'])
        if 'imaging_features' in case:
            features.extend(case['imaging_features'])
        if 'tnm_stage' in case:
            features.extend(case['tnm_stage'])
        if 'gcs' in case:
            features.extend(case['gcs'])
        if 'findings' in case:
            features.extend(case['findings'])

        # 고정 길이로 패딩 (최대 길이를 50으로 설정)
        max_features = 50
        if len(features) < max_features:
            features.extend([0.0] * (max_features - len(features)))
        elif len(features) > max_features:
            features = features[:max_features]

        features = torch.tensor(features, dtype=torch.float32)
        diagnosis = torch.tensor(case['diagnosis'], dtype=torch.long)
        prognosis_risk = torch.tensor(case['prognosis_risk'], dtype=torch.float32)

        # case_data에서 가변 길이 필드를 제외하고 반환
        case_info = {
            'diagnosis_name': case.get('diagnosis', 0),
            'patient_type': self.data_type
        }

        return {
            'image': image,
            'features': features,
            'diagnosis': diagnosis,
            'prognosis_risk': prognosis_risk,
            'case_data': case_info
        }

# 임상 의사결정 지원 네트워크
class ClinicalDecisionSupportNet(nn.Module):
    def __init__(self, num_clinical_features, num_classes=4, image_features=512):
        super(ClinicalDecisionSupportNet, self).__init__()

        # 이미지 특징 추출기 (CNN)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),

            # ResNet-like blocks
            self._make_layer(64, 128, 2),
            self._make_layer(128, 256, 2),
            self._make_layer(256, 512, 2),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # 임상 특징 처리기
        self.clinical_encoder = nn.Sequential(
            nn.Linear(num_clinical_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 융합 레이어
        self.fusion_layer = nn.Sequential(
            nn.Linear(image_features + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 진단 분류기
        self.diagnosis_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # 예후 예측기
        self.prognosis_predictor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 신뢰도 추정기
        self.confidence_estimator = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
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

    def forward(self, images, clinical_features):
        # 이미지 특징 추출
        image_features = self.image_encoder(images)
        image_features = image_features.view(image_features.size(0), -1)

        # 임상 특징 처리
        clinical_encoded = self.clinical_encoder(clinical_features)

        # 특징 융합
        fused_features = torch.cat([image_features, clinical_encoded], dim=1)
        fused_features = self.fusion_layer(fused_features)

        # 예측
        diagnosis_logits = self.diagnosis_classifier(fused_features)
        prognosis_risk = self.prognosis_predictor(fused_features)
        confidence = self.confidence_estimator(fused_features)

        return {
            'diagnosis_logits': diagnosis_logits,
            'prognosis_risk': prognosis_risk.squeeze(),
            'confidence': confidence.squeeze()
        }

def train_clinical_decision_support(dataset_type='cardiology', num_epochs=50, batch_size=16, lr=0.001):
    """
    임상 의사결정 지원 시스템 훈련

    Args:
        dataset_type: 임상 데이터셋 타입
        num_epochs: 훈련 에포크 수
        batch_size: 배치 크기
        lr: 학습률
    """

    # 로거 설정
    logger = create_logger_for_clinical_ai('decision_support', dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # 데이터셋 준비
    dataset = ClinicalDataset(data_type=dataset_type)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델 설정
    # 고정된 feature 차원 사용 (패딩된 크기)
    num_clinical_features = 50  # 패딩된 고정 크기

    model = ClinicalDecisionSupportNet(
        num_clinical_features=num_clinical_features,
        num_classes=4
    ).to(device)

    # 손실 함수
    criterion_diagnosis = nn.CrossEntropyLoss()
    criterion_prognosis = nn.MSELoss()
    criterion_confidence = nn.BCELoss()

    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)

    # 훈련 메트릭 저장
    train_losses = []
    val_losses = []
    diagnosis_accuracies = []

    logger.log("Starting clinical decision support training...")

    for epoch in range(num_epochs):
        # 훈련 단계
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            features = batch['features'].to(device)
            diagnosis = batch['diagnosis'].to(device)
            prognosis_risk = batch['prognosis_risk'].to(device)

            optimizer.zero_grad()

            # 순전파
            outputs = model(images, features)

            # 신뢰도 타겟 생성 (진단 정확도 기반)
            with torch.no_grad():
                pred_diagnosis = torch.argmax(outputs['diagnosis_logits'], dim=1)
                confidence_targets = (pred_diagnosis == diagnosis).float()

            # 손실 계산
            loss_diagnosis = criterion_diagnosis(outputs['diagnosis_logits'], diagnosis)
            loss_prognosis = criterion_prognosis(outputs['prognosis_risk'], prognosis_risk)
            loss_confidence = criterion_confidence(outputs['confidence'], confidence_targets)

            total_loss = loss_diagnosis + 0.5 * loss_prognosis + 0.3 * loss_confidence

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
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                features = batch['features'].to(device)
                diagnosis = batch['diagnosis'].to(device)
                prognosis_risk = batch['prognosis_risk'].to(device)

                outputs = model(images, features)

                # 신뢰도 타겟
                pred_diagnosis = torch.argmax(outputs['diagnosis_logits'], dim=1)
                confidence_targets = (pred_diagnosis == diagnosis).float()

                # 손실 계산
                loss_diagnosis = criterion_diagnosis(outputs['diagnosis_logits'], diagnosis)
                loss_prognosis = criterion_prognosis(outputs['prognosis_risk'], prognosis_risk)
                loss_confidence = criterion_confidence(outputs['confidence'], confidence_targets)

                total_loss = loss_diagnosis + 0.5 * loss_prognosis + 0.3 * loss_confidence
                val_loss += total_loss.item()

                # 정확도 계산
                correct_diagnosis += (pred_diagnosis == diagnosis).sum().item()
                total_samples += diagnosis.size(0)

        # 메트릭 계산
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        diagnosis_accuracy = correct_diagnosis / total_samples

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        diagnosis_accuracies.append(diagnosis_accuracy)

        logger.log(f'Epoch [{epoch+1}/{num_epochs}]')
        logger.log(f'Train Loss: {avg_train_loss:.4f}')
        logger.log(f'Val Loss: {avg_val_loss:.4f}')
        logger.log(f'Diagnosis Accuracy: {diagnosis_accuracy:.4f}')

        # 스케줄러 업데이트
        scheduler.step(avg_val_loss)

        # 메트릭 저장
        logger.log_metrics(
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            diagnosis_accuracy=diagnosis_accuracy,
        )

        # 중간 결과 저장 (매 10 에포크)
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # 샘플 예측 및 시각화
                sample_batch = next(iter(val_loader))
                sample_images = sample_batch['image'][:4].to(device)
                sample_features = sample_batch['features'][:4].to(device)
                sample_diagnosis = sample_batch['diagnosis'][:4]

                outputs = model(sample_images, sample_features)
                pred_diagnosis = torch.argmax(outputs['diagnosis_logits'], dim=1)
                pred_prognosis = outputs['prognosis_risk']
                pred_confidence = outputs['confidence']

                # 시각화를 위한 이미지 준비
                vis_images = []
                for i in range(len(sample_images)):
                    # 원본 이미지
                    img = sample_images[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min())
                    vis_images.append(img)

                # 예측 결과 제목 생성
                titles = []
                for i in range(len(sample_images)):
                    true_dx = sample_diagnosis[i].item()
                    pred_dx = pred_diagnosis[i].item()
                    confidence = pred_confidence[i].item()
                    titles.append(f'T:{true_dx} P:{pred_dx} C:{confidence:.2f}')

                logger.save_image_grid(vis_images,
                                     f'clinical_decision_epoch_{epoch+1}.png',
                                     titles=titles,
                                     nrow=2)

    # 최종 모델 저장
    logger.save_model(model, "clinical_decision_support_final",
                     optimizer=optimizer, epoch=num_epochs,
                     config={'dataset_type': dataset_type, 'num_epochs': num_epochs})

    # 훈련 곡선 저장
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Clinical Decision Support Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(diagnosis_accuracies)
    plt.title('Diagnosis Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    # 모델 성능 분석
    model.eval()
    all_confidences = []
    all_accuracies = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            features = batch['features'].to(device)
            diagnosis = batch['diagnosis'].to(device)

            outputs = model(images, features)
            pred_diagnosis = torch.argmax(outputs['diagnosis_logits'], dim=1)

            confidences = outputs['confidence'].cpu().numpy()
            accuracies = (pred_diagnosis == diagnosis).cpu().numpy()

            all_confidences.extend(confidences)
            all_accuracies.extend(accuracies)

    # 신뢰도별 정확도 분석
    conf_bins = np.linspace(0, 1, 11)
    bin_accuracies = []

    for i in range(len(conf_bins) - 1):
        mask = (np.array(all_confidences) >= conf_bins[i]) & (np.array(all_confidences) < conf_bins[i+1])
        if mask.sum() > 0:
            bin_accuracies.append(np.array(all_accuracies)[mask].mean())
        else:
            bin_accuracies.append(0)

    plt.bar(conf_bins[:-1], bin_accuracies, width=0.08, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.title('Model Calibration')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'clinical_decision_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logger.log("Clinical decision support training completed successfully!")
    logger.log(f"Final diagnosis accuracy: {diagnosis_accuracies[-1]:.4f}")
    logger.log(f"Results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

if __name__ == "__main__":
    print("🏥 임상 의사결정 지원 시스템 (Clinical Decision Support)")
    print("=" * 60)

    # 하이퍼파라미터 설정
    config = {
        'dataset_type': 'cardiology',
        'num_epochs': 5,  # 빠른 테스트를 위해 5로 설정
        'batch_size': 8,
        'lr': 0.001
    }

    print(f"Dataset: {config['dataset_type']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print()

    try:
        model, results_dir = train_clinical_decision_support(
            dataset_type=config['dataset_type'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )

        print("\n✅ Clinical decision support training completed successfully!")
        print(f"📁 Results saved to: {results_dir}")

        print("\n📊 Generated files include:")
        print("- images/: Clinical decision support visualizations")
        print("- models/: Trained clinical AI model")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves and calibration analysis")
        print("- metrics/: Training metrics in JSON format")

        print("\n🎯 Clinical Decision Support Features:")
        print("- Multi-modal diagnosis prediction")
        print("- Prognosis risk assessment")
        print("- Model confidence estimation")
        print("- Treatment recommendation support")

    except Exception as e:
        print(f"\n❌ Error during clinical decision support training: {str(e)}")
        import traceback
        traceback.print_exc()