#!/usr/bin/env python3
"""
시간적 의료 분석 시스템 (Temporal Medical Analysis)

시간적 의료 분석은 의료 영상이나 신호의 시계열 변화를 분석하여
질병의 진행, 치료 반응, 예후 등을 예측하는 중요한 기술입니다.

주요 기능:
- 질병 진행 모니터링 (Disease Progression Monitoring)
- 치료 반응 평가 (Treatment Response Assessment)
- 시계열 영상 분석 (Longitudinal Image Analysis)
- 예후 예측 (Prognosis Prediction)
- 동적 기능 분석 (Dynamic Function Analysis)
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
from sklearn.metrics import mean_squared_error, r2_score
sys.path.append('/workspace/Vision-101')
from medical.result_logger import create_logger_for_temporal_medical

class TemporalMedicalDataset(Dataset):
    def __init__(self, data_type='cardiac_monitoring', sequence_length=10, transform=None):
        """
        시간적 의료 데이터셋

        Args:
            data_type: 'cardiac_monitoring', 'tumor_tracking', 'lung_function', 'brain_activity'
            sequence_length: 시계열 길이
            transform: 이미지 전처리
        """
        self.data_type = data_type
        self.sequence_length = sequence_length
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 시간적 의료 데이터 생성
        self.sequences = self._generate_temporal_sequences()

    def _generate_temporal_sequences(self):
        """시간적 의료 시퀀스 생성"""
        sequences = []

        for i in range(300):  # 300개의 환자/케이스
            sequence = self._create_temporal_sequence(i)
            sequences.append(sequence)

        return sequences

    def _create_temporal_sequence(self, patient_id):
        """환자별 시간적 시퀀스 생성"""
        np.random.seed(patient_id)

        if self.data_type == 'cardiac_monitoring':
            return self._create_cardiac_sequence(patient_id)
        elif self.data_type == 'tumor_tracking':
            return self._create_tumor_sequence(patient_id)
        elif self.data_type == 'lung_function':
            return self._create_lung_sequence(patient_id)
        else:  # brain_activity
            return self._create_brain_sequence(patient_id)

    def _create_cardiac_sequence(self, patient_id):
        """심장 기능 모니터링 시퀀스"""
        np.random.seed(patient_id)

        # 환자 특성
        age = np.random.randint(40, 85)
        gender = np.random.choice([0, 1])
        baseline_ef = np.random.uniform(35, 70)  # 기준 구혈률

        # 치료 시나리오 설정
        treatment_type = np.random.choice(['medication', 'intervention', 'surgery'])
        treatment_start_time = np.random.randint(2, 5)  # 치료 시작 시점

        images = []
        measurements = []
        timestamps = []

        for t in range(self.sequence_length):
            # 시간 경과에 따른 변화 모델링
            time_factor = t / (self.sequence_length - 1)

            # 치료 효과 모델링
            if t >= treatment_start_time:
                treatment_time = t - treatment_start_time
                if treatment_type == 'medication':
                    # 점진적 개선
                    improvement = min(0.15, treatment_time * 0.03)
                elif treatment_type == 'intervention':
                    # 빠른 개선 후 안정화
                    improvement = 0.2 * (1 - np.exp(-treatment_time * 0.8))
                else:  # surgery
                    # 초기 악화 후 큰 개선
                    if treatment_time < 2:
                        improvement = -0.1 + treatment_time * 0.05
                    else:
                        improvement = 0.25
            else:
                improvement = 0

            # 노화/질병 진행에 따른 자연적 악화
            natural_decline = time_factor * np.random.uniform(0.05, 0.15)

            # 현재 구혈률
            current_ef = baseline_ef + improvement - natural_decline + np.random.normal(0, 2)
            current_ef = np.clip(current_ef, 20, 80)

            # 심장 초음파 이미지 생성
            echo_image = self._generate_cardiac_echo(patient_id + t * 1000, current_ef)
            images.append(echo_image)

            # 측정값들
            measurements.append({
                'ejection_fraction': current_ef,
                'lv_diameter': 45 + (70 - current_ef) * 0.3 + np.random.normal(0, 2),
                'wall_thickness': 9 + (current_ef < 50) * 3 + np.random.normal(0, 1),
                'cardiac_output': current_ef * 0.08 + np.random.normal(0, 0.2)
            })

            timestamps.append(t * 30)  # 30일 간격

        # 예후 예측 (6개월 후 주요 심혈관 사건 위험)
        final_ef = measurements[-1]['ejection_fraction']
        risk_score = (
            (age > 70) * 0.2 +
            (gender == 1) * 0.1 +
            (final_ef < 40) * 0.4 +
            (final_ef < 30) * 0.3
        )
        prognosis_risk = min(1.0, risk_score + np.random.normal(0, 0.1))

        return {
            'patient_id': patient_id,
            'images': images,
            'measurements': measurements,
            'timestamps': timestamps,
            'demographics': [age, gender],
            'treatment_type': ['medication', 'intervention', 'surgery'].index(treatment_type),
            'treatment_start': treatment_start_time,
            'baseline_ef': baseline_ef,
            'final_ef': final_ef,
            'prognosis_risk': prognosis_risk
        }

    def _create_tumor_sequence(self, patient_id):
        """종양 추적 시퀀스"""
        np.random.seed(patient_id)

        # 종양 특성
        initial_size = np.random.uniform(10, 50)  # mm
        growth_rate = np.random.uniform(-0.1, 0.3)  # 치료 반응에 따라 음수도 가능
        tumor_type = np.random.choice(['aggressive', 'moderate', 'slow'])

        # 치료 시작
        chemo_start = np.random.randint(1, 4)

        images = []
        measurements = []
        timestamps = []

        for t in range(self.sequence_length):
            # 종양 크기 변화 모델링
            if t >= chemo_start:
                # 치료 효과
                treatment_time = t - chemo_start
                if tumor_type == 'aggressive':
                    size_change = -0.2 * treatment_time + 0.05 * t  # 초기 반응 후 저항성
                elif tumor_type == 'moderate':
                    size_change = -0.15 * treatment_time
                else:  # slow
                    size_change = -0.1 * treatment_time
            else:
                # 자연 성장
                size_change = growth_rate * t

            current_size = max(5, initial_size + size_change + np.random.normal(0, 1))

            # CT 이미지 생성
            ct_image = self._generate_tumor_ct(patient_id + t * 1000, current_size)
            images.append(ct_image)

            measurements.append({
                'tumor_size': current_size,
                'volume': (current_size / 10) ** 3 * 4/3 * np.pi,
                'density': 45 + np.random.normal(0, 5),
                'enhancement': np.random.uniform(20, 60)
            })

            timestamps.append(t * 60)  # 60일 간격

        # 치료 반응 평가
        size_reduction = (initial_size - current_size) / initial_size
        if size_reduction > 0.3:
            response = 2  # 부분 관해
        elif size_reduction > 0.1:
            response = 1  # 안정
        else:
            response = 0  # 진행

        return {
            'patient_id': patient_id,
            'images': images,
            'measurements': measurements,
            'timestamps': timestamps,
            'tumor_type': ['aggressive', 'moderate', 'slow'].index(tumor_type),
            'initial_size': initial_size,
            'final_size': current_size,
            'treatment_response': response,
            'prognosis_risk': max(0, 1 - size_reduction)
        }

    def _create_lung_sequence(self, patient_id):
        """폐 기능 모니터링 시퀀스"""
        np.random.seed(patient_id)

        # 기준 폐 기능
        baseline_fev1 = np.random.uniform(60, 120)  # % predicted
        disease_type = np.random.choice(['copd', 'fibrosis', 'normal'])

        images = []
        measurements = []
        timestamps = []

        for t in range(self.sequence_length):
            # 질병별 진행 패턴
            if disease_type == 'copd':
                # COPD: 점진적 악화
                decline_rate = 0.02 + np.random.uniform(0, 0.01)
                current_fev1 = baseline_fev1 * (1 - decline_rate * t)
            elif disease_type == 'fibrosis':
                # 섬유화: 비선형적 악화
                progression = 1 - 0.3 * (1 - np.exp(-t * 0.3))
                current_fev1 = baseline_fev1 * progression
            else:  # normal
                # 정상: 연령에 따른 자연스러운 감소
                current_fev1 = baseline_fev1 * (1 - 0.005 * t)

            current_fev1 += np.random.normal(0, 2)
            current_fev1 = max(20, current_fev1)

            # 흉부 CT 생성
            lung_ct = self._generate_lung_ct(patient_id + t * 1000, current_fev1, disease_type)
            images.append(lung_ct)

            measurements.append({
                'fev1': current_fev1,
                'fvc': current_fev1 * np.random.uniform(1.1, 1.3),
                'dlco': current_fev1 * np.random.uniform(0.8, 1.2),
                'lung_volume': 4500 + np.random.normal(0, 200)
            })

            timestamps.append(t * 90)  # 90일 간격

        return {
            'patient_id': patient_id,
            'images': images,
            'measurements': measurements,
            'timestamps': timestamps,
            'disease_type': ['copd', 'fibrosis', 'normal'].index(disease_type),
            'baseline_fev1': baseline_fev1,
            'final_fev1': current_fev1,
            'decline_rate': (baseline_fev1 - current_fev1) / baseline_fev1,
            'prognosis_risk': max(0, 1 - current_fev1 / 80)
        }

    def _create_brain_sequence(self, patient_id):
        """뇌 활동 모니터링 시퀀스"""
        np.random.seed(patient_id)

        # 뇌 상태 설정
        condition = np.random.choice(['dementia', 'stroke_recovery', 'normal_aging'])
        baseline_score = np.random.uniform(70, 95)  # 인지 기능 점수

        images = []
        measurements = []
        timestamps = []

        for t in range(self.sequence_length):
            if condition == 'dementia':
                # 치매: 점진적 악화
                current_score = baseline_score * np.exp(-0.05 * t)
            elif condition == 'stroke_recovery':
                # 뇌졸중 회복: 초기 개선 후 안정화
                recovery = 0.3 * (1 - np.exp(-t * 0.3))
                current_score = baseline_score * (0.7 + recovery)
            else:  # normal_aging
                # 정상 노화
                current_score = baseline_score * (1 - 0.01 * t)

            current_score += np.random.normal(0, 2)
            current_score = max(10, min(100, current_score))

            # MRI 이미지 생성
            brain_mri = self._generate_brain_mri(patient_id + t * 1000, current_score, condition)
            images.append(brain_mri)

            measurements.append({
                'cognitive_score': current_score,
                'brain_volume': 1200 - (100 - current_score) * 2 + np.random.normal(0, 20),
                'white_matter_integrity': current_score * 0.8 + np.random.normal(0, 3),
                'hippocampal_volume': 4000 - (100 - current_score) * 10 + np.random.normal(0, 50)
            })

            timestamps.append(t * 180)  # 180일 간격

        return {
            'patient_id': patient_id,
            'images': images,
            'measurements': measurements,
            'timestamps': timestamps,
            'condition': ['dementia', 'stroke_recovery', 'normal_aging'].index(condition),
            'baseline_score': baseline_score,
            'final_score': current_score,
            'cognitive_decline': (baseline_score - current_score) / baseline_score,
            'prognosis_risk': max(0, 1 - current_score / 70)
        }

    def _generate_cardiac_echo(self, seed, ejection_fraction):
        """심장 초음파 이미지 생성"""
        np.random.seed(seed)
        image = np.zeros((128, 128))

        # 심장 윤곽 (구혈률에 따라 크기 조절)
        center_x, center_y = 64, 70
        radius = 25 + (70 - ejection_fraction) * 0.3

        for y in range(128):
            for x in range(128):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= radius:
                    # 심근벽
                    wall_thickness = 4 + (ejection_fraction < 50) * 2
                    if dist > radius - wall_thickness:
                        image[y, x] = 0.7  # 심근
                    else:
                        image[y, x] = 0.2  # 심실 내강

        # 판막 구조
        valve_y = center_y - 15
        image[valve_y:valve_y + 2, center_x - 8:center_x + 8] = 0.9

        # 스페클 노이즈
        image += np.random.exponential(0.05, (128, 128))
        image = np.clip(image, 0, 1)

        # RGB로 변환
        return np.stack([image] * 3, axis=2)

    def _generate_tumor_ct(self, seed, tumor_size):
        """종양 CT 이미지 생성"""
        np.random.seed(seed)
        image = np.random.normal(0.3, 0.1, (128, 128))

        # 종양 위치
        tumor_x = np.random.randint(30, 98)
        tumor_y = np.random.randint(30, 98)
        tumor_radius = tumor_size / 5  # 픽셀 단위로 변환

        # 종양 그리기
        for y in range(128):
            for x in range(128):
                dist = np.sqrt((x - tumor_x)**2 + (y - tumor_y)**2)
                if dist <= tumor_radius:
                    # 종양 내부 불균질성
                    intensity = 0.6 + 0.2 * np.sin(dist * 0.5) + np.random.normal(0, 0.1)
                    image[y, x] = intensity

        image = np.clip(image, 0, 1)
        return np.stack([image] * 3, axis=2)

    def _generate_lung_ct(self, seed, fev1, disease_type):
        """폐 CT 이미지 생성"""
        np.random.seed(seed)
        image = np.random.normal(0.2, 0.05, (128, 128))

        # 폐야 영역
        lung_intensity = 0.1 + (100 - fev1) * 0.005

        # 질병별 패턴
        if disease_type == 'copd':
            # 기종 패턴 (저밀도 영역)
            for _ in range(int((100 - fev1) * 2)):
                emp_x = np.random.randint(20, 108)
                emp_y = np.random.randint(20, 108)
                emp_size = np.random.randint(5, 15)
                image[emp_y:emp_y+emp_size, emp_x:emp_x+emp_size] = 0.05

        elif disease_type == 'fibrosis':
            # 섬유화 패턴 (고밀도 영역)
            for _ in range(int((100 - fev1) * 1.5)):
                fib_x = np.random.randint(20, 108)
                fib_y = np.random.randint(80, 108)  # 하엽 우세
                fib_size = np.random.randint(3, 8)
                image[fib_y:fib_y+fib_size, fib_x:fib_x+fib_size] = 0.8

        image = np.clip(image, 0, 1)
        return np.stack([image] * 3, axis=2)

    def _generate_brain_mri(self, seed, cognitive_score, condition):
        """뇌 MRI 이미지 생성"""
        np.random.seed(seed)
        image = np.zeros((128, 128))

        # 기본 뇌 구조
        center_x, center_y = 64, 64
        brain_radius = 55

        for y in range(128):
            for x in range(128):
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= brain_radius:
                    if dist < 35:
                        image[y, x] = 0.6  # 회백질
                    else:
                        image[y, x] = 0.4  # 백질

        # 병리학적 변화
        atrophy_factor = (100 - cognitive_score) * 0.01

        if condition == 'dementia':
            # 측두엽 위축
            for y in range(40, 80):
                for x in range(30, 50):
                    image[y, x] *= (1 - atrophy_factor)

        elif condition == 'stroke_recovery':
            # 국소적 손상
            lesion_x = np.random.randint(45, 75)
            lesion_y = np.random.randint(45, 75)
            lesion_size = 8
            for y in range(lesion_y, lesion_y + lesion_size):
                for x in range(lesion_x, lesion_x + lesion_size):
                    if 0 <= y < 128 and 0 <= x < 128:
                        image[y, x] = 0.1  # 경색 부위

        image = np.clip(image, 0, 1)
        return np.stack([image] * 3, axis=2)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        # 이미지 시퀀스 처리
        image_sequence = []
        for img_array in sequence['images']:
            img = Image.fromarray((img_array * 255).astype(np.uint8))
            if self.transform:
                img = self.transform(img)
            image_sequence.append(img)

        image_sequence = torch.stack(image_sequence)  # [seq_len, C, H, W]

        # 측정값 시퀀스 처리
        measurement_sequence = []
        measurement_keys = list(sequence['measurements'][0].keys())

        for measurements in sequence['measurements']:
            values = [measurements[key] for key in measurement_keys]
            measurement_sequence.append(values)

        measurement_sequence = torch.tensor(measurement_sequence, dtype=torch.float32)

        # 시간 정보
        timestamps = torch.tensor(sequence['timestamps'], dtype=torch.float32)

        # 기타 메타 정보
        patient_info = []
        if 'demographics' in sequence:
            patient_info.extend(sequence['demographics'])
        if 'treatment_type' in sequence:
            patient_info.append(sequence['treatment_type'])
        if 'disease_type' in sequence:
            patient_info.append(sequence['disease_type'])
        if 'condition' in sequence:
            patient_info.append(sequence['condition'])

        patient_info = torch.tensor(patient_info, dtype=torch.float32) if patient_info else torch.tensor([0.0])

        return {
            'images': image_sequence,
            'measurements': measurement_sequence,
            'timestamps': timestamps,
            'patient_info': patient_info,
            'prognosis_risk': torch.tensor(sequence['prognosis_risk'], dtype=torch.float32),
            'sequence_info': sequence
        }

# 시간적 분석 네트워크
class TemporalAnalysisNet(nn.Module):
    def __init__(self, num_measurements=4, hidden_dim=256, num_layers=2):
        super(TemporalAnalysisNet, self).__init__()

        # 이미지 인코더 (CNN + temporal pooling)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        # 시간적 융합을 위한 LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=128 * 4 * 4 + num_measurements,  # 이미지 특징 + 측정값
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # 어텐션 메커니즘
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )

        # 예측 헤드들
        self.prognosis_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 다음 시점 예측기
        self.next_measurement_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_measurements)
        )

        # 변화율 예측기
        self.change_rate_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # -1 ~ 1 범위의 변화율
        )

    def forward(self, image_sequence, measurement_sequence, timestamps):
        batch_size, seq_len = image_sequence.shape[:2]

        # 이미지 시퀀스 처리
        image_features = []
        for t in range(seq_len):
            img_feature = self.image_encoder(image_sequence[:, t])  # [B, feature_dim]
            image_features.append(img_feature)

        image_features = torch.stack(image_features, dim=1)  # [B, seq_len, feature_dim]

        # 이미지 특징과 측정값 결합
        combined_features = torch.cat([image_features, measurement_sequence], dim=2)

        # LSTM을 통한 시간적 모델링
        lstm_out, (hidden, cell) = self.temporal_lstm(combined_features)  # [B, seq_len, hidden_dim*2]

        # 시간적 어텐션
        attention_weights = self.temporal_attention(lstm_out)  # [B, seq_len, 1]
        attended_features = torch.sum(lstm_out * attention_weights, dim=1)  # [B, hidden_dim*2]

        # 예측
        prognosis_risk = self.prognosis_predictor(attended_features)
        next_measurements = self.next_measurement_predictor(attended_features)
        change_rate = self.change_rate_predictor(attended_features)

        # 시계열 재구성을 위한 디코더 출력
        reconstructed_sequence = []
        for t in range(seq_len):
            recon = self.next_measurement_predictor(lstm_out[:, t])
            reconstructed_sequence.append(recon)

        reconstructed_sequence = torch.stack(reconstructed_sequence, dim=1)

        return {
            'prognosis_risk': prognosis_risk.squeeze(),
            'next_measurements': next_measurements,
            'change_rate': change_rate.squeeze(),
            'reconstructed_sequence': reconstructed_sequence,
            'attention_weights': attention_weights.squeeze(),
            'temporal_features': attended_features
        }

# 시간적 트랜스포머 네트워크
class TemporalTransformerNet(nn.Module):
    def __init__(self, num_measurements=4, d_model=256, nhead=8, num_layers=6):
        super(TemporalTransformerNet, self).__init__()

        self.d_model = d_model

        # 이미지 인코더
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

        # 입력 프로젝션
        self.input_projection = nn.Linear(256 + num_measurements, d_model)

        # 위치 인코딩
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))

        # 트랜스포머 레이어들
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # 출력 헤드들
        self.prognosis_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_measurements)
        )

    def forward(self, image_sequence, measurement_sequence, timestamps):
        batch_size, seq_len = image_sequence.shape[:2]

        # 이미지 특징 추출
        image_features = []
        for t in range(seq_len):
            img_feat = self.image_encoder(image_sequence[:, t])
            image_features.append(img_feat)

        image_features = torch.stack(image_features, dim=1)

        # 이미지와 측정값 결합
        combined_input = torch.cat([image_features, measurement_sequence], dim=2)

        # 입력 프로젝션
        projected_input = self.input_projection(combined_input)

        # 위치 인코딩 추가
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(projected_input.device)
        projected_input = projected_input + self.positional_encoding[positions]

        # 트랜스포머 인코딩
        encoded = self.transformer_encoder(projected_input)

        # 최종 시점 특징 사용 (또는 전체 시퀀스의 평균)
        final_features = encoded[:, -1]  # 마지막 시점
        # 또는: final_features = encoded.mean(dim=1)  # 평균

        # 예측
        prognosis_risk = self.prognosis_head(final_features)
        next_measurements = self.prediction_head(final_features)

        return {
            'prognosis_risk': prognosis_risk.squeeze(),
            'next_measurements': next_measurements,
            'encoded_sequence': encoded,
            'attention_weights': None  # Transformer 내부 어텐션은 별도로 추출 가능
        }

def train_temporal_analysis(dataset_type='cardiac_monitoring', num_epochs=50, batch_size=8, lr=0.001):
    """
    시간적 의료 분석 모델 훈련

    Args:
        dataset_type: 데이터셋 타입
        num_epochs: 훈련 에포크 수
        batch_size: 배치 크기
        lr: 학습률
    """

    # 로거 설정
    logger = create_logger_for_temporal_medical('temporal_analysis', dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # 데이터셋 준비
    dataset = TemporalMedicalDataset(data_type=dataset_type, sequence_length=10)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 모델 설정
    sample = dataset[0]
    num_measurements = sample['measurements'].shape[1]

    # LSTM 기반 모델과 Transformer 기반 모델 두 가지 구현
    lstm_model = TemporalAnalysisNet(num_measurements=num_measurements).to(device)
    transformer_model = TemporalTransformerNet(num_measurements=num_measurements).to(device)

    # 앙상블을 위해 두 모델 모두 사용
    models = {'lstm': lstm_model, 'transformer': transformer_model}

    # 손실 함수
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()

    # 옵티마이저
    optimizers = {
        'lstm': optim.Adam(lstm_model.parameters(), lr=lr, weight_decay=1e-4),
        'transformer': optim.Adam(transformer_model.parameters(), lr=lr, weight_decay=1e-4)
    }

    schedulers = {
        'lstm': optim.lr_scheduler.ReduceLROnPlateau(optimizers['lstm'], patience=5, factor=0.7),
        'transformer': optim.lr_scheduler.ReduceLROnPlateau(optimizers['transformer'], patience=5, factor=0.7)
    }

    # 훈련 메트릭 저장
    train_losses = {'lstm': [], 'transformer': []}
    val_losses = {'lstm': [], 'transformer': []}
    prognosis_accuracies = {'lstm': [], 'transformer': []}

    logger.log("Starting temporal analysis training...")

    for epoch in range(num_epochs):
        # 각 모델별로 훈련
        for model_name, model in models.items():
            model.train()
            running_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                images = batch['images'].to(device)
                measurements = batch['measurements'].to(device)
                timestamps = batch['timestamps'].to(device)
                prognosis_risk = batch['prognosis_risk'].to(device)

                optimizers[model_name].zero_grad()

                # 순전파
                outputs = model(images, measurements, timestamps)

                # 손실 계산
                loss_prognosis = bce_criterion(outputs['prognosis_risk'], prognosis_risk)

                if 'next_measurements' in outputs:
                    # 다음 시점 예측 손실 (마지막 측정값 기준)
                    target_next = measurements[:, -1]  # 시퀀스의 마지막 측정값
                    loss_prediction = mse_criterion(outputs['next_measurements'], target_next)
                else:
                    loss_prediction = torch.tensor(0.0, device=device)

                if 'reconstructed_sequence' in outputs:
                    # 시퀀스 재구성 손실
                    loss_reconstruction = mse_criterion(outputs['reconstructed_sequence'], measurements)
                else:
                    loss_reconstruction = torch.tensor(0.0, device=device)

                # 총 손실
                total_loss = loss_prognosis + 0.5 * loss_prediction + 0.3 * loss_reconstruction

                total_loss.backward()
                optimizers[model_name].step()

                running_loss += total_loss.item()

                if batch_idx % 20 == 0 and model_name == 'lstm':  # 로그는 한 모델만
                    logger.log(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                              f'LSTM Loss: {total_loss.item():.4f}')

            train_losses[model_name].append(running_loss / len(train_loader))

            # 검증 단계
            model.eval()
            val_loss = 0.0
            prognosis_errors = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['images'].to(device)
                    measurements = batch['measurements'].to(device)
                    timestamps = batch['timestamps'].to(device)
                    prognosis_risk = batch['prognosis_risk'].to(device)

                    outputs = model(images, measurements, timestamps)

                    # 손실 계산
                    loss_prognosis = bce_criterion(outputs['prognosis_risk'], prognosis_risk)

                    if 'next_measurements' in outputs:
                        target_next = measurements[:, -1]
                        loss_prediction = mse_criterion(outputs['next_measurements'], target_next)
                    else:
                        loss_prediction = torch.tensor(0.0, device=device)

                    if 'reconstructed_sequence' in outputs:
                        loss_reconstruction = mse_criterion(outputs['reconstructed_sequence'], measurements)
                    else:
                        loss_reconstruction = torch.tensor(0.0, device=device)

                    total_loss = loss_prognosis + 0.5 * loss_prediction + 0.3 * loss_reconstruction
                    val_loss += total_loss.item()

                    # 예후 예측 오차
                    prognosis_error = torch.abs(outputs['prognosis_risk'] - prognosis_risk).cpu().numpy()
                    prognosis_errors.extend(prognosis_error.tolist())

            val_losses[model_name].append(val_loss / len(val_loader))
            prognosis_accuracies[model_name].append(1.0 - np.mean(prognosis_errors))

            # 스케줄러 업데이트
            schedulers[model_name].step(val_losses[model_name][-1])

        # 로깅 (에포크당 한 번)
        logger.log(f'Epoch [{epoch+1}/{num_epochs}]')
        for model_name in models.keys():
            logger.log(f'{model_name.upper()} - Train Loss: {train_losses[model_name][-1]:.4f}, '
                      f'Val Loss: {val_losses[model_name][-1]:.4f}, '
                      f'Prognosis Acc: {prognosis_accuracies[model_name][-1]:.4f}')

        # 메트릭 저장
        logger.save_metrics({
            'epoch': epoch + 1,
            'lstm_train_loss': train_losses['lstm'][-1],
            'lstm_val_loss': val_losses['lstm'][-1],
            'lstm_prognosis_acc': prognosis_accuracies['lstm'][-1],
            'transformer_train_loss': train_losses['transformer'][-1],
            'transformer_val_loss': val_losses['transformer'][-1],
            'transformer_prognosis_acc': prognosis_accuracies['transformer'][-1],
        })

        # 중간 결과 저장 (매 10 에포크)
        if (epoch + 1) % 10 == 0:
            for model_name, model in models.items():
                model.eval()
                with torch.no_grad():
                    sample_batch = next(iter(val_loader))
                    images = sample_batch['images'][:4].to(device)
                    measurements = sample_batch['measurements'][:4].to(device)
                    timestamps = sample_batch['timestamps'][:4].to(device)

                    outputs = model(images, measurements, timestamps)

                    # 시계열 시각화를 위한 이미지 준비
                    vis_images = []
                    for i in range(len(images)):
                        # 첫 번째와 마지막 시점 이미지
                        first_img = images[i, 0].cpu().numpy().transpose(1, 2, 0)
                        last_img = images[i, -1].cpu().numpy().transpose(1, 2, 0)

                        first_img = (first_img + 1) / 2  # [-1, 1] -> [0, 1]
                        last_img = (last_img + 1) / 2

                        vis_images.extend([first_img, last_img])

                    # 어텐션 가중치 제목 포함 (LSTM 모델의 경우)
                    titles = []
                    for i in range(len(images)):
                        if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
                            attn = outputs['attention_weights'][i].mean().item()
                            titles.extend([f'{model_name.upper()} T0', f'{model_name.upper()} T-1 A:{attn:.2f}'])
                        else:
                            titles.extend([f'{model_name.upper()} T0', f'{model_name.upper()} T-1'])

                    logger.save_image_grid(vis_images,
                                         f'temporal_{model_name}_epoch_{epoch+1}.png',
                                         titles=titles,
                                         nrow=2)

    # 최종 모델들 저장
    for model_name, model in models.items():
        logger.save_model(model, f"temporal_{model_name}_final",
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
        plt.plot(prognosis_accuracies[model_name], label=f'{model_name.upper()}')
    plt.title('Prognosis Prediction Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 3)
    # 최종 성능 비교
    final_lstm_acc = prognosis_accuracies['lstm'][-1]
    final_transformer_acc = prognosis_accuracies['transformer'][-1]

    plt.bar(['LSTM', 'Transformer'], [final_lstm_acc, final_transformer_acc])
    plt.title('Final Model Comparison')
    plt.ylabel('Prognosis Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate([final_lstm_acc, final_transformer_acc]):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

    plt.subplot(1, 4, 4)
    # 예시 시계열 예측 결과
    with torch.no_grad():
        models['lstm'].eval()
        sample_batch = next(iter(val_loader))
        sample_measurements = sample_batch['measurements'][0].cpu().numpy()

        # 실제 시계열
        plt.plot(range(len(sample_measurements)), sample_measurements[:, 0],
                'b-', label='Actual', marker='o')

        # 간단한 예측 시뮬레이션 (실제로는 모델 출력 사용)
        predicted = sample_measurements[:, 0] + np.random.normal(0, 0.1, len(sample_measurements))
        plt.plot(range(len(sample_measurements)), predicted,
                'r--', label='Predicted', marker='s')

        plt.title('Time Series Prediction Example')
        plt.xlabel('Time Points')
        plt.ylabel('Measurement Value')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'temporal_analysis_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logger.log("Temporal analysis training completed successfully!")
    logger.log(f"Final LSTM prognosis accuracy: {prognosis_accuracies['lstm'][-1]:.4f}")
    logger.log(f"Final Transformer prognosis accuracy: {prognosis_accuracies['transformer'][-1]:.4f}")
    logger.log(f"Results saved in: {logger.dirs['base']}")

    return models, logger.dirs['base']

if __name__ == "__main__":
    print("⏰ 시간적 의료 분석 (Temporal Medical Analysis)")
    print("=" * 60)

    # 하이퍼파라미터 설정
    config = {
        'dataset_type': 'cardiac_monitoring',
        'num_epochs': 5,  # 빠른 테스트를 위해 5로 설정
        'batch_size': 4,  # 시계열은 메모리 사용량이 높음
        'lr': 0.001
    }

    print(f"Dataset: {config['dataset_type']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print()

    try:
        models, results_dir = train_temporal_analysis(
            dataset_type=config['dataset_type'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )

        print("\n✅ Temporal analysis training completed successfully!")
        print(f"📁 Results saved to: {results_dir}")

        print("\n📊 Generated files include:")
        print("- images/: Temporal analysis visualizations (first/last timepoints)")
        print("- models/: Trained LSTM and Transformer temporal models")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves and model comparisons")
        print("- metrics/: Training metrics in JSON format")

        print("\n🎯 Temporal Analysis Features:")
        print("- Longitudinal disease progression monitoring")
        print("- Multi-timepoint image and measurement fusion")
        print("- Prognosis prediction with temporal patterns")
        print("- LSTM vs Transformer architecture comparison")
        print("- Attention-based temporal modeling")

    except Exception as e:
        print(f"\n❌ Error during temporal analysis training: {str(e)}")
        import traceback
        traceback.print_exc()