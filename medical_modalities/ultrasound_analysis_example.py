#!/usr/bin/env python3
"""
초음파(Ultrasound) 의료영상 분석

초음파는 실시간 영상 획득이 가능한 비침습적 진단 도구로,
임신 진단, 심장 기능 평가, 복부 장기 검사에 널리 사용됩니다.

주요 기능:
- 태아 생체 계측 (Fetal Biometry)
- 심장 기능 분석 (Cardiac Function Analysis)
- 혈류 속도 측정 (Doppler Analysis)
- 실시간 장기 추적 (Real-time Organ Tracking)
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
sys.path.append('/workspace/Vision-101')
from medical.result_logger import create_logger_for_medical_modalities

class UltrasoundDataset(Dataset):
    def __init__(self, data_type='cardiac_echo', transform=None):
        """
        초음파 데이터셋 로더

        Args:
            data_type: 'cardiac_echo', 'obstetric', 'abdominal', 'vascular'
            transform: 이미지 전처리
        """
        self.data_type = data_type
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 합성 초음파 데이터 생성
        self.images, self.measurements = self._generate_synthetic_ultrasound_data()

    def _generate_synthetic_ultrasound_data(self):
        """합성 초음파 데이터 생성"""
        images = []
        measurements = []

        for i in range(800):
            ultrasound_image = self._create_synthetic_ultrasound(i)
            measurement = self._create_synthetic_measurements(i)

            images.append(ultrasound_image)
            measurements.append(measurement)

        return images, measurements

    def _create_synthetic_ultrasound(self, seed):
        """초음파 특성을 반영한 합성 이미지 생성"""
        np.random.seed(seed)

        if self.data_type == 'cardiac_echo':
            return self._create_cardiac_echo(seed)
        elif self.data_type == 'obstetric':
            return self._create_obstetric_ultrasound(seed)
        elif self.data_type == 'abdominal':
            return self._create_abdominal_ultrasound(seed)
        else:
            # 기본 초음파 영상
            image = np.random.exponential(0.3, (256, 256))
            image = np.clip(image, 0, 1)
            return (image * 255).astype(np.uint8)

    def _create_cardiac_echo(self, seed):
        """심장 초음파 합성 생성"""
        np.random.seed(seed)
        image = np.zeros((256, 256))

        # 심장 윤곽 생성
        center_x, center_y = 128, 140

        # 좌심실 (타원형)
        a, b = 40, 60  # 장축, 단축
        for y in range(256):
            for x in range(256):
                ellipse_eq = ((x - center_x)**2 / a**2) + ((y - center_y)**2 / b**2)
                if ellipse_eq <= 1:
                    # 심근 벽 두께 시뮬레이션
                    wall_thickness = 8 + np.random.normal(0, 2)
                    if ellipse_eq > (1 - wall_thickness/40)**2:
                        image[y, x] = 0.7 + np.random.normal(0, 0.1)  # 심근
                    else:
                        image[y, x] = 0.3 + np.random.normal(0, 0.05)  # 혈액

        # 승모판 구조
        valve_y = center_y - 20
        for x in range(center_x - 15, center_x + 15):
            image[valve_y:valve_y + 3, x] = 0.8

        # 초음파 특유의 스페클 노이즈
        speckle = np.random.rayleigh(0.05, (256, 256))
        image = np.clip(image + speckle, 0, 1)

        return (image * 255).astype(np.uint8)

    def _create_obstetric_ultrasound(self, seed):
        """산부인과 초음파 합성 생성"""
        np.random.seed(seed)
        image = np.random.exponential(0.2, (256, 256))

        # 태아 머리 윤곽 (원형)
        head_center_x = np.random.randint(80, 176)
        head_center_y = np.random.randint(60, 140)
        head_radius = np.random.randint(30, 50)

        for y in range(256):
            for x in range(256):
                dist = np.sqrt((x - head_center_x)**2 + (y - head_center_y)**2)
                if dist <= head_radius:
                    # 태아 뇌 조직
                    image[y, x] = 0.4 + 0.1 * np.sin(dist * 0.2)

                    # 뇌실 구조
                    if dist <= head_radius * 0.3:
                        image[y, x] = 0.2

        # 척추 구조 (세로선)
        spine_x = np.random.randint(100, 156)
        for y in range(180, 240):
            image[y, spine_x:spine_x+3] = 0.8

        # 양막 경계
        amniotic_boundary = np.random.randint(200, 240)
        image[amniotic_boundary:, :] *= 0.5

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_abdominal_ultrasound(self, seed):
        """복부 초음파 합성 생성"""
        np.random.seed(seed)
        image = np.random.exponential(0.25, (256, 256))

        # 간 경계
        liver_boundary = 80 + np.sin(np.linspace(0, 2*np.pi, 256)) * 20
        for i, y_boundary in enumerate(liver_boundary):
            image[int(y_boundary):int(y_boundary) + 60, i] = 0.5

        # 담낭 (타원형 저에코 영역)
        gb_center_x, gb_center_y = 180, 120
        for y in range(256):
            for x in range(256):
                ellipse_eq = ((x - gb_center_x)**2 / 15**2) + ((y - gb_center_y)**2 / 25**2)
                if ellipse_eq <= 1:
                    image[y, x] = 0.1  # 담즙 (저에코)

        # 혈관 구조
        for i in range(3):
            vessel_x = 50 + i * 70
            for y in range(50, 200):
                vessel_thickness = 2 + np.sin(y * 0.1)
                image[y:y+int(vessel_thickness), vessel_x] = 0.2

        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def _create_synthetic_measurements(self, seed):
        """합성 계측 데이터 생성"""
        np.random.seed(seed)

        if self.data_type == 'cardiac_echo':
            return {
                'lv_ejection_fraction': np.random.uniform(45, 75),  # 좌심실 구혈률
                'lv_end_diastolic_volume': np.random.uniform(90, 140),  # 좌심실 확장말기 용적
                'septal_thickness': np.random.uniform(8, 12),  # 중격 두께
                'wall_motion_score': np.random.uniform(1.0, 2.5)  # 벽운동 점수
            }
        elif self.data_type == 'obstetric':
            gestational_age = np.random.uniform(20, 40)  # 임신 주수
            return {
                'biparietal_diameter': 30 + gestational_age * 2 + np.random.normal(0, 3),  # 양정경
                'head_circumference': 120 + gestational_age * 8 + np.random.normal(0, 10),  # 두위
                'abdominal_circumference': 100 + gestational_age * 9 + np.random.normal(0, 12),  # 복위
                'femur_length': 10 + gestational_age * 1.8 + np.random.normal(0, 2),  # 대퇴골 길이
                'estimated_fetal_weight': 500 + (gestational_age - 20) * 200 + np.random.normal(0, 100)
            }
        else:
            return {
                'organ_volume': np.random.uniform(100, 500),
                'blood_flow_velocity': np.random.uniform(20, 80),
                'resistance_index': np.random.uniform(0.5, 0.9)
            }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        if self.transform:
            image = self.transform(image)

        measurements = self.measurements[idx]
        # 측정값들을 텐서로 변환
        measurement_values = torch.tensor(list(measurements.values()), dtype=torch.float32)

        return image, measurement_values, measurements

# 초음파 특화 CNN 네트워크
class UltrasoundNet(nn.Module):
    def __init__(self, num_measurements=4):
        super(UltrasoundNet, self).__init__()

        # 스페클 노이즈 처리를 위한 초기 레이어
        self.noise_reduction = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )

        # 측정값 예측기
        self.measurement_predictor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_measurements)
        )

        # 품질 평가기
        self.quality_assessor = nn.Sequential(
            nn.Linear(512 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0~1 사이의 품질 점수
        )

    def forward(self, x):
        # 노이즈 감소
        x = self.noise_reduction(x)

        # 특징 추출
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)

        # 측정값 예측
        measurements = self.measurement_predictor(features_flat)

        # 영상 품질 평가
        quality = self.quality_assessor(features_flat)

        return measurements, quality

# 시간적 일관성을 위한 LSTM 레이어
class TemporalUltrasoundNet(nn.Module):
    def __init__(self, num_measurements=4, sequence_length=10):
        super(TemporalUltrasoundNet, self).__init__()

        self.sequence_length = sequence_length

        # 프레임별 특징 추출기 (UltrasoundNet 기반)
        self.frame_encoder = UltrasoundNet(num_measurements)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=512 * 8 * 8,  # UltrasoundNet의 feature size
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # 최종 측정값 예측기
        self.temporal_predictor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_measurements)
        )

        # 움직임 분석기
        self.motion_analyzer = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # x, y 방향 움직임 + 회전
        )

    def forward(self, x_sequence):
        batch_size, seq_len, channels, height, width = x_sequence.shape

        # 각 프레임별로 특징 추출
        frame_features = []
        frame_measurements = []
        frame_qualities = []

        for t in range(seq_len):
            frame = x_sequence[:, t]  # [batch_size, channels, height, width]

            # UltrasoundNet의 feature_extractor만 사용
            noise_reduced = self.frame_encoder.noise_reduction(frame)
            features = self.frame_encoder.feature_extractor(noise_reduced)
            features_flat = features.view(batch_size, -1)

            measurements, quality = self.frame_encoder.measurement_predictor(features_flat), \
                                   self.frame_encoder.quality_assessor(features_flat)

            frame_features.append(features_flat)
            frame_measurements.append(measurements)
            frame_qualities.append(quality)

        # 시퀀스 형태로 변환
        feature_sequence = torch.stack(frame_features, dim=1)  # [batch, seq_len, features]

        # LSTM을 통한 시간적 모델링
        lstm_out, (hidden, cell) = self.lstm(feature_sequence)

        # 마지막 시점의 출력 사용
        final_temporal_features = lstm_out[:, -1]  # [batch, hidden_size]

        # 최종 예측
        temporal_measurements = self.temporal_predictor(final_temporal_features)
        motion_analysis = self.motion_analyzer(final_temporal_features)

        # 개별 프레임 결과들도 반환
        individual_measurements = torch.stack(frame_measurements, dim=1)
        individual_qualities = torch.stack(frame_qualities, dim=1)

        return {
            'temporal_measurements': temporal_measurements,
            'individual_measurements': individual_measurements,
            'individual_qualities': individual_qualities,
            'motion_analysis': motion_analysis
        }

def train_ultrasound_analysis(dataset_type='cardiac_echo', num_epochs=50, batch_size=16, lr=0.001):
    """
    초음파 분석 모델 훈련

    Args:
        dataset_type: 초음파 데이터셋 타입
        num_epochs: 훈련 에포크 수
        batch_size: 배치 크기
        lr: 학습률
    """

    # 로거 설정
    logger = create_logger_for_medical_modalities('ultrasound_analysis', dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # 데이터셋 준비
    dataset = UltrasoundDataset(data_type=dataset_type)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델 및 손실 함수 설정
    num_measurements = len(dataset.measurements[0])
    model = UltrasoundNet(num_measurements=num_measurements).to(device)

    criterion_measurement = nn.MSELoss()
    criterion_quality = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)

    # 훈련 메트릭 저장
    train_losses = []
    val_losses = []
    measurement_accuracies = []

    logger.log("Starting ultrasound analysis training...")

    for epoch in range(num_epochs):
        # 훈련 단계
        model.train()
        running_loss = 0.0

        for batch_idx, (images, measurements, _) in enumerate(train_loader):
            images, measurements = images.to(device), measurements.to(device)

            optimizer.zero_grad()

            # 순전파
            pred_measurements, pred_quality = model(images)

            # 품질 점수 생성 (측정값의 정확도에 기반)
            measurement_error = torch.abs(pred_measurements - measurements).mean(dim=1)
            quality_targets = torch.exp(-measurement_error * 2).to(device)  # 에러가 낮을수록 높은 품질

            # 손실 계산
            loss_measurement = criterion_measurement(pred_measurements, measurements)
            loss_quality = criterion_quality(pred_quality.squeeze(), quality_targets)

            total_loss = loss_measurement + 0.3 * loss_quality

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if batch_idx % 20 == 0:
                logger.log(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                          f'Loss: {total_loss.item():.4f}')

        # 검증 단계
        model.eval()
        val_loss = 0.0
        total_measurement_error = 0.0

        with torch.no_grad():
            for images, measurements, _ in val_loader:
                images, measurements = images.to(device), measurements.to(device)

                pred_measurements, pred_quality = model(images)

                measurement_error = torch.abs(pred_measurements - measurements).mean(dim=1)
                quality_targets = torch.exp(-measurement_error * 2).to(device)

                loss_measurement = criterion_measurement(pred_measurements, measurements)
                loss_quality = criterion_quality(pred_quality.squeeze(), quality_targets)
                total_loss = loss_measurement + 0.3 * loss_quality

                val_loss += total_loss.item()
                total_measurement_error += torch.abs(pred_measurements - measurements).mean().item()

        # 메트릭 계산
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_measurement_accuracy = 1.0 - (total_measurement_error / len(val_loader)) / 100.0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        measurement_accuracies.append(avg_measurement_accuracy)

        logger.log(f'Epoch [{epoch+1}/{num_epochs}]')
        logger.log(f'Train Loss: {avg_train_loss:.4f}')
        logger.log(f'Val Loss: {avg_val_loss:.4f}')
        logger.log(f'Measurement Accuracy: {avg_measurement_accuracy:.4f}')

        # 스케줄러 업데이트
        scheduler.step(avg_val_loss)

        # 메트릭 저장
        logger.save_metrics({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'measurement_accuracy': avg_measurement_accuracy,
        })

        # 중간 결과 저장 (매 10 에포크)
        if (epoch + 1) % 10 == 0:
            # 샘플 예측 및 시각화
            model.eval()
            with torch.no_grad():
                sample_images, sample_measurements, sample_metadata = next(iter(val_loader))
                sample_images = sample_images[:4].to(device)
                sample_measurements = sample_measurements[:4]

                pred_measurements, pred_quality = model(sample_images)

                # 시각화를 위한 이미지 준비
                vis_images = []
                for i in range(len(sample_images)):
                    # 원본 이미지
                    img = sample_images[i].cpu().numpy().squeeze()
                    img = (img + 1) / 2  # [-1, 1] -> [0, 1]

                    vis_images.append(img)

                logger.save_image_grid(vis_images,
                                     f'ultrasound_analysis_epoch_{epoch+1}.png',
                                     titles=[f'Sample {i+1}' for i in range(len(vis_images))],
                                     nrow=2)

    # 최종 모델 저장
    logger.save_model(model, "ultrasound_analysis_final",
                     optimizer=optimizer, epoch=num_epochs,
                     config={'dataset_type': dataset_type, 'num_epochs': num_epochs})

    # 훈련 곡선 저장
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Ultrasound Analysis Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(measurement_accuracies)
    plt.title('Measurement Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    # 마지막 배치의 예측 vs 실제 비교
    if dataset_type == 'cardiac_echo':
        measurement_names = ['EF', 'EDV', 'Septal Thick', 'Wall Motion']
    elif dataset_type == 'obstetric':
        measurement_names = ['BPD', 'HC', 'AC', 'FL']
    else:
        measurement_names = [f'Measure {i+1}' for i in range(num_measurements)]

    actual = sample_measurements[0].numpy()
    predicted = pred_measurements[0].cpu().numpy()

    x_pos = np.arange(len(measurement_names))
    width = 0.35

    plt.bar(x_pos - width/2, actual, width, label='Actual', alpha=0.7)
    plt.bar(x_pos + width/2, predicted, width, label='Predicted', alpha=0.7)

    plt.title('Measurement Comparison (Sample)')
    plt.xlabel('Measurements')
    plt.ylabel('Values')
    plt.xticks(x_pos, measurement_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'ultrasound_analysis_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logger.log("Ultrasound analysis training completed successfully!")
    logger.log(f"Final measurement accuracy: {measurement_accuracies[-1]:.4f}")
    logger.log(f"Results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

if __name__ == "__main__":
    print("🔊 초음파(Ultrasound) 의료영상 분석")
    print("=" * 60)

    # 하이퍼파라미터 설정
    config = {
        'dataset_type': 'cardiac_echo',
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
        model, results_dir = train_ultrasound_analysis(
            dataset_type=config['dataset_type'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )

        print("\n✅ Ultrasound analysis training completed successfully!")
        print(f"📁 Results saved to: {results_dir}")

        print("\n📊 Generated files include:")
        print("- images/: Ultrasound analysis visualizations")
        print("- models/: Trained ultrasound analysis model")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves and measurement comparisons")
        print("- metrics/: Training metrics in JSON format")

    except Exception as e:
        print(f"\n❌ Error during ultrasound analysis training: {str(e)}")
        import traceback
        traceback.print_exc()