#!/usr/bin/env python3
"""
OCT (Optical Coherence Tomography) 의료영상 분석
OCT는 망막의 층별 구조를 마이크로미터 단위로 분석하는 비침습 영상기법입니다.

주요 기능:
- 망막층 분할 (Layer Segmentation)
- 병변 검출 (Lesion Detection)
- 망막 두께 측정 (Thickness Analysis)
- 드루젠 검출 (Drusen Detection)
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
from result_logger import create_logger_for_medical_modalities

# OCT 전용 데이터로더
class OCTDataset(Dataset):
    def __init__(self, data_type='retinal_oct', transform=None):
        """
        OCT 데이터셋 로더

        Args:
            data_type: 'retinal_oct', 'macular_oct', 'optic_disc_oct'
            transform: 이미지 전처리
        """
        self.data_type = data_type
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # 합성 OCT 데이터 생성
        self.images, self.labels = self._generate_synthetic_oct_data()

    def _generate_synthetic_oct_data(self):
        """합성 OCT 데이터 생성"""
        images = []
        labels = []

        for i in range(1000):
            # OCT 특성을 반영한 합성 이미지 생성
            oct_image = self._create_synthetic_oct(i)

            # 병변 라벨 생성 (0: 정상, 1: AMD, 2: DME, 3: CNV)
            label = np.random.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])

            images.append(oct_image)
            labels.append(label)

        return images, labels

    def _create_synthetic_oct(self, seed):
        """OCT 특성을 반영한 합성 이미지 생성"""
        np.random.seed(seed)

        # 기본 망막 구조 생성
        image = np.zeros((512, 512))

        # 망막 층 구조 시뮬레이션
        for layer in range(10):  # 10개 망막층
            y_pos = 50 + layer * 40
            thickness = np.random.uniform(3, 8)

            # 각 층의 강도 패턴
            for x in range(512):
                noise = np.random.normal(0, 0.1)
                intensity = 0.3 + layer * 0.05 + noise

                for dy in range(int(thickness)):
                    if y_pos + dy < 512:
                        image[int(y_pos + dy), x] = intensity

        # 혈관 구조 추가
        for _ in range(5):
            start_x = np.random.randint(0, 512)
            start_y = np.random.randint(100, 400)

            for t in range(100):
                x = start_x + int(t * 0.5 + np.random.normal(0, 2))
                y = start_y + int(np.sin(t * 0.1) * 10 + np.random.normal(0, 1))

                if 0 <= x < 512 and 0 <= y < 512:
                    image[y, x] = 0.8

        # 정규화
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# OCT 분석용 네트워크
class OCTAnalysisNet(nn.Module):
    def __init__(self, num_classes=4):
        super(OCTAnalysisNet, self).__init__()

        # 망막층 특징 추출기
        self.layer_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 256x256
        )

        # 구조적 특징 분석기
        self.structure_analyzer = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 128x128
        )

        # 병변 검출기
        self.lesion_detector = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        # 분류기
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # 회귀기 (두께 측정)
        self.thickness_regressor = nn.Sequential(
            nn.Linear(512 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10개 망막층 두께
        )

    def forward(self, x):
        # 특징 추출
        layer_features = self.layer_extractor(x)
        structure_features = self.structure_analyzer(layer_features)
        lesion_features = self.lesion_detector(structure_features)

        # 평활화
        features = lesion_features.view(lesion_features.size(0), -1)

        # 분류 및 회귀 출력
        classification = self.classifier(features)
        thickness = self.thickness_regressor(features)

        return classification, thickness

# 망막층 분할 네트워크
class RetinalLayerSegmentation(nn.Module):
    def __init__(self, num_layers=10):
        super(RetinalLayerSegmentation, self).__init__()

        # U-Net 기반 아키텍처
        self.encoder1 = self._make_encoder_block(1, 64)
        self.encoder2 = self._make_encoder_block(64, 128)
        self.encoder3 = self._make_encoder_block(128, 256)
        self.encoder4 = self._make_encoder_block(256, 512)

        self.center = self._make_encoder_block(512, 1024)

        self.decoder4 = self._make_decoder_block(1024, 512)
        self.decoder3 = self._make_decoder_block(1024, 256)  # 512 + 512 from skip
        self.decoder2 = self._make_decoder_block(512, 128)   # 256 + 256 from skip
        self.decoder1 = self._make_decoder_block(256, 64)    # 128 + 128 from skip

        self.final = nn.Conv2d(128, num_layers, 1)  # 64 + 64 from skip

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 인코더
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        center = self.center(F.max_pool2d(enc4, 2))

        # 디코더
        dec4 = self.decoder4(F.interpolate(center, scale_factor=2))
        dec4 = torch.cat([dec4, enc4], dim=1)

        dec3 = self.decoder3(F.interpolate(dec4, scale_factor=2))
        dec3 = torch.cat([dec3, enc3], dim=1)

        dec2 = self.decoder2(F.interpolate(dec3, scale_factor=2))
        dec2 = torch.cat([dec2, enc2], dim=1)

        dec1 = self.decoder1(F.interpolate(dec2, scale_factor=2))
        dec1 = torch.cat([dec1, enc1], dim=1)

        return self.final(dec1)

def train_oct_analysis(dataset_type='retinal_oct', num_epochs=50, batch_size=32, lr=0.001):
    """
    OCT 분석 모델 훈련

    Args:
        dataset_type: OCT 데이터셋 타입
        num_epochs: 훈련 에포크 수
        batch_size: 배치 크기
        lr: 학습률
    """

    # 로거 설정
    logger = create_logger_for_medical_modalities('oct_analysis', dataset_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    # 데이터셋 준비
    dataset = OCTDataset(data_type=dataset_type)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 모델 및 손실 함수 설정
    model = OCTAnalysisNet(num_classes=4).to(device)
    segmentation_model = RetinalLayerSegmentation(num_layers=10).to(device)

    criterion_class = nn.CrossEntropyLoss()
    criterion_thickness = nn.MSELoss()
    criterion_seg = nn.CrossEntropyLoss()

    optimizer = optim.Adam(list(model.parameters()) + list(segmentation_model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # 훈련 메트릭 저장
    train_losses = []
    val_losses = []
    val_accuracies = []

    logger.log("Starting OCT analysis training...")

    for epoch in range(num_epochs):
        # 훈련 단계
        model.train()
        segmentation_model.train()
        running_loss = 0.0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            # 분류 및 두께 예측
            class_output, thickness_output = model(data)

            # 분할 예측 (합성 레이블 생성)
            seg_targets = torch.randint(0, 10, (data.size(0), data.size(2), data.size(3))).to(device)
            seg_output = segmentation_model(data)

            # 손실 계산
            loss_class = criterion_class(class_output, targets)

            # 두께 타겟 생성 (정상화된 값)
            thickness_targets = torch.randn(data.size(0), 10).to(device) * 0.1 + 0.3
            loss_thickness = criterion_thickness(thickness_output, thickness_targets)

            loss_seg = criterion_seg(seg_output, seg_targets)

            # 총 손실
            total_loss = loss_class + 0.5 * loss_thickness + 0.3 * loss_seg

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            if batch_idx % 10 == 0:
                logger.log(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], '
                          f'Loss: {total_loss.item():.4f}')

        # 검증 단계
        model.eval()
        segmentation_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)

                class_output, thickness_output = model(data)
                seg_targets = torch.randint(0, 10, (data.size(0), data.size(2), data.size(3))).to(device)
                seg_output = segmentation_model(data)

                loss_class = criterion_class(class_output, targets)
                thickness_targets = torch.randn(data.size(0), 10).to(device) * 0.1 + 0.3
                loss_thickness = criterion_thickness(thickness_output, thickness_targets)
                loss_seg = criterion_seg(seg_output, seg_targets)

                total_loss = loss_class + 0.5 * loss_thickness + 0.3 * loss_seg
                val_loss += total_loss.item()

                _, predicted = torch.max(class_output.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # 메트릭 계산
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        logger.log(f'Epoch [{epoch+1}/{num_epochs}]')
        logger.log(f'Train Loss: {avg_train_loss:.4f}')
        logger.log(f'Val Loss: {avg_val_loss:.4f}')
        logger.log(f'Val Accuracy: {val_accuracy:.2f}%')

        # 학습률 조정
        scheduler.step(avg_val_loss)

        # 메트릭 저장
        logger.save_metrics({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy,
        })

        # 중간 결과 저장 (매 10 에포크)
        if (epoch + 1) % 10 == 0:
            # 샘플 이미지 생성 및 저장
            model.eval()
            segmentation_model.eval()
            with torch.no_grad():
                sample_data, sample_targets = next(iter(val_loader))
                sample_data = sample_data[:8].to(device)

                class_pred, thickness_pred = model(sample_data)
                seg_pred = segmentation_model(sample_data)

                # 분할 결과 시각화
                sample_images = sample_data.cpu().numpy()
                seg_results = torch.argmax(seg_pred, dim=1).cpu().numpy()

                # 이미지 저장을 위한 배열 준비
                visualization_images = []
                for i in range(min(4, len(sample_images))):
                    original = sample_images[i, 0]  # (H, W)
                    segmentation = seg_results[i]   # (H, W)

                    # 정규화
                    original = (original + 1) / 2  # [-1, 1] -> [0, 1]
                    segmentation = segmentation / 9.0  # [0, 9] -> [0, 1]

                    visualization_images.extend([original, segmentation])

                logger.save_image_grid(visualization_images,
                                     f'oct_analysis_epoch_{epoch+1}.png',
                                     titles=['Original', 'Layer Seg'] * 4,
                                     nrow=2)

    # 최종 모델 저장
    logger.save_model(model, "oct_analysis_final",
                     optimizer=optimizer, epoch=num_epochs,
                     config={'dataset_type': dataset_type, 'num_epochs': num_epochs})

    logger.save_model(segmentation_model, "oct_segmentation_final",
                     optimizer=optimizer, epoch=num_epochs,
                     config={'num_layers': 10})

    # 훈련 곡선 저장
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    # 두께 분석 결과 시각화 (마지막 배치)
    thickness_sample = thickness_pred[:4].cpu().numpy()
    layers = ['NFL', 'GCL', 'IPL', 'INL', 'OPL', 'ONL', 'ELM', 'IS/OS', 'RPE', 'Choroid']
    x_pos = np.arange(len(layers))

    for i in range(4):
        plt.plot(x_pos, thickness_sample[i], label=f'Sample {i+1}')

    plt.title('Retinal Layer Thickness Analysis')
    plt.xlabel('Retinal Layers')
    plt.ylabel('Thickness (normalized)')
    plt.xticks(x_pos, layers, rotation=45)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'oct_training_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logger.log("OCT analysis training completed successfully!")
    logger.log(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
    logger.log(f"Results saved in: {logger.dirs['base']}")

    return model, segmentation_model, logger.dirs['base']

if __name__ == "__main__":
    print("🔬 OCT (Optical Coherence Tomography) 의료영상 분석")
    print("=" * 60)

    # 하이퍼파라미터 설정
    config = {
        'dataset_type': 'retinal_oct',
        'num_epochs': 5,  # 빠른 테스트를 위해 5로 설정
        'batch_size': 16,
        'lr': 0.001
    }

    print(f"Dataset: {config['dataset_type']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['lr']}")
    print()

    try:
        model, seg_model, results_dir = train_oct_analysis(
            dataset_type=config['dataset_type'],
            num_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            lr=config['lr']
        )

        print("\n✅ OCT analysis training completed successfully!")
        print(f"📁 Results saved to: {results_dir}")

        print("\n📊 Generated files include:")
        print("- images/: OCT analysis visualizations and layer segmentation")
        print("- models/: Trained OCT analysis and segmentation models")
        print("- logs/: Training logs and configuration")
        print("- plots/: Training curves and thickness analysis")
        print("- metrics/: Training metrics in JSON format")

    except Exception as e:
        print(f"\n❌ Error during OCT analysis training: {str(e)}")
        import traceback
        traceback.print_exc()