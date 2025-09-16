"""
Result Logger and Saver for Vision-101

모든 알고리즘의 훈련 과정과 결과물을 체계적으로 저장하는 유틸리티입니다.
- 훈련 로그 저장
- 결과 이미지 저장
- 모델 체크포인트 저장
- 메트릭 추적 및 시각화
"""

import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Dict, List, Any, Optional

class ResultLogger:
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "results",
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        결과 로거 초기화

        Args:
            experiment_name: 실험 이름 (예: 'vae_chest_xray', 'nerf_synthetic')
            base_dir: 결과 저장 기본 디렉토리
            category: 실험을 구분하기 위한 카테고리 (예: 'medical', 'non_medical')
            metadata: 로그에 기록할 초기 메타데이터
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(base_dir, f"{experiment_name}_{self.timestamp}")

        # Create directories
        self.dirs = {
            'base': self.experiment_dir,
            'images': os.path.join(self.experiment_dir, 'images'),
            'models': os.path.join(self.experiment_dir, 'models'),
            'logs': os.path.join(self.experiment_dir, 'logs'),
            'metrics': os.path.join(self.experiment_dir, 'metrics'),
            'plots': os.path.join(self.experiment_dir, 'plots')
        }

        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Initialize log file
        self.log_file = os.path.join(self.dirs['logs'], 'training.log')
        self.metrics_file = os.path.join(self.dirs['metrics'], 'metrics.json')
        self.metadata_file = os.path.join(self.dirs['logs'], 'metadata.json')

        # Initialize metrics storage
        self.metrics = {
            'training_losses': [],
            'validation_losses': [],
            'epochs': [],
            'timestamps': [],
            'custom_metrics': {}
        }
        self.metadata: Dict[str, Any] = {}

        self.log(f"Experiment started: {experiment_name}")
        self.log(f"Results will be saved to: {self.experiment_dir}")

        if category:
            self.add_metadata(category=category)
        if metadata:
            self.add_metadata(**metadata)

    def log(self, message: str, level: str = "INFO"):
        """
        로그 메시지 기록

        Args:
            message: 로그 메시지
            level: 로그 레벨 (INFO, WARNING, ERROR)
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] {message}"

        # Print to console
        print(log_message)

        # Write to file
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')

    def add_metadata(self, **metadata: Any):
        """
        추가 메타데이터를 기록합니다.

        Args:
            **metadata: 기록할 키워드 메타데이터
        """
        cleaned_metadata = {k: v for k, v in metadata.items() if v is not None}
        if not cleaned_metadata:
            return

        self.metadata.update(cleaned_metadata)

        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        self.log(f"Metadata updated: {cleaned_metadata}")

    def save_image(self, image_array: np.ndarray, filename: str, title: str = None, cmap: str = 'gray'):
        """
        이미지 배열을 파일로 저장

        Args:
            image_array: 저장할 이미지 배열
            filename: 파일명 (.png가 자동 추가됨)
            title: 이미지 제목
            cmap: 컬러맵 (grayscale 이미지용)
        """
        filepath = os.path.join(self.dirs['images'], f"{filename}.png")

        plt.figure(figsize=(10, 8))

        if len(image_array.shape) == 2:
            plt.imshow(image_array, cmap=cmap)
        elif len(image_array.shape) == 3:
            if image_array.shape[2] == 1:
                plt.imshow(image_array[:, :, 0], cmap=cmap)
            else:
                plt.imshow(image_array)

        if title:
            plt.title(title, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        self.log(f"Image saved: {filepath}")

    def save_image_grid(self, images: List[np.ndarray], filename: str, titles: List[str] = None,
                       rows: int = None, cols: int = None, nrow: int = None, cmap: str = 'gray'):
        """
        여러 이미지를 그리드로 저장

        Args:
            images: 이미지 리스트
            filename: 파일명
            titles: 각 이미지의 제목 리스트
            rows, cols: 그리드 크기 (None이면 자동 계산)
            nrow: 행 개수 (torchvision 호환성을 위해)
            cmap: 컬러맵
        """
        n_images = len(images)

        # Handle nrow parameter for torchvision compatibility
        if nrow is not None:
            cols = nrow
            rows = int(np.ceil(n_images / cols))
        elif rows is None and cols is None:
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        elif rows is None:
            rows = int(np.ceil(n_images / cols))
        elif cols is None:
            cols = int(np.ceil(n_images / rows))

        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))

        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i in range(n_images):
            img = images[i]

            if len(img.shape) == 2:
                axes[i].imshow(img, cmap=cmap)
            elif len(img.shape) == 3:
                if img.shape[2] == 1:
                    axes[i].imshow(img[:, :, 0], cmap=cmap)
                else:
                    axes[i].imshow(img)

            if titles and i < len(titles):
                axes[i].set_title(titles[i], fontsize=10)
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        filepath = os.path.join(self.dirs['images'], f"{filename}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        self.log(f"Image grid saved: {filepath}")

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float = None, **kwargs):
        """
        에포크별 메트릭 기록

        Args:
            epoch: 에포크 번호
            train_loss: 훈련 손실
            val_loss: 검증 손실
            **kwargs: 추가 메트릭들
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['training_losses'].append(train_loss)
        self.metrics['validation_losses'].append(val_loss)
        self.metrics['timestamps'].append(datetime.datetime.now().isoformat())

        # Store custom metrics
        for key, value in kwargs.items():
            if key not in self.metrics['custom_metrics']:
                self.metrics['custom_metrics'][key] = []
            self.metrics['custom_metrics'][key].append(value)

        # Save metrics to file
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)

        # Log the metrics
        log_msg = f"Epoch {epoch}: Train Loss = {train_loss:.6f}"
        if val_loss is not None:
            log_msg += f", Val Loss = {val_loss:.6f}"
        for key, value in kwargs.items():
            log_msg += f", {key} = {value}"
        self.log(log_msg)

    def plot_training_curves(self, save_name: str = "training_curves"):
        """
        훈련 곡선 플롯 및 저장

        Args:
            save_name: 저장 파일명
        """
        if not self.metrics['epochs']:
            self.log("No training metrics to plot", "WARNING")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        # Training and validation loss
        ax1 = axes[0]
        epochs = self.metrics['epochs']
        train_losses = self.metrics['training_losses']

        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)

        if any(loss is not None for loss in self.metrics['validation_losses']):
            val_losses = [loss for loss in self.metrics['validation_losses'] if loss is not None]
            val_epochs = [epoch for epoch, loss in zip(epochs, self.metrics['validation_losses']) if loss is not None]
            ax1.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True)

        # Custom metrics
        custom_metrics = self.metrics['custom_metrics']
        for i, (metric_name, values) in enumerate(custom_metrics.items()):
            if i >= 3:  # Only plot first 3 custom metrics
                break

            ax_idx = i + 1
            if ax_idx < len(axes):
                axes[ax_idx].plot(epochs[:len(values)], values, 'g-', linewidth=2)
                axes[ax_idx].set_xlabel('Epoch')
                axes[ax_idx].set_ylabel(metric_name)
                axes[ax_idx].set_title(f'{metric_name} Over Time')
                axes[ax_idx].grid(True)

        # Hide unused subplots
        for i in range(1 + len(custom_metrics), len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        filepath = os.path.join(self.dirs['plots'], f"{save_name}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        self.log(f"Training curves saved: {filepath}")

    def save_model(self, model, filename: str, optimizer=None, epoch: int = None, **kwargs):
        """
        모델 체크포인트 저장

        Args:
            model: PyTorch 모델
            filename: 파일명
            optimizer: 옵티마이저 (선택사항)
            epoch: 에포크 번호
            **kwargs: 추가 정보
        """
        filepath = os.path.join(self.dirs['models'], f"{filename}.pth")

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        checkpoint.update(kwargs)

        torch.save(checkpoint, filepath)
        self.log(f"Model checkpoint saved: {filepath}")

    def save_config(self, config: Dict[str, Any], filename: str = "config"):
        """
        실험 설정 저장

        Args:
            config: 설정 딕셔너리
            filename: 파일명
        """
        filepath = os.path.join(self.dirs['logs'], f"{filename}.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        self.log(f"Configuration saved: {filepath}")

    def save_numpy_array(self, array: np.ndarray, filename: str, description: str = None):
        """
        NumPy 배열 저장

        Args:
            array: 저장할 배열
            filename: 파일명
            description: 배열 설명
        """
        filepath = os.path.join(self.dirs['base'], f"{filename}.npy")
        np.save(filepath, array)

        if description:
            desc_file = os.path.join(self.dirs['logs'], f"{filename}_description.txt")
            with open(desc_file, 'w', encoding='utf-8') as f:
                f.write(f"Array: {filename}.npy\n")
                f.write(f"Shape: {array.shape}\n")
                f.write(f"Dtype: {array.dtype}\n")
                f.write(f"Description: {description}\n")

        self.log(f"Array saved: {filepath}")

    def save_point_cloud(self, points: np.ndarray, colors: np.ndarray = None, filename: str = "point_cloud"):
        """
        3D point cloud 저장 (PLY 형식)

        Args:
            points: 3D 점들 [N, 3]
            colors: 점 색상들 [N, 3] (선택사항)
            filename: 파일명
        """
        filepath = os.path.join(self.dirs['base'], f"{filename}.ply")

        with open(filepath, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")

            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")

            f.write("end_header\n")

            # Vertex data
            for i, point in enumerate(points):
                if colors is not None:
                    color = colors[i]
                    f.write(f"{point[0]} {point[1]} {point[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
                else:
                    f.write(f"{point[0]} {point[1]} {point[2]}\n")

        self.log(f"Point cloud saved: {filepath}")

    def finalize_experiment(self):
        """
        실험 종료 처리
        """
        # Plot final training curves
        self.plot_training_curves("final_training_curves")

        # Create experiment summary
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': self.timestamp,
            'duration': datetime.datetime.now().isoformat(),
            'total_epochs': len(self.metrics['epochs']),
            'final_train_loss': self.metrics['training_losses'][-1] if self.metrics['training_losses'] else None,
            'best_train_loss': min(self.metrics['training_losses']) if self.metrics['training_losses'] else None,
            'results_directory': self.experiment_dir,
            'category': self.metadata.get('category'),
            'metadata': self.metadata,
        }

        summary_file = os.path.join(self.dirs['base'], 'experiment_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.log("Experiment completed successfully!")
        self.log(f"All results saved in: {self.experiment_dir}")

        return self.experiment_dir

# Specialized subclasses ----------------------------------------------------

class MedicalResultLogger(ResultLogger):
    """Result logger with defaults tailored for medical experiments."""

    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "results/medical",
        specialty: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        combined_metadata = dict(metadata or {})
        if specialty and "specialty" not in combined_metadata:
            combined_metadata["specialty"] = specialty

        super().__init__(
            experiment_name,
            base_dir=base_dir,
            category="medical",
            metadata=combined_metadata,
        )

        self.specialty = specialty

        if specialty:
            self.log(f"Medical specialty set to: {specialty}")


class NonMedicalResultLogger(ResultLogger):
    """Result logger variant for non-medical experiments."""

    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "results/non_medical",
        domain: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        combined_metadata = dict(metadata or {})
        if domain and "domain" not in combined_metadata:
            combined_metadata["domain"] = domain

        super().__init__(
            experiment_name,
            base_dir=base_dir,
            category="non_medical",
            metadata=combined_metadata,
        )

        self.domain = domain

        if domain:
            self.log(f"Non-medical domain set to: {domain}")

# Convenience functions for different algorithm types
def create_logger_for_generating(
    algorithm_name: str,
    dataset_type: str = "",
    medical: bool = True,
    specialty: Optional[str] = None,
    domain: Optional[str] = None,
    base_dir: Optional[str] = None,
):
    """Generating 알고리즘용 로거 생성"""
    exp_name = f"generating_{algorithm_name}"
    if dataset_type:
        exp_name += f"_{dataset_type}"

    experiment_metadata: Dict[str, Any] = {}
    if dataset_type:
        experiment_metadata['dataset_type'] = dataset_type

    if medical:
        target_base_dir = base_dir or "results/medical"
        return MedicalResultLogger(
            exp_name,
            base_dir=target_base_dir,
            specialty=specialty,
            metadata=experiment_metadata or None,
        )

    target_base_dir = base_dir or "results/non_medical"
    return NonMedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        domain=domain,
        metadata=experiment_metadata or None,
    )

def create_logger_for_medical_segmentation(
    algorithm_name: str,
    dataset_type: str,
    specialty: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create a logger for medical segmentation algorithms"""
    exp_name = f"medical_segmentation_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        specialty=specialty,
        metadata=metadata,
    )


def create_logger_for_medical_detection(
    algorithm_name: str,
    dataset_type: str,
    specialty: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create a logger for medical detection algorithms"""
    exp_name = f"medical_detection_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        specialty=specialty,
        metadata=metadata,
    )


def create_logger_for_medical_registration(
    algorithm_name: str,
    dataset_type: str,
    specialty: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create a logger for medical registration algorithms"""
    exp_name = f"medical_registration_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        specialty=specialty,
        metadata=metadata,
    )


def create_logger_for_medical_enhancement(
    algorithm_name: str,
    dataset_type: str,
    specialty: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create a logger for medical enhancement algorithms"""
    exp_name = f"medical_enhancement_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        specialty=specialty,
        metadata=metadata,
    )


def create_logger_for_medical_cad(
    algorithm_name: str,
    dataset_type: str,
    specialty: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create a logger for computer-aided diagnosis algorithms"""
    exp_name = f"medical_cad_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        specialty=specialty,
        metadata=metadata,
    )


def create_logger_for_medical_3d(
    algorithm_name: str,
    dataset_type: str,
    specialty: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create a logger for 3D medical imaging algorithms"""
    exp_name = f"medical_3d_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        specialty=specialty,
        metadata=metadata,
    )


def create_logger_for_specialized_modalities(
    algorithm_name: str,
    dataset_type: str,
    modality: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create a logger for specialized medical modality algorithms"""
    exp_name = f"specialized_modalities_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    combined_metadata = dict(metadata or {})
    if modality and "modality" not in combined_metadata:
        combined_metadata["modality"] = modality
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        metadata=combined_metadata,
    )


def create_logger_for_medical_modalities(
    algorithm_name: str,
    dataset_type: str,
    modality: Optional[str] = None,
    specialty: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create logger for specialized medical modalities"""
    exp_name = f"medical_modalities_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    combined_metadata = dict(metadata or {})
    if modality and "modality" not in combined_metadata:
        combined_metadata["modality"] = modality
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        specialty=specialty,
        metadata=combined_metadata,
    )


def create_logger_for_medical_synthesis(
    algorithm_name: str,
    dataset_type: str,
    specialty: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create a logger for medical image synthesis algorithms"""
    exp_name = f"medical_synthesis_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        specialty=specialty,
        metadata=metadata,
    )


def create_logger_for_clinical_ai(
    algorithm_name: str,
    dataset_type: str,
    application_area: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create logger for clinical AI applications"""
    exp_name = f"clinical_ai_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    combined_metadata = dict(metadata or {})
    if application_area and "application_area" not in combined_metadata:
        combined_metadata["application_area"] = application_area
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        metadata=combined_metadata,
    )


def create_logger_for_multimodal_medical(
    algorithm_name: str,
    dataset_type: str,
    modalities: Optional[List[str]] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create a logger for multi-modal medical AI algorithms"""
    exp_name = f"multimodal_medical_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    combined_metadata = dict(metadata or {})
    if modalities and "modalities" not in combined_metadata:
        combined_metadata["modalities"] = modalities
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        metadata=combined_metadata,
    )


def create_logger_for_temporal_medical(
    algorithm_name: str,
    dataset_type: str,
    analysis_type: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> MedicalResultLogger:
    """Create logger for temporal medical analysis"""
    exp_name = f"temporal_medical_{algorithm_name}_{dataset_type}"
    target_base_dir = base_dir or "results/medical"
    combined_metadata = dict(metadata or {})
    if analysis_type and "analysis_type" not in combined_metadata:
        combined_metadata["analysis_type"] = analysis_type
    return MedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        metadata=combined_metadata,
    )


def create_logger_for_3d(
    algorithm_name: str,
    scene_type: str = "",
    medical: bool = True,
    specialty: Optional[str] = None,
    domain: Optional[str] = None,
    base_dir: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ResultLogger:
    """3D reconstruction 알고리즘용 로거 생성"""
    exp_name = f"3d_{algorithm_name}"
    if scene_type:
        exp_name += f"_{scene_type}"

    combined_metadata = dict(metadata or {})
    if scene_type and "scene_type" not in combined_metadata:
        combined_metadata["scene_type"] = scene_type

    if medical:
        target_base_dir = base_dir or "results/medical"
        return MedicalResultLogger(
            exp_name,
            base_dir=target_base_dir,
            specialty=specialty,
            metadata=combined_metadata,
        )

    target_base_dir = base_dir or "results/non_medical"
    return NonMedicalResultLogger(
        exp_name,
        base_dir=target_base_dir,
        domain=domain,
        metadata=combined_metadata,
    )

if __name__ == "__main__":
    # Example usage
    logger = ResultLogger("test_experiment")

    # Simulate training loop
    for epoch in range(5):
        train_loss = 1.0 - epoch * 0.1
        val_loss = 1.2 - epoch * 0.08
        accuracy = epoch * 0.15 + 0.3

        logger.log_metrics(epoch, train_loss, val_loss, accuracy=accuracy)

    # Create some dummy images
    dummy_images = [np.random.rand(64, 64) for _ in range(9)]
    logger.save_image_grid(dummy_images, "test_grid",
                          titles=[f"Sample {i+1}" for i in range(9)])

    # Finalize
    results_dir = logger.finalize_experiment()
    print(f"Test results saved to: {results_dir}")