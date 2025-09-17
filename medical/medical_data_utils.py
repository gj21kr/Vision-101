"""
Medical Image Data Utilities

Medical image 데이터셋을 로드하고 전처리하는 유틸리티 모음입니다.
다양한 의료 영상 형식 (DICOM, NIfTI, PNG, JPG)을 지원하며,
2D generating 알고리즘과 3D reconstruction 알고리즘에서 사용할 수 있습니다.

지원 데이터셋:
- ChestX-ray14: 흉부 X-ray 이미지
- ISIC: 피부암 이미지
- Brain MRI: 뇌 MRI 이미지
- BraTS: 뇌종양 3D MRI
- Custom DICOM/NIfTI 파일들

사용법:
```python
# 2D 이미지 로드 (Generating 알고리즘용)
loader = MedicalImageLoader('chest_xray')
images = loader.load_2d_dataset(data_path, num_samples=1000)

# 3D 이미지 로드 (3D reconstruction용)
volumes = loader.load_3d_dataset(data_path, num_volumes=50)
```
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Union
import warnings

# Optional medical image libraries
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("Warning: pydicom not available. DICOM support disabled.")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not available. NIfTI support disabled.")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: opencv-python not available. Some preprocessing disabled.")

class MedicalImageLoader:
    def __init__(self, dataset_type='chest_xray', image_size=256):
        """
        Medical Image Data Loader

        Args:
            dataset_type: 데이터셋 종류
                - 'chest_xray': 흉부 X-ray
                - 'brain_mri': 뇌 MRI
                - 'skin_lesion': 피부병변
                - 'ct_scan': CT 스캔
                - 'custom': 사용자 정의
            image_size: 출력 이미지 크기
        """
        self.dataset_type = dataset_type
        self.image_size = image_size

        # Dataset-specific configurations
        self.config = self._get_dataset_config()

    def _get_dataset_config(self):
        """데이터셋별 설정 반환"""
        configs = {
            'chest_xray': {
                'modality': 'X-ray',
                'channels': 1,
                'normalization': 'lung_window',
                'augmentations': ['rotate', 'flip', 'contrast'],
                'file_formats': ['.png', '.jpg', '.jpeg', '.dcm']
            },
            'brain_mri': {
                'modality': 'MRI',
                'channels': 1,
                'normalization': 'brain_window',
                'augmentations': ['rotate', 'flip', 'noise'],
                'file_formats': ['.nii', '.nii.gz', '.dcm', '.png']
            },
            'skin_lesion': {
                'modality': 'Dermoscopy',
                'channels': 3,
                'normalization': 'rgb',
                'augmentations': ['rotate', 'flip', 'color_jitter'],
                'file_formats': ['.png', '.jpg', '.jpeg']
            },
            'ct_scan': {
                'modality': 'CT',
                'channels': 1,
                'normalization': 'ct_window',
                'augmentations': ['rotate', 'flip', 'window_level'],
                'file_formats': ['.nii', '.nii.gz', '.dcm']
            }
        }
        return configs.get(self.dataset_type, configs['chest_xray'])

    def create_synthetic_medical_data(self, num_samples=100, data_type='2d'):
        """
        테스트용 synthetic medical data 생성

        Args:
            num_samples: 생성할 샘플 수
            data_type: '2d' 또는 '3d'

        Returns:
            numpy.ndarray: 생성된 의료 영상 데이터
        """
        print(f"Creating {num_samples} synthetic {self.dataset_type} images...")

        if data_type == '2d':
            return self._create_synthetic_2d(num_samples)
        else:
            return self._create_synthetic_3d(num_samples)

    def _create_synthetic_2d(self, num_samples):
        """2D synthetic medical images 생성"""
        images = []

        for i in range(num_samples):
            if self.dataset_type == 'chest_xray':
                img = self._create_chest_xray()
            elif self.dataset_type == 'brain_mri':
                img = self._create_brain_mri()
            elif self.dataset_type == 'skin_lesion':
                img = self._create_skin_lesion()
            else:
                img = self._create_generic_medical()

            images.append(img)

        return np.array(images)

    def _create_chest_xray(self):
        """Synthetic chest X-ray 생성"""
        size = self.image_size

        # Base lung field (두 개의 타원형)
        img = np.zeros((size, size), dtype=np.float32)

        # Left lung
        y, x = np.ogrid[:size, :size]
        left_lung = ((x - size*0.3)**2 / (size*0.2)**2 +
                     (y - size*0.5)**2 / (size*0.3)**2) < 1
        img[left_lung] = 0.3

        # Right lung
        right_lung = ((x - size*0.7)**2 / (size*0.2)**2 +
                      (y - size*0.5)**2 / (size*0.3)**2) < 1
        img[right_lung] = 0.3

        # Ribs (horizontal lines)
        for i in range(6):
            y_pos = int(size * (0.25 + i * 0.08))
            img[y_pos:y_pos+2, :] += 0.2

        # Heart shadow
        heart = ((x - size*0.45)**2 / (size*0.15)**2 +
                 (y - size*0.6)**2 / (size*0.2)**2) < 1
        img[heart] += 0.4

        # Add noise
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)

        # Convert to uint8 and add channel dimension
        img = (img * 255).astype(np.uint8)
        if self.config['channels'] == 1:
            return img
        else:
            return np.stack([img] * 3, axis=-1)

    def _create_brain_mri(self):
        """Synthetic brain MRI 생성"""
        size = self.image_size

        # Brain outline (circle)
        y, x = np.ogrid[:size, :size]
        center_x, center_y = size//2, size//2
        brain_mask = (x - center_x)**2 + (y - center_y)**2 < (size*0.4)**2

        img = np.zeros((size, size), dtype=np.float32)

        # Brain tissue (different intensities)
        img[brain_mask] = 0.5

        # Gray matter (outer)
        gray_matter = ((x - center_x)**2 + (y - center_y)**2 < (size*0.4)**2) & \
                      ((x - center_x)**2 + (y - center_y)**2 > (size*0.25)**2)
        img[gray_matter] = 0.6

        # White matter (inner)
        white_matter = (x - center_x)**2 + (y - center_y)**2 < (size*0.25)**2
        img[white_matter] = 0.8

        # Ventricles (dark)
        ventricle_left = ((x - center_x + size*0.08)**2 + (y - center_y)**2 < (size*0.05)**2)
        ventricle_right = ((x - center_x - size*0.08)**2 + (y - center_y)**2 < (size*0.05)**2)
        img[ventricle_left | ventricle_right] = 0.1

        # Add some anatomical structures
        # Corpus callosum
        corpus_callosum = ((x - center_x)**2 / (size*0.15)**2 +
                           (y - center_y + size*0.05)**2 / (size*0.02)**2) < 1
        img[corpus_callosum] = 0.9

        # Add noise
        noise = np.random.normal(0, 0.03, img.shape)
        img = np.clip(img + noise, 0, 1)

        img = (img * 255).astype(np.uint8)
        if self.config['channels'] == 1:
            return img
        else:
            return np.stack([img] * 3, axis=-1)

    def _create_skin_lesion(self):
        """Synthetic skin lesion 생성"""
        size = self.image_size

        # Skin background
        skin_color = [220, 180, 150]  # RGB skin tone
        img = np.full((size, size, 3), skin_color, dtype=np.uint8)

        # Lesion (irregular shape)
        y, x = np.ogrid[:size, :size]
        center_x, center_y = size//2, size//2

        # Create irregular lesion shape
        angles = np.linspace(0, 2*np.pi, 100)
        radii = size * 0.15 * (1 + 0.3 * np.sin(5 * angles) + 0.2 * np.random.random(100))

        lesion_mask = np.zeros((size, size), dtype=bool)
        for i, (angle, radius) in enumerate(zip(angles, radii)):
            lx = int(center_x + radius * np.cos(angle))
            ly = int(center_y + radius * np.sin(angle))
            if 0 <= lx < size and 0 <= ly < size:
                # Draw filled circle at each point
                lesion_part = (x - lx)**2 + (y - ly)**2 < (size*0.02)**2
                lesion_mask |= lesion_part

        # Color the lesion
        lesion_color = [139, 69, 19]  # Brown color
        img[lesion_mask] = lesion_color

        # Add some texture
        texture = np.random.normal(0, 10, img.shape).astype(int)
        img = np.clip(img + texture, 0, 255).astype(np.uint8)

        return img

    def _create_generic_medical(self):
        """Generic medical image 생성"""
        size = self.image_size

        # Create structured pattern
        img = np.zeros((size, size), dtype=np.float32)

        # Add some anatomical-like structures
        y, x = np.ogrid[:size, :size]

        # Circular structure
        circle = (x - size//2)**2 + (y - size//2)**2 < (size*0.3)**2
        img[circle] = 0.5

        # Linear structures
        for i in range(3):
            line_y = int(size * (0.3 + i * 0.2))
            img[line_y:line_y+3, :] = 0.7

        # Add noise
        noise = np.random.normal(0, 0.05, img.shape)
        img = np.clip(img + noise, 0, 1)

        img = (img * 255).astype(np.uint8)
        if self.config['channels'] == 1:
            return img
        else:
            return np.stack([img] * 3, axis=-1)

    def _create_synthetic_3d(self, num_volumes):
        """3D synthetic medical volumes 생성"""
        volumes = []
        depth = 64  # Z dimension

        for i in range(num_volumes):
            volume = np.zeros((depth, self.image_size, self.image_size), dtype=np.float32)

            # Create 3D anatomical structure
            z, y, x = np.ogrid[:depth, :self.image_size, :self.image_size]
            center_z, center_y, center_x = depth//2, self.image_size//2, self.image_size//2

            if self.dataset_type == 'brain_mri':
                # Brain-like 3D structure
                brain_mask = ((x - center_x)**2 + (y - center_y)**2 +
                             (z - center_z)**2) < (min(depth, self.image_size) * 0.3)**2
                volume[brain_mask] = 0.5

                # Add layers
                for slice_idx in range(depth):
                    # Add 2D structure to each slice
                    slice_img = self._create_brain_mri()
                    if len(slice_img.shape) == 3:
                        slice_img = slice_img[:, :, 0]
                    volume[slice_idx] = slice_img / 255.0

            elif self.dataset_type == 'ct_scan':
                # CT-like 3D structure
                # Body outline
                body_mask = ((x - center_x)**2 + (y - center_y)**2) < (self.image_size * 0.4)**2
                volume[:, body_mask] = 0.3

                # Organs (simplified)
                organ_mask = ((x - center_x)**2 + (y - center_y)**2) < (self.image_size * 0.2)**2
                volume[:, organ_mask] = 0.6

            else:
                # Generic 3D structure
                sphere_mask = ((x - center_x)**2 + (y - center_y)**2 +
                              (z - center_z)**2) < (min(depth, self.image_size) * 0.25)**2
                volume[sphere_mask] = 0.7

            # Add noise
            noise = np.random.normal(0, 0.02, volume.shape)
            volume = np.clip(volume + noise, 0, 1)

            volumes.append(volume)

        return np.array(volumes)

    def load_real_dataset(self, data_path, max_samples=None):
        """
        실제 의료 이미지 데이터셋 로드

        Args:
            data_path: 데이터 디렉토리 경로
            max_samples: 최대 로드할 샘플 수

        Returns:
            numpy.ndarray: 로드된 이미지들
        """
        if not os.path.exists(data_path):
            print(f"Warning: {data_path} not found. Using synthetic data.")
            return self.create_synthetic_medical_data(max_samples or 100)

        images = []
        file_count = 0

        # Supported file extensions
        extensions = self.config['file_formats']

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    if max_samples and file_count >= max_samples:
                        break

                    file_path = os.path.join(root, file)

                    try:
                        img = self._load_single_file(file_path)
                        if img is not None:
                            images.append(img)
                            file_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to load {file_path}: {e}")

        if not images:
            print("No valid images found. Using synthetic data.")
            return self.create_synthetic_medical_data(max_samples or 100)

        print(f"Loaded {len(images)} medical images from {data_path}")
        return np.array(images)

    def _load_single_file(self, file_path):
        """단일 파일 로드"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == '.dcm' and DICOM_AVAILABLE:
            return self._load_dicom(file_path)
        elif ext in ['.nii', '.gz'] and NIBABEL_AVAILABLE:
            return self._load_nifti(file_path)
        elif ext in ['.png', '.jpg', '.jpeg']:
            return self._load_image(file_path)
        else:
            return None

    def _load_dicom(self, file_path):
        """DICOM 파일 로드"""
        try:
            ds = pydicom.dcmread(file_path)
            img = ds.pixel_array.astype(np.float32)

            # Normalize based on window settings if available
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                center = float(ds.WindowCenter)
                width = float(ds.WindowWidth)
                img = self._apply_window(img, center, width)
            else:
                img = (img - img.min()) / (img.max() - img.min())

            # Resize
            img = self._resize_image(img)

            # Convert to appropriate format
            img = (img * 255).astype(np.uint8)

            if self.config['channels'] == 3 and len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)

            return img

        except Exception as e:
            print(f"Error loading DICOM {file_path}: {e}")
            return None

    def _load_nifti(self, file_path):
        """NIfTI 파일 로드"""
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()

            # Take middle slice if 3D
            if len(data.shape) == 3:
                mid_slice = data.shape[2] // 2
                img = data[:, :, mid_slice]
            else:
                img = data

            # Normalize
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # Resize
            img = self._resize_image(img)

            # Convert to appropriate format
            img = (img * 255).astype(np.uint8)

            if self.config['channels'] == 3 and len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)

            return img

        except Exception as e:
            print(f"Error loading NIfTI {file_path}: {e}")
            return None

    def _load_image(self, file_path):
        """일반 이미지 파일 로드"""
        try:
            img = Image.open(file_path)

            # Convert to appropriate format
            if self.config['channels'] == 1:
                img = img.convert('L')
                img_array = np.array(img)
            else:
                img = img.convert('RGB')
                img_array = np.array(img)

            # Resize
            img_array = self._resize_image(img_array)

            return img_array

        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None

    def _resize_image(self, img):
        """이미지 리사이즈"""
        if CV2_AVAILABLE:
            if len(img.shape) == 2:
                return cv2.resize(img, (self.image_size, self.image_size))
            else:
                return cv2.resize(img, (self.image_size, self.image_size))
        else:
            # Fallback to PIL
            if len(img.shape) == 2:
                pil_img = Image.fromarray(img.astype(np.uint8))
            else:
                pil_img = Image.fromarray(img.astype(np.uint8))
            pil_img = pil_img.resize((self.image_size, self.image_size))
            return np.array(pil_img)

    def _apply_window(self, img, center, width):
        """DICOM window/level 적용"""
        min_val = center - width / 2
        max_val = center + width / 2
        img = np.clip(img, min_val, max_val)
        return (img - min_val) / (max_val - min_val)

    def get_data_statistics(self, images):
        """데이터 통계 정보 반환"""
        stats = {
            'num_images': len(images),
            'image_shape': images[0].shape if len(images) > 0 else None,
            'pixel_mean': np.mean(images) if len(images) > 0 else None,
            'pixel_std': np.std(images) if len(images) > 0 else None,
            'pixel_min': np.min(images) if len(images) > 0 else None,
            'pixel_max': np.max(images) if len(images) > 0 else None
        }
        return stats

    def visualize_samples(self, images, num_samples=9, title="Medical Images"):
        """샘플 이미지들 시각화"""
        if len(images) == 0:
            print("No images to visualize")
            return

        num_samples = min(num_samples, len(images))
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.flatten()

        for i in range(num_samples):
            img = images[i]

            if len(img.shape) == 3 and img.shape[-1] == 3:
                # RGB image
                axes[i].imshow(img)
            else:
                # Grayscale image
                if len(img.shape) == 3:
                    img = img[:, :, 0]
                axes[i].imshow(img, cmap='gray')

            axes[i].set_title(f'{self.dataset_type} Sample {i+1}')
            axes[i].axis('off')

        # Hide unused subplots
        for i in range(num_samples, 9):
            axes[i].axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

# Convenience functions for quick setup
def load_chest_xray_data(data_path=None, num_samples=100, image_size=256):
    """흉부 X-ray 데이터 로드"""
    loader = MedicalImageLoader('chest_xray', image_size)
    if data_path and os.path.exists(data_path):
        return loader.load_real_dataset(data_path, num_samples)
    else:
        return loader.create_synthetic_medical_data(num_samples)

def load_brain_mri_data(data_path=None, num_samples=100, image_size=256):
    """뇌 MRI 데이터 로드"""
    loader = MedicalImageLoader('brain_mri', image_size)
    if data_path and os.path.exists(data_path):
        return loader.load_real_dataset(data_path, num_samples)
    else:
        return loader.create_synthetic_medical_data(num_samples)

def load_skin_lesion_data(data_path=None, num_samples=100, image_size=256):
    """피부 병변 데이터 로드"""
    loader = MedicalImageLoader('skin_lesion', image_size)
    if data_path and os.path.exists(data_path):
        return loader.load_real_dataset(data_path, num_samples)
    else:
        return loader.create_synthetic_medical_data(num_samples)

if __name__ == "__main__":
    print("Medical Image Data Utilities Demo")

    # Test different medical image types
    datasets = ['chest_xray', 'brain_mri', 'skin_lesion']

    for dataset_type in datasets:
        print(f"\n=== {dataset_type.upper()} ===")
        loader = MedicalImageLoader(dataset_type, image_size=256)

        # Create synthetic data
        images = loader.create_synthetic_medical_data(num_samples=10)
        print(f"Created {len(images)} synthetic images")

        # Show statistics
        stats = loader.get_data_statistics(images)
        print(f"Statistics: {stats}")

        # Visualize
        loader.visualize_samples(images, num_samples=6)

    print("\nTo use real medical data:")
    print("1. Download datasets (ChestX-ray14, ISIC, etc.)")
    print("2. Use load_real_dataset() with data path")
    print("3. Or use convenience functions with real data paths")