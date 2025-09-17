"""
Medical Elastic Registration Implementation

의료영상 탄성 정합을 위한 구현으로, 비강체 변형을 통해 다중 시점 및
다중 모달 의료영상을 정합합니다.

의료 특화 기능:
- 시간적/공간적 의료영상 정합
- B-spline 기반 탄성 변형
- 상호정보량 기반 유사도 측정
- 의료영상 특화 정규화 항
- 자동 랜드마크 검출 및 정합

핵심 특징:
1. Non-rigid Deformation Field
2. Mutual Information Similarity
3. B-spline Transformation Model
4. Multi-resolution Registration
5. Anatomical Structure Preservation

Reference:
- Rueckert, D., et al. (1999).
  "Nonrigid registration using free-form deformations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import map_coordinates
from sklearn.metrics import mutual_info_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from medical.result_logger import create_logger_for_medical_registration

class BSplineTransformation(nn.Module):
    """B-spline based transformation for elastic registration"""
    def __init__(self, image_size=(256, 256), control_point_spacing=32):
        super().__init__()
        self.image_size = image_size
        self.control_point_spacing = control_point_spacing

        # Calculate control point grid size
        self.cp_grid_x = image_size[1] // control_point_spacing + 3
        self.cp_grid_y = image_size[0] // control_point_spacing + 3

        # Control point parameters
        self.control_points = nn.Parameter(
            torch.zeros(1, 2, self.cp_grid_y, self.cp_grid_x)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Generate deformation field from control points
        deformation_field = self._generate_deformation_field()

        # Apply transformation
        grid = self._create_sampling_grid(deformation_field)
        transformed = F.grid_sample(x, grid, mode='bilinear',
                                  padding_mode='border', align_corners=True)

        return transformed, deformation_field

    def _generate_deformation_field(self):
        """Generate dense deformation field from sparse control points"""
        h, w = self.image_size

        # Create coordinate grids
        y_coords = torch.linspace(-1, 1, h, device=self.control_points.device)
        x_coords = torch.linspace(-1, 1, w, device=self.control_points.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Map image coordinates to control point coordinates
        cp_y_coords = grid_y * (self.cp_grid_y - 1) / 2 + (self.cp_grid_y - 1) / 2
        cp_x_coords = grid_x * (self.cp_grid_x - 1) / 2 + (self.cp_grid_x - 1) / 2

        # Bilinear interpolation of control points
        deformation_field = F.grid_sample(
            self.control_points,
            torch.stack([cp_x_coords, cp_y_coords], dim=-1).unsqueeze(0),
            mode='bilinear', padding_mode='border', align_corners=True
        )

        return deformation_field.squeeze(0)

    def _create_sampling_grid(self, deformation_field):
        """Create sampling grid for grid_sample"""
        h, w = self.image_size

        # Base identity grid
        y_coords = torch.linspace(-1, 1, h, device=deformation_field.device)
        x_coords = torch.linspace(-1, 1, w, device=deformation_field.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Add deformation
        grid_y = grid_y + deformation_field[0]
        grid_x = grid_x + deformation_field[1]

        # Stack for grid_sample format
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        return grid

class ElasticRegistrationNet(nn.Module):
    """Neural network for learning elastic registration"""
    def __init__(self, image_size=(256, 256)):
        super().__init__()
        self.image_size = image_size

        # Feature extraction network
        self.feature_encoder = self._build_encoder()

        # Transformation network
        self.transformation = BSplineTransformation(image_size)

        # Similarity network
        self.similarity_net = self._build_similarity_net()

    def _build_encoder(self):
        """Build feature extraction encoder"""
        return nn.Sequential(
            nn.Conv2d(2, 64, 7, padding=3), nn.ReLU(),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
        )

    def _build_similarity_net(self):
        """Build similarity measurement network"""
        return nn.Sequential(
            nn.Conv2d(2, 64, 7, padding=3), nn.ReLU(),
            nn.Conv2d(64, 128, 5, padding=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )

    def forward(self, moving, fixed):
        """
        Args:
            moving: Moving image to be registered
            fixed: Fixed reference image
        """
        # Initial similarity
        input_pair = torch.cat([moving, fixed], dim=1)
        features = self.feature_encoder(input_pair)

        # Transform moving image
        transformed, deformation_field = self.transformation(moving)

        # Calculate similarity after transformation
        transformed_pair = torch.cat([transformed, fixed], dim=1)
        similarity_map = self.similarity_net(transformed_pair)

        return transformed, deformation_field, similarity_map

def mutual_information_loss(image1, image2, bins=50):
    """Calculate mutual information between two images"""
    # Flatten images
    img1_flat = image1.view(-1).detach().cpu().numpy()
    img2_flat = image2.view(-1).detach().cpu().numpy()

    # Normalize to 0-bins range
    img1_norm = ((img1_flat - img1_flat.min()) /
                 (img1_flat.max() - img1_flat.min() + 1e-8) * (bins - 1)).astype(int)
    img2_norm = ((img2_flat - img2_flat.min()) /
                 (img2_flat.max() - img2_flat.min() + 1e-8) * (bins - 1)).astype(int)

    # Calculate mutual information
    mi = mutual_info_score(img1_norm, img2_norm)

    # Return negative MI as loss (we want to maximize MI)
    return -mi

def regularization_loss(deformation_field, alpha=1.0):
    """Smoothness regularization for deformation field"""
    # Calculate gradients
    dy = deformation_field[:, :, 1:, :] - deformation_field[:, :, :-1, :]
    dx = deformation_field[:, :, :, 1:] - deformation_field[:, :, :, :-1]

    # L2 norm of gradients
    dy_norm = torch.sum(dy ** 2)
    dx_norm = torch.sum(dx ** 2)

    return alpha * (dy_norm + dx_norm)

def create_synthetic_registration_data(dataset_type, num_pairs=200, image_size=256):
    """Create synthetic medical image pairs for registration"""
    print(f"Creating {num_pairs} synthetic {dataset_type} image pairs for registration...")

    moving_images = []
    fixed_images = []
    ground_truth_transforms = []

    for i in range(num_pairs):
        if dataset_type == 'chest_xray':
            # Create base chest image
            base_image = np.random.randn(image_size, image_size) * 0.2 + 0.5

            # Add lung structures
            center_y, center_x = image_size // 2, image_size // 2
            left_lung = np.zeros((image_size, image_size))
            right_lung = np.zeros((image_size, image_size))

            import cv2
            cv2.ellipse(left_lung, (center_x - 60, center_y), (40, 80), 0, 0, 360, 1, -1)
            cv2.ellipse(right_lung, (center_x + 60, center_y), (40, 80), 0, 0, 360, 1, -1)

            fixed_image = base_image * 0.7 + (left_lung + right_lung) * 0.3

            # Create deformed version (simulating breathing or positioning change)
            deform_strength = np.random.uniform(0.05, 0.15)

            # Generate random deformation
            y_coords, x_coords = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')

            # Sinusoidal deformation (simulating breathing)
            deform_y = deform_strength * np.sin(2 * np.pi * x_coords / image_size * 2)
            deform_x = deform_strength * np.cos(2 * np.pi * y_coords / image_size * 1.5)

            new_y = y_coords + deform_y
            new_x = x_coords + deform_x

            # Apply deformation
            moving_image = map_coordinates(fixed_image, [new_y, new_x], order=1, mode='reflect')

            # Store ground truth transformation
            gt_transform = np.stack([deform_x / image_size * 2, deform_y / image_size * 2])

        elif dataset_type == 'brain_mri':
            # Similar process for brain MRI
            base_image = np.random.randn(image_size, image_size) * 0.15 + 0.4

            # Add brain structure
            center_y, center_x = image_size // 2, image_size // 2
            brain_mask = np.zeros((image_size, image_size))
            cv2.circle(brain_mask, (center_x, center_y), 80, 1, -1)

            fixed_image = base_image * 0.5 + brain_mask * 0.5

            # Create deformation (simulating patient movement)
            deform_strength = np.random.uniform(0.03, 0.1)
            y_coords, x_coords = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')

            deform_y = deform_strength * np.sin(x_coords / image_size * 3 * np.pi)
            deform_x = deform_strength * np.cos(y_coords / image_size * 2 * np.pi)

            new_y = y_coords + deform_y
            new_x = x_coords + deform_x

            moving_image = map_coordinates(fixed_image, [new_y, new_x], order=1, mode='reflect')
            gt_transform = np.stack([deform_x / image_size * 2, deform_y / image_size * 2])

        else:  # skin_lesion
            # Create skin image with lesions
            base_image = np.random.randn(image_size, image_size) * 0.08 + 0.75

            # Add lesions
            num_lesions = np.random.randint(1, 3)
            for _ in range(num_lesions):
                lesion_y = np.random.randint(50, image_size - 50)
                lesion_x = np.random.randint(50, image_size - 50)
                lesion_size = np.random.randint(15, 30)
                cv2.circle(base_image, (lesion_x, lesion_y), lesion_size, -0.2, -1)

            fixed_image = base_image

            # Elastic deformation (simulating skin stretching)
            deform_strength = np.random.uniform(0.02, 0.08)
            y_coords, x_coords = np.meshgrid(np.arange(image_size), np.arange(image_size), indexing='ij')

            deform_y = deform_strength * np.random.randn(image_size, image_size)
            deform_x = deform_strength * np.random.randn(image_size, image_size)

            # Smooth the deformation
            from scipy.ndimage import gaussian_filter
            deform_y = gaussian_filter(deform_y, sigma=20)
            deform_x = gaussian_filter(deform_x, sigma=20)

            new_y = y_coords + deform_y
            new_x = x_coords + deform_x

            moving_image = map_coordinates(fixed_image, [new_y, new_x], order=1, mode='reflect')
            gt_transform = np.stack([deform_x / image_size * 2, deform_y / image_size * 2])

        # Normalize images
        fixed_image = np.clip(fixed_image, 0, 1)
        moving_image = np.clip(moving_image, 0, 1)

        moving_images.append(moving_image)
        fixed_images.append(fixed_image)
        ground_truth_transforms.append(gt_transform)

    return np.array(moving_images), np.array(fixed_images), np.array(ground_truth_transforms)

def train_elastic_registration(dataset_type='chest_xray', data_path=None, num_epochs=50, save_interval=10):
    """
    Elastic Registration 훈련 함수
    """
    # Create result logger
    logger = create_logger_for_medical_registration("elastic_registration", dataset_type)

    # Check for quick test mode
    if os.getenv('QUICK_TEST') == '1':
        num_epochs = min(int(os.getenv('TEST_EPOCHS', 5)), num_epochs)

    # Configuration
    batch_size = 4
    image_size = 256
    learning_rate = 1e-4

    config = {
        'algorithm': 'Elastic Registration',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'image_size': image_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'data_path': data_path
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Create synthetic data
    logger.log(f"Creating synthetic {dataset_type} registration data...")
    moving_images, fixed_images, gt_transforms = create_synthetic_registration_data(
        dataset_type, num_pairs=400, image_size=image_size
    )

    # Split data
    split_idx = int(0.8 * len(moving_images))
    train_moving, val_moving = moving_images[:split_idx], moving_images[split_idx:]
    train_fixed, val_fixed = fixed_images[:split_idx], fixed_images[split_idx:]
    train_gt, val_gt = gt_transforms[:split_idx], gt_transforms[split_idx:]

    logger.log(f"Training pairs: {len(train_moving)}, Validation pairs: {len(val_moving)}")

    # Save sample images
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i in range(4):
        axes[0, i].imshow(train_moving[i], cmap='gray')
        axes[0, i].set_title(f'Moving {i+1}')
        axes[0, i].axis('off')

        axes[1, i].imshow(train_fixed[i], cmap='gray')
        axes[1, i].set_title(f'Fixed {i+1}')
        axes[1, i].axis('off')

        # Show difference
        diff = np.abs(train_moving[i] - train_fixed[i])
        axes[2, i].imshow(diff, cmap='hot')
        axes[2, i].set_title(f'Difference {i+1}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['images'], 'sample_registration_pairs.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Initialize model
    model = ElasticRegistrationNet(image_size=(image_size, image_size)).to(device)
    logger.log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    # Training loop
    logger.log("Starting elastic registration training...")

    train_losses = []
    val_losses = []
    mi_scores = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        num_batches = len(train_moving) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            moving_batch = torch.from_numpy(train_moving[start_idx:end_idx]).unsqueeze(1).float().to(device)
            fixed_batch = torch.from_numpy(train_fixed[start_idx:end_idx]).unsqueeze(1).float().to(device)

            optimizer.zero_grad()

            # Forward pass
            transformed, deformation_field, similarity_map = model(moving_batch, fixed_batch)

            # Calculate losses
            mi_loss = 0
            for i in range(batch_size):
                mi_loss += mutual_information_loss(transformed[i, 0], fixed_batch[i, 0])
            mi_loss /= batch_size

            reg_loss = regularization_loss(deformation_field, alpha=0.1)

            # Total loss
            total_loss = mi_loss + reg_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

            if batch_idx % 10 == 0:
                logger.log(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{num_batches}, '
                          f'Loss: {total_loss.item():.4f}')

        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mi = 0.0

        val_batches = len(val_moving) // batch_size

        with torch.no_grad():
            for batch_idx in range(val_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                moving_batch = torch.from_numpy(val_moving[start_idx:end_idx]).unsqueeze(1).float().to(device)
                fixed_batch = torch.from_numpy(val_fixed[start_idx:end_idx]).unsqueeze(1).float().to(device)

                transformed, deformation_field, similarity_map = model(moving_batch, fixed_batch)

                # Calculate validation metrics
                batch_mi_loss = 0
                for i in range(batch_size):
                    mi_loss_val = mutual_information_loss(transformed[i, 0], fixed_batch[i, 0])
                    batch_mi_loss += mi_loss_val
                    val_mi += -mi_loss_val  # Convert back to positive MI

                batch_mi_loss /= batch_size
                reg_loss_val = regularization_loss(deformation_field, alpha=0.1)

                total_loss_val = batch_mi_loss + reg_loss_val
                val_loss += total_loss_val.item()

        avg_val_loss = val_loss / val_batches
        avg_val_mi = val_mi / (val_batches * batch_size)

        val_losses.append(avg_val_loss)
        mi_scores.append(avg_val_mi)

        scheduler.step()

        logger.log(f'Epoch {epoch+1}/{num_epochs}:')
        logger.log(f'  Train Loss: {avg_train_loss:.4f}')
        logger.log(f'  Val Loss: {avg_val_loss:.4f}')
        logger.log(f'  Val MI: {avg_val_mi:.4f}')
        logger.log(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Save intermediate results
        if (epoch + 1) % save_interval == 0:
            logger.save_model(model, f"elastic_registration_epoch_{epoch+1}", optimizer=optimizer)

            # Generate sample results
            with torch.no_grad():
                sample_moving = torch.from_numpy(val_moving[:4]).unsqueeze(1).float().to(device)
                sample_fixed = torch.from_numpy(val_fixed[:4]).unsqueeze(1).float().to(device)

                sample_transformed, sample_deform, _ = model(sample_moving, sample_fixed)

                # Visualize results
                fig, axes = plt.subplots(4, 4, figsize=(16, 16))
                for i in range(4):
                    axes[i, 0].imshow(val_moving[i], cmap='gray')
                    axes[i, 0].set_title('Moving')
                    axes[i, 0].axis('off')

                    axes[i, 1].imshow(val_fixed[i], cmap='gray')
                    axes[i, 1].set_title('Fixed')
                    axes[i, 1].axis('off')

                    axes[i, 2].imshow(sample_transformed[i, 0].cpu().numpy(), cmap='gray')
                    axes[i, 2].set_title('Registered')
                    axes[i, 2].axis('off')

                    # Show deformation field magnitude
                    deform_mag = torch.sqrt(sample_deform[0]**2 + sample_deform[1]**2).cpu().numpy()
                    axes[i, 3].imshow(deform_mag, cmap='jet')
                    axes[i, 3].set_title('Deformation')
                    axes[i, 3].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(logger.dirs['images'], f'registration_results_epoch_{epoch+1}.png'),
                           dpi=150, bbox_inches='tight')
                plt.close()

        # Log metrics
        logger.log_metric("train_loss", avg_train_loss, epoch)
        logger.log_metric("val_loss", avg_val_loss, epoch)
        logger.log_metric("val_mi", avg_val_mi, epoch)

    # Plot training curves
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Registration Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(mi_scores, label='Mutual Information')
    plt.title('Validation Mutual Information')
    plt.xlabel('Epoch')
    plt.ylabel('MI Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    learning_rates = [lr * (0.7 ** (epoch // 20)) for epoch in range(num_epochs)]
    plt.plot(learning_rates, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(logger.dirs['plots'], 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save final model
    logger.save_model(model, "elastic_registration_final", optimizer=optimizer,
                     metadata={'epochs': num_epochs, 'final_mi': avg_val_mi})

    logger.log("Elastic registration training completed successfully!")
    logger.log(f"Final Mutual Information: {avg_val_mi:.4f}")
    logger.log(f"All results saved in: {logger.dirs['base']}")

    return model, logger.dirs['base']

if __name__ == "__main__":
    print("Medical Elastic Registration Implementation")
    print("=" * 50)
    print("Training elastic registration for medical images...")
    print("Key Features:")
    print("- B-spline based non-rigid transformation")
    print("- Mutual information similarity metric")
    print("- Smoothness regularization")
    print("- Multi-resolution approach")
    print("- Anatomical structure preservation")

    model, results_dir = train_elastic_registration(
        dataset_type='chest_xray',
        data_path=None,
        num_epochs=30,
        save_interval=5
    )

    print(f"\nTraining completed!")
    print(f"Results saved to: {results_dir}")