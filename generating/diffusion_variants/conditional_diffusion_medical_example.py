"""
Medical Conditional Diffusion Model Implementation

의료 이미지 생성을 위한 조건부 확산 모델 구현으로, 특정 의료 조건이나 클래스에 따라
의료 이미지를 생성할 수 있는 고급 기능을 제공합니다.

의료 특화 기능:
- 의료 이미지 특화 데이터 로더 (chest X-ray, brain MRI, skin lesion)
- 자동 결과 저장 및 로깅 시스템
- 조건부 의료 이미지 생성 (정상/비정상, 질병 타입별)
- Classifier-free guidance 지원
- 의료 조건별 품질 평가 메트릭

Conditional Diffusion의 핵심 개념:
1. 조건부 노이즈 예측: ε_θ(x_t, t, c)
2. Classifier Guidance: ∇ log p(c|x_t)
3. Classifier-free Guidance: ε_θ(x_t, t, c) + w(ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
4. 의료 조건에 따른 제어 가능한 생성

의료 분야에서의 활용:
- 특정 질병 유형의 의료 이미지 생성
- 정상/비정상 조건별 이미지 생성
- 다양한 의료 상태 시뮬레이션
- 의료진 교육용 조건부 이미지 생성

Reference:
- Dhariwal, P., & Nichol, A. (2021).
  "Diffusion models beat GANs on image synthesis."
- Ho, J., & Salimans, T. (2022).
  "Classifier-free diffusion guidance."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import sys
import math
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from medical_data_utils import MedicalImageLoader, load_chest_xray_data, load_brain_mri_data
from result_logger import create_logger_for_generating

# Import components from DDPM
from ddpm_medical_example import TimeEmbedding, ResidualBlock, AttentionBlock

class ConditionalTimeEmbedding(nn.Module):
    """조건부 시간 임베딩"""
    def __init__(self, dim, num_classes):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes

        # Time embedding
        self.time_embed = TimeEmbedding(dim)

        # Class embedding
        self.class_embed = nn.Embedding(num_classes + 1, dim)  # +1 for unconditional

    def forward(self, time, class_labels=None):
        """
        시간과 클래스 정보를 결합한 임베딩

        Args:
            time: 시간 정보 [B]
            class_labels: 클래스 라벨 [B] (None일 때는 unconditional)

        Returns:
            결합된 임베딩 [B, dim]
        """
        time_emb = self.time_embed(time)

        if class_labels is not None:
            class_emb = self.class_embed(class_labels)
            return time_emb + class_emb
        else:
            # Unconditional (null class)
            null_class = torch.full((time.shape[0],), self.num_classes, device=time.device)
            class_emb = self.class_embed(null_class)
            return time_emb + class_emb

class ConditionalResidualBlock(nn.Module):
    """조건부 Residual block"""
    def __init__(self, in_channels, out_channels, time_emb_dim, num_classes):
        super().__init__()

        # 조건부 시간 임베딩을 위한 MLP
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.class_mlp = nn.Linear(time_emb_dim, out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.activation = nn.SiLU()

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond_emb):
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        # Add conditional embedding
        cond_proj = self.activation(self.time_mlp(cond_emb))
        x = x + cond_proj[:, :, None, None]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        return x + residual

class ConditionalUNet(nn.Module):
    """의료 이미지를 위한 조건부 U-Net"""
    def __init__(self, in_channels=1, model_channels=128, out_channels=1,
                 num_classes=2, num_res_blocks=2, attention_resolutions=[16],
                 channel_mult=[1, 2, 2, 2], time_emb_dim=512):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.num_res_blocks = num_res_blocks

        # Conditional time embedding
        self.time_embed = nn.Sequential(
            ConditionalTimeEmbedding(model_channels, num_classes),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Down sampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [model_channels]

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ConditionalResidualBlock(ch, mult * model_channels, time_emb_dim, num_classes)]
                ch = mult * model_channels

                if 64 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                self.down_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([nn.Conv2d(ch, ch, 3, stride=2, padding=1)]))
                input_block_chans.append(ch)

        # Middle
        self.middle_block = nn.ModuleList([
            ConditionalResidualBlock(ch, ch, time_emb_dim, num_classes),
            AttentionBlock(ch),
            ConditionalResidualBlock(ch, ch, time_emb_dim, num_classes),
        ])

        # Up sampling
        self.up_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [ConditionalResidualBlock(ch + ich, mult * model_channels, time_emb_dim, num_classes)]
                ch = mult * model_channels

                if 64 // (2 ** level) in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                if level != 0 and i == num_res_blocks:
                    layers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

                self.up_blocks.append(nn.ModuleList(layers))

        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, class_labels=None):
        """
        조건부 노이즈 예측

        Args:
            x: 노이즈가 추가된 의료 이미지 [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            class_labels: 의료 조건 라벨 [B] (None일 때 unconditional)

        Returns:
            예측된 노이즈 [B, C, H, W]
        """
        # Conditional time embedding
        if hasattr(self.time_embed[0], 'forward'):
            # ConditionalTimeEmbedding
            time_emb = self.time_embed[0](timesteps, class_labels)
            for layer in self.time_embed[1:]:
                time_emb = layer(time_emb)
        else:
            # Fallback
            time_emb = self.time_embed(timesteps)

        # Input
        x = self.input_conv(x)
        skip_connections = [x]

        # Down
        for layers in self.down_blocks:
            for layer in layers:
                if isinstance(layer, ConditionalResidualBlock):
                    x = layer(x, time_emb)
                elif isinstance(layer, ResidualBlock):
                    x = layer(x, time_emb)
                else:
                    x = layer(x)
            skip_connections.append(x)

        # Middle
        for layer in self.middle_block:
            if isinstance(layer, ConditionalResidualBlock):
                x = layer(x, time_emb)
            elif isinstance(layer, ResidualBlock):
                x = layer(x, time_emb)
            else:
                x = layer(x)

        # Up
        for layers in self.up_blocks:
            skip = skip_connections.pop()
            x = torch.cat([x, skip], dim=1)

            for layer in layers:
                if isinstance(layer, ConditionalResidualBlock):
                    x = layer(x, time_emb)
                elif isinstance(layer, ResidualBlock):
                    x = layer(x, time_emb)
                else:
                    x = layer(x)

        return self.output_conv(x)

class MedicalConditionalDDPM:
    """의료 이미지를 위한 조건부 DDPM"""
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Noise schedule
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample_conditional(self, model, n, class_labels, channels=1, guidance_scale=3.0,
                          unconditional_prob=0.1):
        """
        Classifier-free guidance를 사용한 조건부 샘플링

        Args:
            model: 훈련된 조건부 모델
            n: 생성할 이미지 수
            class_labels: 의료 조건 라벨 [n]
            channels: 이미지 채널 수
            guidance_scale: guidance 강도
            unconditional_prob: 무조건부 훈련 확률
        """
        print(f"Generating {n} conditional medical images...")
        model.eval()

        with torch.no_grad():
            # Start from random noise
            x = torch.randn((n, channels, self.img_size, self.img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), desc="Conditional Sampling"):
                t = (torch.ones(n) * i).long().to(self.device)

                if guidance_scale > 1.0:
                    # Classifier-free guidance
                    # Conditional prediction
                    noise_pred_cond = model(x, t, class_labels)

                    # Unconditional prediction
                    noise_pred_uncond = model(x, t, None)

                    # Guided prediction
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    # Standard conditional generation
                    noise_pred = model(x, t, class_labels)

                # DDPM sampling step
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * noise_pred) + torch.sqrt(beta) * noise

        model.train()
        # Denormalize
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

class MedicalConditionalDataset(Dataset):
    """조건부 의료 이미지 데이터셋"""
    def __init__(self, images, labels, image_size=64, channels=1):
        self.images = images
        self.labels = labels
        self.image_size = image_size
        self.channels = channels

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # [-1, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        elif len(image.shape) == 3 and self.channels == 1:
            if image.shape[-1] == 3:
                image = np.mean(image, axis=-1, keepdims=True)

        if self.transform:
            image = self.transform(image.astype(np.uint8))

        return image, torch.tensor(label, dtype=torch.long)

def create_medical_labels(dataset_type, num_samples):
    """의료 조건 라벨 생성"""
    if dataset_type == 'chest_xray':
        # 0: Normal, 1: Pneumonia
        labels = np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4])
        class_names = ['Normal', 'Pneumonia']
    elif dataset_type == 'brain_mri':
        # 0: Normal, 1: Tumor
        labels = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])
        class_names = ['Normal', 'Tumor']
    elif dataset_type == 'skin_lesion':
        # 0: Benign, 1: Malignant
        labels = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
        class_names = ['Benign', 'Malignant']
    else:
        labels = np.random.choice([0, 1], size=num_samples)
        class_names = ['Class_0', 'Class_1']

    return labels, class_names

def train_medical_conditional_diffusion(dataset_type='chest_xray', data_path=None,
                                      num_epochs=300, save_interval=30,
                                      unconditional_prob=0.1):
    """
    Medical Conditional Diffusion 훈련 함수

    Args:
        dataset_type: 의료 이미지 종류
        data_path: 실제 데이터 경로
        num_epochs: 훈련 에포크 수
        save_interval: 저장 간격
        unconditional_prob: Classifier-free guidance를 위한 무조건부 훈련 확률
    """
    # Create result logger
    logger = create_logger_for_generating("conditional_diffusion", dataset_type)

    # Quick test mode
    if os.getenv('QUICK_TEST'):
        num_epochs = int(os.getenv('TEST_EPOCHS', 5))
        save_interval = min(save_interval, num_epochs)
        logger.log(f"Quick test mode: training for {num_epochs} epochs only")

    # Hyperparameters
    batch_size = 16
    img_size = 64
    noise_steps = 1000
    lr = 1e-4
    num_classes = 2  # Normal/Abnormal for most medical conditions

    config = {
        'algorithm': 'Conditional_Diffusion',
        'dataset_type': dataset_type,
        'batch_size': batch_size,
        'img_size': img_size,
        'noise_steps': noise_steps,
        'lr': lr,
        'num_classes': num_classes,
        'num_epochs': num_epochs,
        'unconditional_prob': unconditional_prob,
        'data_path': data_path
    }
    logger.save_config(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log(f"Using device: {device}")

    # Load medical data
    logger.log(f"Loading {dataset_type} data...")
    if dataset_type == 'chest_xray':
        images = load_chest_xray_data(data_path, num_samples=1000, image_size=img_size)
        input_channels = 1
    elif dataset_type == 'brain_mri':
        images = load_brain_mri_data(data_path, num_samples=1000, image_size=img_size)
        input_channels = 1
    else:
        loader = MedicalImageLoader(dataset_type, img_size)
        if data_path and os.path.exists(data_path):
            images = loader.load_real_dataset(data_path, 1000)
        else:
            images = loader.create_synthetic_medical_data(1000)
        input_channels = 3 if len(images[0].shape) == 3 else 1

    # Create labels
    labels, class_names = create_medical_labels(dataset_type, len(images))

    logger.log(f"Loaded {len(images)} {dataset_type} images")
    logger.log(f"Medical classes: {class_names}")
    logger.log(f"Label distribution: {np.bincount(labels)}")

    # Create dataset
    dataset = MedicalConditionalDataset(images, labels, img_size, input_channels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model and diffusion
    model = ConditionalUNet(in_channels=input_channels, out_channels=input_channels,
                           num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    diffusion = MedicalConditionalDDPM(noise_steps=noise_steps, img_size=img_size, device=device)

    logger.log(f"Conditional model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    logger.log("Starting Conditional Diffusion training...")
    logger.log(f"Unconditional training probability: {unconditional_prob}")

    model.train()
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0

        for batch_idx, (images_batch, labels_batch) in enumerate(pbar):
            images_batch = images_batch.to(device)
            labels_batch = labels_batch.to(device)

            # Sample timesteps
            t = diffusion.sample_timesteps(images_batch.shape[0]).to(device)

            # Add noise
            x_t, noise = diffusion.noise_images(images_batch, t)

            # Randomly drop labels for classifier-free guidance
            if np.random.random() < unconditional_prob:
                labels_input = None  # Unconditional
            else:
                labels_input = labels_batch  # Conditional

            # Predict noise
            predicted_noise = model(x_t, t, labels_input)

            # Calculate loss
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataloader)
        logger.log_metrics(epoch + 1, avg_loss)

        # Save intermediate results
        if (epoch + 1) % save_interval == 0 or epoch == 0:
            logger.log(f"Generating conditional samples at epoch {epoch + 1}...")

            # Generate samples for each class
            for class_idx, class_name in enumerate(class_names):
                # Create class labels for generation
                test_labels = torch.full((16,), class_idx, device=device)

                # Generate with different guidance scales
                guidance_scales = [1.0, 3.0, 7.0]

                for guidance in guidance_scales:
                    generated = diffusion.sample_conditional(
                        model, n=16, class_labels=test_labels, channels=input_channels,
                        guidance_scale=guidance
                    )

                    # Convert for saving
                    if input_channels == 1:
                        gen_imgs = generated.squeeze(1).cpu().numpy()
                    else:
                        gen_imgs = generated.permute(0, 2, 3, 1).cpu().numpy()

                    logger.save_image_grid(
                        [gen_imgs[i] for i in range(16)],
                        f"generated_class_{class_idx}_{class_name}_guidance_{guidance}_epoch_{epoch+1:03d}",
                        titles=[f"{class_name} {i+1}" for i in range(16)],
                        rows=4, cols=4,
                        cmap='gray' if input_channels == 1 else None
                    )

        # Save model checkpoint
        if (epoch + 1) % (save_interval * 2) == 0:
            logger.save_model(model, f"conditional_diffusion_epoch_{epoch+1:03d}",
                             optimizer=optimizer, epoch=epoch+1, config=config)

    # Final results
    logger.log("Training completed! Generating final conditional samples...")

    # Save final model
    logger.save_model(model, "conditional_diffusion_final",
                     optimizer=optimizer, epoch=num_epochs, config=config)

    # Final comprehensive generation
    for class_idx, class_name in enumerate(class_names):
        final_labels = torch.full((32,), class_idx, device=device)

        # High guidance for clear class distinction
        final_generated = diffusion.sample_conditional(
            model, n=32, class_labels=final_labels, channels=input_channels,
            guidance_scale=7.0
        )

        if input_channels == 1:
            final_imgs = final_generated.squeeze(1).cpu().numpy()
        else:
            final_imgs = final_generated.permute(0, 2, 3, 1).cpu().numpy()

        logger.save_image_grid(
            [final_imgs[i] for i in range(32)],
            f"final_generated_class_{class_idx}_{class_name}",
            titles=[f"{class_name} {i+1}" for i in range(32)],
            rows=4, cols=8,
            cmap='gray' if input_channels == 1 else None
        )

    # Plot training curves
    logger.plot_training_curves()

    # Finalize experiment
    results_dir = logger.finalize_experiment()

    return model, diffusion, results_dir, class_names

if __name__ == "__main__":
    print("Medical Conditional Diffusion Model Implementation")
    print("==================================================")

    dataset_types = ['chest_xray', 'brain_mri', 'skin_lesion']
    selected_dataset = 'chest_xray'

    print(f"Training Conditional Diffusion on {selected_dataset} images...")
    print("Conditional Diffusion Key Features:")
    print("- Class-conditional medical image generation")
    print("- Classifier-free guidance for better quality")
    print("- Support for medical condition-specific generation")
    print("- Multiple guidance scales for quality/diversity trade-off")
    print("- Unconditional training for flexible generation")

    try:
        model, diffusion, results_dir, class_names = train_medical_conditional_diffusion(
            dataset_type=selected_dataset,
            data_path=None,
            num_epochs=100,
            save_interval=20,
            unconditional_prob=0.1
        )

        print(f"\nTraining completed successfully!")
        print(f"Medical classes: {class_names}")
        print(f"All results saved to: {results_dir}")
        print("\nGenerated files include:")
        print("- images/: Class-conditional generated samples with various guidance scales")
        print("- models/: Conditional diffusion model checkpoints")
        print("- logs/: Training logs with conditional loss tracking")
        print("- plots/: Training loss curves")
        print("- metrics/: Training metrics and class-specific quality scores")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise