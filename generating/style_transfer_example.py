"""
Neural Style Transfer Implementation

Neural Style Transfer는 Gatys et al. (2016)에서 제안된 기법으로, 한 이미지의 콘텐츠와
다른 이미지의 스타일을 결합하여 예술적인 이미지를 생성합니다.

핵심 아이디어:
1. **Content Representation**:
   - 사전 훈련된 CNN (VGG19)의 중간 레이어에서 특징 추출
   - 높은 레벨의 의미적 정보 (what)를 캡처
   - L_content = ||F^l(I) - F^l(C)||²

2. **Style Representation**:
   - 여러 레이어의 특징 맵 간 상관관계 (Gram Matrix) 사용
   - 텍스처와 패턴 정보 (how)를 캡처
   - G^l_ij = Σ_k F^l_ik × F^l_jk (Gram matrix)
   - L_style = Σ_l w_l ||G^l(I) - G^l(S)||²

3. **Total Loss**:
   - L_total = α × L_content + β × L_style
   - α, β로 콘텐츠와 스타일의 균형 조절

구현 방식:
1. **Optimization-based** (이 파일): 이미지를 직접 최적화
   - 고품질 결과, 하지만 느림 (몇 분 소요)

2. **Feed-forward** (Fast Style Transfer): 네트워크를 훈련
   - 실시간 처리 가능, 하지만 특정 스타일에 제한

Gram Matrix의 의미:
- 서로 다른 특징 맵들 간의 상관관계를 측정
- 공간적 정보는 무시하고 텍스처 패턴만 캡처
- 스타일의 본질을 수학적으로 표현하는 방법

응용 분야:
- 예술적 이미지 생성
- 비디오 스타일 변환
- 실시간 카메라 필터
- 브랜드별 이미지 스타일링

Reference:
- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016).
  "Image style transfer using convolutional neural networks."
  Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()

        # Extract features from different layers
        for x in range(2):  # relu1_1
            self.slice1.add_module(str(x), vgg[x])
        for x in range(2, 7):  # relu2_1
            self.slice2.add_module(str(x), vgg[x])
        for x in range(7, 12):  # relu3_1
            self.slice3.add_module(str(x), vgg[x])
        for x in range(12, 21):  # relu4_1
            self.slice4.add_module(str(x), vgg[x])
        for x in range(21, 30):  # relu5_1
            self.slice5.add_module(str(x), vgg[x])

        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        VGG 특징 추출: 여러 레이어의 특징 맵 반환

        Style Transfer에서 중요한 이유:
        - 낮은 레이어: 세부적인 특징 (내용 정보)
        - 높은 레이어: 추상적인 특징 (스타일 정보)
        - 다중 레이어 사용으로 특징의 계층적 표현 가능

        Args:
            x (torch.Tensor): 입력 이미지 [batch_size, 3, H, W]

        Returns:
            list: 5개 레이어의 특징 맵 리스트
                 [relu1_1, relu2_1, relu3_1, relu4_1, relu5_1]

        사용되는 레이어:
        - relu1_1: 기본적인 가장자리, 색상 정보
        - relu2_1: 직선, 곌선 등 단순한 패턴
        - relu3_1: 더 복잡한 패턴과 디테일
        - relu4_1: 고수준 시각적 특징
        - relu5_1: 최고 수준의 의미론적 특징
        """
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

def gram_matrix(features):
    """
    Gram Matrix 계산: 스타일 특징의 핵심 개념

    Gram Matrix의 수학적 의미:
    - G_ij = Σ_k F_ik * F_jk (특징 맵의 내적)
    - 서로 다른 특징 채널 간의 상관관계 측정
    - 공간적 위치 정보를 제거하고 탐스처만 보존

    왜 Gram Matrix를 사용하는가:
    1. 탐스처 정보: 비슷한 탐스처를 가진 특징들이 함께 활성화
    2. 위치 무관성: 어디에 있든 비슷한 탐스처는 비슷한 값
    3. 스케일 불변성: 크기 변화에 상대적으로 덜 민감

    Args:
        features (torch.Tensor): 특징 맵 [batch_size, depth, height, width]

    Returns:
        torch.Tensor: Gram matrix [depth, depth]
                     G[i,j] = 채널 i와 j 간의 상관관계

    계산 과정:
    1. features를 2D로 재구성: [depth, height*width]
    2. 내적 연산: F × F^T
    3. 정규화: 전체 원소 수로 나누어 크기 영향 제거
    """
    batch_size, depth, height, width = features.size()
    features = features.view(batch_size * depth, height * width)
    gram = torch.mm(features, features.t())
    return gram.div(batch_size * depth * height * width)

def content_loss(input_features, target_features):
    """
    Content Loss: 원본 이미지의 내용 보존

    핵심 아이디어:
    - 고수준 레이어에서 특징 맵의 직접적 비교
    - 공간적 구조와 객체의 배치 정보 보존
    - 낮은 래벨 디테일이 아닌 고수준 의미 구조 중점

    Args:
        input_features (torch.Tensor): 현재 생성 중인 이미지의 특징
        target_features (torch.Tensor): 목표 내용 이미지의 특징

    Returns:
        torch.Tensor: MSE 내용 손실

    수학적 정의:
        L_content = (1/2) * ||F^l(P) - F^l(C)||^2
        여기서 P는 생성 중인 이미지, C는 내용 이미지
    """
    return nn.MSELoss()(input_features, target_features)

def style_loss(input_features, target_features):
    """
    Style Loss: 예술 작품의 스타일 학습

    핵심 아이디어:
    - Gram Matrix를 통한 탐스처 상관관계 매칭
    - 공간적 위치와 무관하게 스타일 특성 추출
    - 여러 레이어에서 다양한 스케일의 탐스처 이해

    Args:
        input_features (torch.Tensor): 현재 생성 중인 이미지의 특징
        target_features (torch.Tensor): 목표 스타일 이미지의 특징

    Returns:
        torch.Tensor: Gram matrix 간 MSE 스타일 손실

    수학적 정의:
        L_style = Σ_l w_l * ||G^l(P) - G^l(S)||^2
        여기서 G^l은 l번째 레이어의 Gram matrix
        P는 생성 중인 이미지, S는 스타일 이미지

    왜 효과적인가:
    - 범위의 예술적 스타일에 대해 작동
    - 인간의 예술적 직관과 일치하는 결과
    - 계산적으로 효율적인 그래디언트 연산
    """
    input_gram = gram_matrix(input_features)
    target_gram = gram_matrix(target_features)
    return nn.MSELoss()(input_gram, target_gram)

class NeuralStyleTransfer:
    def __init__(self, content_weight=1e5, style_weight=1e10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg = VGGFeatureExtractor().to(self.device)
        self.content_weight = content_weight
        self.style_weight = style_weight

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        self.denormalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0)
        return image.to(self.device)

    def transfer_style(self, content_path, style_path, num_steps=500):
        # Load images
        content_img = self.load_image(content_path)
        style_img = self.load_image(style_path)

        # Initialize target image (start from content image)
        target_img = content_img.clone().requires_grad_(True)

        # Extract features
        content_features = self.vgg(content_img)
        style_features = self.vgg(style_img)

        # Use content from relu4_1 layer
        content_target = content_features[3]

        # Use style from multiple layers
        style_targets = [style_features[i] for i in range(len(style_features))]

        # Optimizer
        optimizer = optim.LBFGS([target_img], max_iter=20)

        run = [0]
        while run[0] <= num_steps:
            def closure():
                optimizer.zero_grad()

                target_features = self.vgg(target_img)

                # Content loss
                c_loss = content_loss(target_features[3], content_target)

                # Style loss
                s_loss = 0
                for i, target_feature in enumerate(target_features):
                    s_loss += style_loss(target_feature, style_targets[i])

                # Total loss
                total_loss = self.content_weight * c_loss + self.style_weight * s_loss

                total_loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print(f"Step {run[0]}: Content Loss: {c_loss.item():.4f}, Style Loss: {s_loss.item():.4f}")

                return total_loss

            optimizer.step(closure)

        return target_img

    def save_result(self, tensor, output_path):
        # Denormalize and convert to PIL
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = self.denormalize(image)
        image = torch.clamp(image, 0, 1)

        # Convert to PIL and save
        transform_to_pil = transforms.ToPILImage()
        pil_image = transform_to_pil(image)
        pil_image.save(output_path)

class FastStyleTransfer(nn.Module):
    def __init__(self):
        super(FastStyleTransfer, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 9, stride=1, padding=4)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.in3 = nn.InstanceNorm2d(128, affine=True)

        # Residual blocks
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = nn.Conv2d(32, 3, 9, stride=1, padding=4)

    def forward(self, x):
        # Encoder
        x = torch.relu(self.in1(self.conv1(x)))
        x = torch.relu(self.in2(self.conv2(x)))
        x = torch.relu(self.in3(self.conv3(x)))

        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        # Decoder
        x = torch.relu(self.in4(self.deconv1(x)))
        x = torch.relu(self.in5(self.deconv2(x)))
        x = torch.tanh(self.deconv3(x))

        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = torch.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return out + residual

def demo_style_transfer():
    # Create sample images for demonstration
    print("Style Transfer Example")
    print("Two approaches demonstrated:")
    print("1. Neural Style Transfer (Gatys et al.) - Slow but high quality")
    print("2. Fast Style Transfer - Real-time but requires pre-training")

    # Example usage:
    # nst = NeuralStyleTransfer()
    # result = nst.transfer_style('content.jpg', 'style.jpg')
    # nst.save_result(result, 'output.jpg')

    # Fast style transfer example
    fast_model = FastStyleTransfer()
    print(f"Fast Style Transfer model parameters: {sum(p.numel() for p in fast_model.parameters())}")

if __name__ == "__main__":
    demo_style_transfer()