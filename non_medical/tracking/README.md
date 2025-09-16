# Computer Vision 객체 추적 알고리즘 (Object Tracking Algorithms)

---

## Korean Version (한글 버전)

### 1. 전통적인 추적 알고리즘 (Classical Tracking Algorithms)
딥러닝이 대중화되기 이전에 주로 사용되었던 방식들로, 지금도 특정 환경에서는 충분히 효과적이며 가볍다는 장점이 있습니다.

#### 1.1. 점/특징점 기반 추적 (Point/Feature-based Tracking)
*   **KLT (Kanade-Lucas-Tomasi) Tracker**: 객체에서 추적하기 좋은 특징점들을 찾아 옵티컬 플로우로 움직임을 추정합니다.
    *   **장점**: 계산이 빠르고, 회전/조명 변화에 강건합니다.
    *   **단점**: 가려짐(occlusion)이나 심한 형태 변화에 취약합니다.

#### 1.2. 커널/템플릿 기반 추적 (Kernel/Template-based Tracking)
*   **Mean-Shift**: 객체의 색상 분포(히스토그램)를 모델링하고, 다음 프레임에서 색상 분포가 가장 유사한 위치로 탐색 윈도우를 이동시킵니다.
    *   **장점**: 구현이 간단하고 비강체(non-rigid) 객체에 효과적입니다.
    *   **단점**: 객체 크기 변화에 대응하지 못하고, 배경과 색상이 비슷하면 실패하기 쉽습니다.
*   **CamShift (Continuously Adaptive Mean-Shift)**: Mean-Shift를 개선하여 탐색 윈도우의 크기와 방향을 동적으로 조절합니다.
    *   **장점**: 객체의 크기 및 회전 변화에 대응할 수 있습니다.
    *   **단점**: 여전히 색상 기반이라 조명 변화에 취약합니다.

#### 1.3. 필터링 기반 추적 (Filtering-based Tracking)
*   **칼만 필터 (Kalman Filter)**: 객체의 움직임을 선형 모델로 가정하고, 매 프레임마다 상태를 예측하고 측정값으로 보정합니다.
    *   **장점**: 예측 가능한 움직임에 매우 정확하고 빠르며, 잡음 제거에 효과적입니다.
    *   **단점**: 비선형적인 움직임에 제대로 대응하지 못합니다.
*   **파티클 필터 (Particle Filter)**: 여러 개의 가설(파티클)을 통해 객체의 확률적 위치를 추정하여 비선형, 비-가우시안 움직임도 추적할 수 있습니다.
    *   **장점**: 복잡한 움직임 추적에 강건합니다.
    *   **단점**: 파티클 수가 많아지면 계산량이 매우 커집니다.

#### 1.4. 상관 필터 기반 추적 (Correlation Filter-based Tracking)
*   **MOSSE**: 주파수 도메인에서 필터링을 수행하여 매우 빠른 속도를 달성한 초기 상관 필터 추적기입니다.
*   **KCF (Kernelized Correlation Filter)**: 커널 트릭을 적용하여 비선형 관계도 학습 가능하게 만들어 정확도를 크게 향상시켰습니다.
*   **DSST**: 위치 필터와 별개로 크기 필터를 추가하여 객체의 크기 변화에 강건한 성능을 보입니다.

### 2. 딥러닝 기반 추적 알고리즘 (Deep Learning-based Tracking)
현재 객체 추적 분야의 SOTA(State-of-the-Art) 성능을 보이는 알고리즘들입니다.

#### 2.1. 단일 객체 추적 (Single Object Tracking, SOT)
하나의 객체를 끝까지 추적하는 데 집중합니다.

*   **샴 네트워크 (Siamese Network) 기반**: 두 개의 동일한 네트워크로 템플릿과 탐색 영역의 유사도를 학습하여 객체를 찾습니다.
    *   **SiamFC**: 가장 기본적인 형태로, 빠르지만 객체 외형 변화에 취약합니다.
    *   **SiamRPN / SiamRPN++**: RPN 구조를 결합하여 바운딩 박스를 더 정확하게 예측하고, 깊은 백본(ResNet 등)을 사용하여 성능을 더욱 향상시켰습니다.
*   **차별적 모델 예측 (Discriminative Model Prediction) 기반**: 온라인에서 객체와 배경을 가장 잘 구별하는 분류기를 효율적으로 학습하는 데 집중합니다.
    *   **ATOM / DiMP**: 매우 효율적인 온라인 학습 전략과 메타러닝을 도입하여 적은 샘플로도 강력한 추적 성능을 보입니다.
*   **트랜스포머 (Transformer) 기반**:
    *   **TransT**: 트랜스포머의 어텐션 메커니즘을 통해 템플릿과 탐색 영역의 특징을 효과적으로 융합하여 정확도를 높입니다.

#### 2.2. 다중 객체 추적 (Multiple Object Tracking, MOT)
여러 객체를 동시에 추적하고, 각 객체에 고유한 ID를 유지합니다.

*   **탐지 기반 추적 (Tracking-by-Detection)**: 매 프레임마다 객체를 탐지(Detection)하고, 이전 프레임의 객체와 연결(Association)하는 방식이 표준입니다.
    *   **SORT**: 칼만 필터와 IoU 기반 연관을 사용하는 간단하고 빠른 MOT의 베이스라인입니다. ID 스위칭이 잦은 단점이 있습니다.
    *   **DeepSORT**: SORT에 외형 특징(Re-ID 모델)을 추가하여 가려짐 상황에서 ID 유지 성능을 크게 향상시켰습니다.
    *   **JDE (Joint Detection and Embedding)**: 탐지와 Re-ID 특징 추출을 하나의 네트워크에서 동시에 수행하여 효율성을 높였습니다.
    *   **ByteTrack**: 신뢰도가 낮은 탐지 결과도 버리지 않고 활용하는 간단한 전략으로, 가려짐 상황에서 매우 높은 성능을 달성했습니다.

### 요약 표

| 구분 | 알고리즘 | 핵심 아이디어 | 장점 | 단점 |
| :--- | :--- | :--- | :--- | :--- |
| **전통적** | KLT | 특징점의 옵티컬 플로우 추적 | 빠름, 회전/조명 변화에 강건 | 가려짐, 형태 변화에 취약 |
| | Mean-Shift | 색상 분포 기반의 중심점 이동 | 구현 간단, 비강체 추적 가능 | 크기 변화 대응 불가, 색상에 민감 |
| | Kalman Filter | 선형 움직임 예측 및 보정 | 예측 가능한 움직임에 빠르고 정확 | 비선형 움직임에 취약 |
| | KCF | 주파수 도메인에서의 상관 필터 | 매우 빠르면서 높은 정확도 | 크기 변화에 직접 대응 어려움 |
| **딥러닝 (SOT)** | SiamRPN++ | 깊은 백본을 사용한 샴 네트워크 | 정확한 박스 예측, 크기/비율 변화 대응 | 상대적으로 느림 |
| | DiMP | 메타러닝으로 온라인 분류기 생성 | 매우 높은 정확도, 안정적 | 상대적으로 복잡함 |
| | TransT | 트랜스포머로 특징 융합 | 복잡한 배경에서 특징 추출 능력 우수 | 계산량이 많음 |
| **딥러닝 (MOT)** | SORT | 칼만 필터 + IoU 연관 | 매우 빠름, 간단함 | ID 스위칭이 잦음 |
| | DeepSORT | SORT + 외형 특징(Re-ID) | 가려짐에 강건, ID 유지 성능 우수 | Re-ID 모델로 인해 느려짐 |
| | ByteTrack | 신뢰도 낮은 탐지 결과도 활용 | 간단하면서 매우 높은 성능, 가려짐에 강함 | 탐지기 성능에 의존적 |

---

## English Version

### 1. Classical Tracking Algorithms
These methods were mainly used before deep learning became popular. They are still effective and lightweight in certain environments.

#### 1.1. Point/Feature-based Tracking
*   **KLT (Kanade-Lucas-Tomasi) Tracker**: Finds good features to track in an object and estimates their movement using optical flow.
    *   **Pros**: Fast computation, robust to rotation/illumination changes.
    *   **Cons**: Vulnerable to occlusion or significant shape changes.

#### 1.2. Kernel/Template-based Tracking
*   **Mean-Shift**: Models the object's color distribution (histogram) and moves the search window to the most similar location in the next frame.
    *   **Pros**: Simple implementation, effective for non-rigid objects.
    *   **Cons**: Cannot handle scale changes, fails with similar background colors.
*   **CamShift (Continuously Adaptive Mean-Shift)**: An improved Mean-Shift that dynamically adjusts the search window's size and orientation.
    *   **Pros**: Can handle changes in object size and rotation.
    *   **Cons**: Still color-based and vulnerable to illumination changes.

#### 1.3. Filtering-based Tracking
*   **Kalman Filter**: Assumes a linear motion model for the object, predicting and updating its state in each frame.
    *   **Pros**: Very accurate and fast for predictable motion, effective for noise removal.
    *   **Cons**: Does not handle non-linear motion well.
*   **Particle Filter**: Estimates the probabilistic location of an object using multiple hypotheses (particles), allowing it to track non-linear, non-Gaussian motion.
    *   **Pros**: Robust for tracking complex movements.
    *   **Cons**: Computationally expensive as the number of particles increases.

#### 1.4. Correlation Filter-based Tracking
*   **MOSSE**: An early correlation filter tracker that achieved very high speed by performing filtering in the frequency domain.
*   **KCF (Kernelized Correlation Filter)**: Greatly improved accuracy by applying the kernel trick, enabling it to learn non-linear relationships.
*   **DSST**: Shows robust performance against scale changes by adding a separate scale filter in addition to the position filter.

### 2. Deep Learning-based Tracking Algorithms
These are the state-of-the-art (SOTA) algorithms in the field of object tracking.

#### 2.1. Single Object Tracking (SOT)
Focuses on tracking a single object to the end.

*   **Siamese Network-based**: Uses two identical networks to learn the similarity between a template and a search region to find the object.
    *   **SiamFC**: The most basic form; fast but vulnerable to appearance changes.
    *   **SiamRPN / SiamRPN++**: Incorporates an RPN structure for more accurate bounding box prediction and uses deep backbones (e.g., ResNet) to further improve performance.
*   **Discriminative Model Prediction-based**: Focuses on efficiently learning a classifier online that best distinguishes the object from the background.
    *   **ATOM / DiMP**: Achieve powerful tracking performance with few samples by introducing highly efficient online learning strategies and meta-learning.
*   **Transformer-based**:
    *   **TransT**: Effectively fuses features from the template and search region using the Transformer's attention mechanism to improve accuracy.

#### 2.2. Multiple Object Tracking (MOT)
Simultaneously tracks multiple objects and maintains a unique ID for each.

*   **Tracking-by-Detection**: The standard paradigm, which involves detecting objects in each frame and then associating them with objects from the previous frame.
    *   **SORT**: A simple and fast MOT baseline using a Kalman filter and IoU-based association. Prone to frequent ID switches.
    *   **DeepSORT**: Greatly improves ID retention during occlusion by adding an appearance feature (Re-ID model) to SORT.
    *   **JDE (Joint Detection and Embedding)**: Increases efficiency by performing detection and Re-ID feature extraction simultaneously in a single network.
    *   **ByteTrack**: Achieves very high performance in occlusion scenarios with a simple strategy of utilizing low-confidence detection results instead of discarding them.

### Summary Table

| Category | Algorithm | Core Idea | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Classical** | KLT | Optical flow of feature points | Fast, robust to rotation/illumination | Vulnerable to occlusion, shape change |
| | Mean-Shift | Center-shift based on color distribution | Simple, good for non-rigid objects | No scale handling, sensitive to color |
| | Kalman Filter | Linear motion prediction and correction | Fast & accurate for predictable motion | Weak with non-linear motion |
| | KCF | Correlation filter in frequency domain | Very fast with high accuracy | Hard to handle scale changes directly |
| **Deep (SOT)** | SiamRPN++ | Siamese network with deep backbone | Accurate box prediction, handles scale/ratio | Relatively slow |
| | DiMP | Meta-learning for online classifier | Very high accuracy, stable | Relatively complex |
| | TransT | Feature fusion with Transformer | Excellent feature extraction in clutter | Computationally intensive |
| **Deep (MOT)** | SORT | Kalman Filter + IoU association | Very fast, simple | Frequent ID switches |
| | DeepSORT | SORT + Appearance Feature (Re-ID) | Robust to occlusion, good ID retention | Slower due to Re-ID model |
| | ByteTrack | Utilizes low-score detections | Simple yet very high performance, robust to occlusion | Dependent on detector performance |

