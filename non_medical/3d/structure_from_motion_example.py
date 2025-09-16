"""
Structure from Motion (SfM) Implementation

Structure from Motion은 여러 장의 2D 이미지로부터 카메라의 위치와 3D 장면 구조를
동시에 복원하는 컴퓨터 비전의 고전적이면서도 핵심적인 기술입니다.

핵심 개념:

1. **Feature Detection & Matching**:
   - SIFT, ORB, SuperPoint 등으로 특징점 추출
   - 이미지 간 feature correspondence 찾기
   - RANSAC으로 outlier 제거

2. **Camera Pose Estimation**:
   - Essential/Fundamental Matrix 추정
   - Two-view geometry로 초기 카메라 포즈 계산
   - PnP (Perspective-n-Point)로 추가 뷰 등록

3. **Triangulation**:
   - 2개 이상 뷰에서 대응되는 점들로부터 3D 점 계산
   - Linear/Non-linear triangulation methods
   - Reprojection error minimization

4. **Bundle Adjustment**:
   - 전역 최적화로 카메라 파라미터와 3D 점들 정제
   - Non-linear least squares optimization
   - Sparse matrix techniques for efficiency

수학적 원리:

Camera Projection:
x = K[R|t]X, 여기서 x: 2D point, X: 3D point, K: intrinsic, [R|t]: extrinsic

Triangulation (DLT - Direct Linear Transform):
AX = 0, where A는 cross product로 구성된 행렬

Bundle Adjustment:
min Σᵢⱼ ||xᵢⱼ - π(Kᵢ, Rᵢ, tᵢ, Xⱼ)||²

Epipolar Geometry:
x₂ᵀFx₁ = 0 (Fundamental Matrix)
x₂ᵀEx₁ = 0 (Essential Matrix, calibrated case)

장점:
- 카메라 보정 없이도 3D 복원 가능
- 넓은 시야각과 복잡한 장면 처리 가능
- 수십 년간 검증된 안정적 기술
- COLMAP, OpenMVG 등 성숙한 구현 존재

단점:
- 텍스처가 부족한 영역에서 어려움
- 반복적 패턴에서 false matching
- 긴 처리 시간 (특히 large-scale)
- 카메라 간 baseline이 너무 작거나 클 때 불안정

Reference:
- Hartley, R. & Zisserman, A. (2003).
  "Multiple View Geometry in Computer Vision."
- Schönberger, J. L. & Frahm, J. M. (2016).
  "Structure-from-Motion Revisited."
  Conference on Computer Vision and Pattern Recognition (CVPR).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import scipy.optimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import warnings

class FeatureExtractor:
    def __init__(self, method='sift', max_features=2000):
        """
        Feature Extractor: 이미지에서 keypoint와 descriptor 추출

        지원하는 방법들:
        - SIFT: Scale-Invariant Feature Transform (가장 안정적)
        - ORB: Oriented FAST and Rotated BRIEF (빠름)
        - SURF: Speeded Up Robust Features (특허 문제)

        Args:
            method: 특징점 추출 방법
            max_features: 최대 특징점 수
        """
        self.method = method
        self.max_features = max_features

        if method == 'sift':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif method == 'orb':
            self.detector = cv2.ORB_create(nfeatures=max_features)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def extract_features(self, image):
        """
        단일 이미지에서 특징점 추출

        Args:
            image: 입력 이미지 (numpy array or torch tensor)

        Returns:
            tuple: (keypoints, descriptors)
                - keypoints: cv2.KeyPoint 리스트
                - descriptors: 특징 벡터 [n_features, descriptor_dim]

        SIFT 특징:
        - Scale invariant: 크기 변화에 강인
        - Rotation invariant: 회전에 강인
        - Partial illumination invariant: 조명 변화에 어느 정도 강인
        - 128차원 descriptor
        """
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:  # CHW -> HWC
                image = image.permute(1, 2, 0).cpu().numpy()
            else:
                image = image.cpu().numpy()

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Extract keypoints and descriptors
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)

        return keypoints, descriptors

    def match_features(self, desc1, desc2, ratio_threshold=0.7):
        """
        두 이미지 간 특징점 매칭

        Lowe's Ratio Test:
        - 첫 번째 nearest neighbor와 두 번째 nearest neighbor의 거리 비율 확인
        - ratio < threshold인 경우만 good match로 판정
        - False positive 크게 감소

        Args:
            desc1, desc2: 특징 벡터들
            ratio_threshold: Lowe's ratio test 임계값

        Returns:
            list: Good matches (cv2.DMatch 리스트)

        매칭 알고리즘:
        1. Brute Force: 모든 조합 계산 (정확하지만 느림)
        2. FLANN: Fast approximation (빠르지만 근사적)
        3. Ratio test로 품질 필터링
        """
        if desc1 is None or desc2 is None:
            return []

        # Create matcher
        if self.method == 'sift':
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:  # ORB
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Find 2 nearest neighbors
        matches = matcher.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

class PoseEstimator:
    def __init__(self, camera_matrix=None):
        """
        Pose Estimator: 카메라 포즈 추정

        Two-view geometry를 이용한 relative pose estimation:
        1. Essential Matrix 추정 (calibrated case)
        2. Fundamental Matrix 추정 (uncalibrated case)
        3. Pose decomposition
        4. Triangulation으로 validation

        Args:
            camera_matrix: 카메라 내부 파라미터 K [3, 3]
        """
        self.camera_matrix = camera_matrix
        self.is_calibrated = camera_matrix is not None

    def estimate_pose_two_view(self, pts1, pts2, method='essential'):
        """
        Two-view에서 relative pose 추정

        Essential Matrix method (calibrated):
        - E = [t]×R (cross product of translation and rotation)
        - x₂ᵀEx₁ = 0 (epipolar constraint)
        - 5-point algorithm으로 해결

        Fundamental Matrix method (uncalibrated):
        - F = K₂⁻ᵀEK₁⁻¹
        - x₂ᵀFx₁ = 0
        - 8-point algorithm으로 해결

        Args:
            pts1, pts2: 대응점들 [N, 2]
            method: 'essential' or 'fundamental'

        Returns:
            dict: {'R': rotation, 't': translation, 'inliers': mask}

        중요한 점:
        - RANSAC으로 outlier 제거 필수
        - Cheirality check로 올바른 해 선택
        - Triangulation으로 결과 검증
        """
        pts1 = np.array(pts1, dtype=np.float32)
        pts2 = np.array(pts2, dtype=np.float32)

        if method == 'essential' and self.is_calibrated:
            # Essential Matrix 추정
            E, mask = cv2.findEssentialMat(
                pts1, pts2,
                self.camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )

            # Pose recovery
            _, R, t, mask = cv2.recoverPose(
                E, pts1, pts2, self.camera_matrix, mask=mask
            )

        else:
            # Fundamental Matrix 추정 (uncalibrated case)
            F, mask = cv2.findFundamentalMat(
                pts1, pts2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99
            )

            # Essential Matrix로 변환 (if calibrated)
            if self.is_calibrated:
                E = self.camera_matrix.T @ F @ self.camera_matrix
                _, R, t, mask = cv2.recoverPose(
                    E, pts1, pts2, self.camera_matrix, mask=mask
                )
            else:
                R, t = None, None  # 추가 정보 없이는 metric reconstruction 불가

        return {
            'R': R,
            't': t,
            'inliers': mask.ravel().astype(bool) if mask is not None else None,
            'essential_matrix': E if 'E' in locals() else None,
            'fundamental_matrix': F if 'F' in locals() else None
        }

    def triangulate_points(self, pts1, pts2, P1, P2):
        """
        Triangulation: 대응점들로부터 3D 점 복원

        DLT (Direct Linear Transform):
        - x₁ × (P₁X) = 0
        - x₂ × (P₂X) = 0
        - 이를 AX = 0 형태의 homogeneous linear system으로 변환

        Args:
            pts1, pts2: 대응점들 [N, 2]
            P1, P2: 투영 행렬들 [3, 4]

        Returns:
            numpy.ndarray: 3D 점들 [N, 3]

        수학적 배경:
        Cross product constraint:
        [x₁]× P₁X = 0
        [y₁]
        [1 ]

        이를 전개하면 2개의 선형 방정식:
        x₁(P₁₃ᵀX) - (P₁₁ᵀX) = 0
        y₁(P₁₃ᵀX) - (P₁₂ᵀX) = 0
        """
        pts1 = pts1.reshape(-1, 2)
        pts2 = pts2.reshape(-1, 2)

        # OpenCV triangulation
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Convert from homogeneous to 3D
        points_3d = points_4d[:3] / points_4d[3:]
        points_3d = points_3d.T

        return points_3d

    def pnp_pose_estimation(self, points_3d, points_2d, camera_matrix):
        """
        PnP (Perspective-n-Point): 3D-2D 대응으로부터 카메라 포즈 추정

        문제 정의:
        - 3D 점들의 world coordinates 알고 있음
        - 해당 점들의 2D image coordinates 관찰
        - 카메라의 position & orientation 구하기

        Args:
            points_3d: 3D 점들 [N, 3]
            points_2d: 대응하는 2D 점들 [N, 2]
            camera_matrix: 카메라 내부 파라미터 [3, 3]

        Returns:
            dict: {'R': rotation_vector, 't': translation, 'inliers': mask}

        PnP variants:
        - P3P: 3점 (minimal case, multiple solutions)
        - EPnP: Efficient PnP (4+ points)
        - UPnP: Unified PnP
        - RANSAC-PnP: outlier rejection
        """
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)

        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            camera_matrix,
            None,  # No distortion
            reprojectionError=8.0,
            confidence=0.99
        )

        if success:
            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)

            return {
                'R': R,
                't': tvec,
                'rvec': rvec,
                'tvec': tvec,
                'inliers': inliers.ravel() if inliers is not None else None,
                'success': True
            }
        else:
            return {'success': False}

class BundleAdjustment:
    def __init__(self, camera_matrices, poses, points_3d, observations):
        """
        Bundle Adjustment: 전역 최적화를 통한 카메라 파라미터와 3D 점 정제

        목표:
        - 모든 관찰 데이터에 대해 reprojection error 최소화
        - 카메라 포즈와 3D 점을 동시에 최적화
        - Sparse structure 활용으로 효율적 계산

        Args:
            camera_matrices: 카메라 내부 파라미터들
            poses: 카메라 포즈들 (R, t)
            points_3d: 3D 점들
            observations: 관찰 데이터 (camera_idx, point_idx, 2D_coords)

        수학적 정의:
        minimize: Σᵢⱼ ||xᵢⱼ - π(Kᵢ, Rᵢ, tᵢ, Xⱼ)||²

        여기서:
        - i: camera index
        - j: 3D point index
        - π: projection function
        - xᵢⱼ: observed 2D point
        """
        self.camera_matrices = camera_matrices
        self.poses = poses
        self.points_3d = points_3d.copy()
        self.observations = observations

        self.n_cameras = len(poses)
        self.n_points = len(points_3d)
        self.n_observations = len(observations)

    def project_point(self, point_3d, camera_matrix, rvec, tvec):
        """
        3D 점을 2D 이미지 평면에 투영

        투영 방정식:
        x = K[R|t]X

        Args:
            point_3d: 3D 점 [3]
            camera_matrix: 내부 파라미터 [3, 3]
            rvec: 회전 벡터 [3]
            tvec: 변환 벡터 [3]

        Returns:
            numpy.ndarray: 투영된 2D 점 [2]
        """
        # Use OpenCV projection
        projected, _ = cv2.projectPoints(
            point_3d.reshape(1, 1, 3),
            rvec, tvec, camera_matrix, None
        )
        return projected.reshape(2)

    def compute_residuals(self, params):
        """
        Residual 계산: 예측 위치와 관찰 위치의 차이

        Residuals = observed_2D - projected_2D

        Args:
            params: 최적화 파라미터
                [camera_params, point_coords]
                camera_params: [rvec1, tvec1, rvec2, tvec2, ...]
                point_coords: [X1, Y1, Z1, X2, Y2, Z2, ...]

        Returns:
            numpy.ndarray: Residual vector [2 * n_observations]

        중요한 점:
        - 각 관찰마다 2D residual (x, y)
        - 전체 residual은 모든 관찰의 concatenation
        - Bundle adjustment는 이 residual의 L2 norm 최소화
        """
        # Extract camera parameters
        camera_params = params[:self.n_cameras * 6].reshape(self.n_cameras, 6)
        point_coords = params[self.n_cameras * 6:].reshape(self.n_points, 3)

        residuals = []

        for obs_idx, (cam_idx, pt_idx, observed_2d) in enumerate(self.observations):
            # Get camera parameters
            rvec = camera_params[cam_idx, :3]
            tvec = camera_params[cam_idx, 3:]
            camera_matrix = self.camera_matrices[cam_idx]

            # Get 3D point
            point_3d = point_coords[pt_idx]

            # Project to 2D
            projected_2d = self.project_point(point_3d, camera_matrix, rvec, tvec)

            # Compute residual
            residual = observed_2d - projected_2d
            residuals.extend(residual)

        return np.array(residuals)

    def optimize(self, max_iterations=100, ftol=1e-6):
        """
        Non-linear least squares optimization

        Levenberg-Marquardt algorithm:
        - Gauss-Newton과 Gradient Descent의 결합
        - Adaptive damping parameter
        - Sparse Jacobian 활용으로 효율성 향상

        Returns:
            dict: 최적화 결과
                - optimized_poses: 최적화된 카메라 포즈
                - optimized_points: 최적화된 3D 점들
                - initial_error: 초기 reprojection error
                - final_error: 최종 reprojection error
        """
        print("Starting Bundle Adjustment optimization...")

        # Initialize parameters
        initial_params = []

        # Add camera parameters (rvec, tvec for each camera)
        for pose in self.poses:
            rvec, _ = cv2.Rodrigues(pose['R'])
            tvec = pose['t'].flatten()
            initial_params.extend(rvec.flatten())
            initial_params.extend(tvec)

        # Add 3D points
        initial_params.extend(self.points_3d.flatten())
        initial_params = np.array(initial_params)

        # Compute initial error
        initial_residuals = self.compute_residuals(initial_params)
        initial_error = np.mean(np.sqrt(initial_residuals[::2]**2 + initial_residuals[1::2]**2))

        print(f"Initial reprojection error: {initial_error:.4f} pixels")

        # Optimize using Levenberg-Marquardt
        try:
            result = scipy.optimize.least_squares(
                self.compute_residuals,
                initial_params,
                method='lm',
                max_nfev=max_iterations * len(initial_params),
                ftol=ftol,
                verbose=1
            )

            optimized_params = result.x
            success = result.success

        except Exception as e:
            print(f"Optimization failed: {e}")
            optimized_params = initial_params
            success = False

        # Extract optimized parameters
        camera_params = optimized_params[:self.n_cameras * 6].reshape(self.n_cameras, 6)
        optimized_points = optimized_params[self.n_cameras * 6:].reshape(self.n_points, 3)

        # Convert camera parameters back to poses
        optimized_poses = []
        for i in range(self.n_cameras):
            rvec = camera_params[i, :3]
            tvec = camera_params[i, 3:]
            R, _ = cv2.Rodrigues(rvec)
            optimized_poses.append({'R': R, 't': tvec.reshape(3, 1)})

        # Compute final error
        final_residuals = self.compute_residuals(optimized_params)
        final_error = np.mean(np.sqrt(final_residuals[::2]**2 + final_residuals[1::2]**2))

        print(f"Final reprojection error: {final_error:.4f} pixels")
        print(f"Error reduction: {initial_error - final_error:.4f} pixels")
        print(f"Optimization {'succeeded' if success else 'failed'}")

        return {
            'optimized_poses': optimized_poses,
            'optimized_points': optimized_points,
            'initial_error': initial_error,
            'final_error': final_error,
            'success': success
        }

class StructureFromMotion:
    def __init__(self, camera_matrix=None):
        """
        Complete Structure from Motion Pipeline

        전체 파이프라인:
        1. Feature extraction from all images
        2. Feature matching between image pairs
        3. Two-view initialization
        4. Incremental reconstruction
        5. Bundle adjustment
        6. Dense reconstruction (optional)

        Args:
            camera_matrix: 카메라 내부 파라미터 (알려진 경우)
        """
        self.camera_matrix = camera_matrix
        self.feature_extractor = FeatureExtractor()
        self.pose_estimator = PoseEstimator(camera_matrix)

        # Storage for reconstruction
        self.images = []
        self.features = []  # (keypoints, descriptors) for each image
        self.matches = {}   # pairwise matches
        self.poses = {}     # camera poses
        self.points_3d = []
        self.point_colors = []
        self.track_info = {}  # which 3D point corresponds to which image features

    def add_image(self, image, image_id=None):
        """
        이미지를 reconstruction에 추가

        Args:
            image: 입력 이미지
            image_id: 이미지 식별자 (None이면 자동 생성)

        Returns:
            int: 이미지 인덱스
        """
        if image_id is None:
            image_id = len(self.images)

        # Extract features
        keypoints, descriptors = self.feature_extractor.extract_features(image)

        self.images.append(image)
        self.features.append((keypoints, descriptors))

        print(f"Added image {image_id} with {len(keypoints)} features")
        return image_id

    def match_all_pairs(self):
        """
        모든 이미지 쌍에 대해 feature matching 수행

        전략:
        - Sequential matching: 연속된 이미지들만 매칭 (비디오)
        - All-pairs matching: 모든 조합 매칭 (unordered images)
        - Loop closure detection: 루프 감지로 drift 교정
        """
        n_images = len(self.images)
        print(f"Matching features for {n_images} images...")

        for i in range(n_images):
            for j in range(i + 1, n_images):
                # Match features between images i and j
                _, desc_i = self.features[i]
                _, desc_j = self.features[j]

                matches = self.feature_extractor.match_features(desc_i, desc_j)

                if len(matches) >= 10:  # Minimum matches threshold
                    self.matches[(i, j)] = matches
                    print(f"  Images {i}-{j}: {len(matches)} matches")

        print(f"Found matches for {len(self.matches)} image pairs")

    def initialize_reconstruction(self):
        """
        Two-view initialization: 가장 좋은 이미지 쌍으로 초기 reconstruction

        선택 기준:
        - 충분한 수의 매치 (>100)
        - 적절한 baseline (너무 작지도 크지도 않게)
        - 좋은 geometric configuration
        - High inlier ratio in essential matrix estimation
        """
        best_pair = None
        best_score = 0

        print("Searching for best initialization pair...")

        for (i, j), matches in self.matches.items():
            if len(matches) < 50:
                continue

            # Get corresponding points
            kpts_i, _ = self.features[i]
            kpts_j, _ = self.features[j]

            pts_i = np.array([kpts_i[m.queryIdx].pt for m in matches])
            pts_j = np.array([kpts_j[m.trainIdx].pt for m in matches])

            # Estimate pose
            result = self.pose_estimator.estimate_pose_two_view(pts_i, pts_j)

            if result['inliers'] is not None:
                inlier_ratio = np.sum(result['inliers']) / len(matches)
                score = inlier_ratio * len(matches)  # Score = inliers * matches

                if score > best_score:
                    best_score = score
                    best_pair = (i, j, result, pts_i, pts_j, matches)

        if best_pair is None:
            raise RuntimeError("Failed to find good initialization pair")

        i, j, pose_result, pts_i, pts_j, matches = best_pair
        print(f"Selected initialization pair: images {i}-{j}")
        print(f"  Matches: {len(matches)}, Inliers: {np.sum(pose_result['inliers'])}")

        # Set up initial cameras
        self.poses[i] = {'R': np.eye(3), 't': np.zeros((3, 1))}  # Reference camera
        self.poses[j] = pose_result

        # Triangulate initial 3D points
        if self.camera_matrix is not None:
            P1 = self.camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = self.camera_matrix @ np.hstack([pose_result['R'], pose_result['t']])

            inlier_pts_i = pts_i[pose_result['inliers']]
            inlier_pts_j = pts_j[pose_result['inliers']]

            points_3d = self.pose_estimator.triangulate_points(inlier_pts_i, inlier_pts_j, P1, P2)

            # Filter out points too far from cameras
            valid_points = []
            valid_colors = []

            for idx, pt_3d in enumerate(points_3d):
                if np.linalg.norm(pt_3d) < 100:  # Reasonable distance
                    valid_points.append(pt_3d)
                    # Get color from first image (simplified)
                    pixel_coord = inlier_pts_i[idx].astype(int)
                    if (0 <= pixel_coord[0] < self.images[i].shape[1] and
                        0 <= pixel_coord[1] < self.images[i].shape[0]):
                        color = self.images[i][pixel_coord[1], pixel_coord[0]]
                        valid_colors.append(color)
                    else:
                        valid_colors.append([128, 128, 128])  # Gray for invalid pixels

            self.points_3d = np.array(valid_points)
            self.point_colors = np.array(valid_colors)

            print(f"Initialized with {len(self.points_3d)} 3D points")

        return best_pair

    def incremental_reconstruction(self):
        """
        Incremental SfM: 한 번에 하나씩 이미지 추가

        과정:
        1. 다음 추가할 이미지 선택 (가장 많은 3D 점과 매치되는)
        2. PnP로 카메라 포즈 추정
        3. 새로운 3D 점들 triangulation
        4. Local bundle adjustment
        5. 반복

        이 방법의 장점:
        - 안정적이고 robust
        - 실패한 이미지를 쉽게 제외 가능
        - 중간 결과 확인 가능
        """
        registered_images = set(self.poses.keys())
        unregistered_images = set(range(len(self.images))) - registered_images

        print(f"Starting incremental reconstruction...")
        print(f"Registered: {len(registered_images)}, Remaining: {len(unregistered_images)}")

        while unregistered_images:
            next_image = self.select_next_image(registered_images, unregistered_images)

            if next_image is None:
                print("No more images can be registered")
                break

            success = self.register_image(next_image, registered_images)

            if success:
                registered_images.add(next_image)
                unregistered_images.remove(next_image)
                print(f"Successfully registered image {next_image}")

                # Perform local bundle adjustment every few images
                if len(registered_images) % 5 == 0:
                    self.local_bundle_adjustment(list(registered_images)[-5:])

            else:
                print(f"Failed to register image {next_image}")
                unregistered_images.remove(next_image)

        print(f"Final reconstruction: {len(registered_images)} cameras, {len(self.points_3d)} points")

    def select_next_image(self, registered_images, unregistered_images):
        """
        다음에 등록할 이미지 선택

        선택 기준:
        - 가장 많은 3D 점과 correspondence를 가짐
        - 좋은 viewing angle (너무 비슷하지 않은 시점)
        - 충분한 수의 matches
        """
        best_image = None
        best_score = 0

        for candidate in unregistered_images:
            score = 0

            # Count correspondences with registered images
            for registered in registered_images:
                pair = tuple(sorted([candidate, registered]))
                if pair in self.matches:
                    score += len(self.matches[pair])

            if score > best_score:
                best_score = score
                best_image = candidate

        return best_image if best_score > 20 else None

    def register_image(self, image_idx, registered_images):
        """
        단일 이미지를 기존 reconstruction에 등록

        PnP 문제:
        - 기존 3D 점들과 새 이미지의 2D 점들 매칭
        - 6DOF 카메라 포즈 추정
        - RANSAC으로 outlier 제거
        """
        if self.camera_matrix is None:
            return False

        # Find 2D-3D correspondences
        points_3d = []
        points_2d = []

        kpts, _ = self.features[image_idx]

        for registered_idx in registered_images:
            pair = tuple(sorted([image_idx, registered_idx]))
            if pair not in self.matches:
                continue

            matches = self.matches[pair]
            for match in matches:
                if registered_idx < image_idx:
                    query_idx = match.trainIdx
                    train_idx = match.queryIdx
                else:
                    query_idx = match.queryIdx
                    train_idx = match.trainIdx

                # Check if this feature corresponds to a 3D point
                # (simplified - 실제로는 더 복잡한 track management 필요)
                if len(self.points_3d) > train_idx:
                    points_3d.append(self.points_3d[train_idx % len(self.points_3d)])
                    points_2d.append(kpts[query_idx].pt)

        if len(points_3d) < 6:  # Minimum points for PnP
            return False

        points_3d = np.array(points_3d)
        points_2d = np.array(points_2d)

        # Solve PnP
        result = self.pose_estimator.pnp_pose_estimation(
            points_3d, points_2d, self.camera_matrix
        )

        if result['success']:
            self.poses[image_idx] = result
            return True

        return False

    def local_bundle_adjustment(self, image_indices):
        """
        Local Bundle Adjustment: 최근 등록된 이미지들에 대해서만 최적화

        전역 BA vs 로컬 BA:
        - 전역: 모든 카메라와 점 최적화 (정확하지만 느림)
        - 로컬: 일부만 최적화 (빠르지만 drift 가능)
        - Sliding window approach
        """
        if len(image_indices) < 2:
            return

        print(f"Performing local bundle adjustment on {len(image_indices)} images...")

        # Prepare data for bundle adjustment
        local_poses = [self.poses[i] for i in image_indices]
        local_camera_matrices = [self.camera_matrix] * len(image_indices)

        # Create observations (simplified)
        observations = []
        for i, img_idx in enumerate(image_indices):
            for j, pt_3d in enumerate(self.points_3d[:min(100, len(self.points_3d))]):
                # Simplified: assume we can observe all points from all cameras
                # 실제로는 track information 필요
                projected = cv2.projectPoints(
                    pt_3d.reshape(1, 1, 3),
                    cv2.Rodrigues(local_poses[i]['R'])[0],
                    local_poses[i]['t'],
                    self.camera_matrix,
                    None
                )[0].reshape(2)

                observations.append((i, j, projected))

        if len(observations) > 0:
            ba = BundleAdjustment(
                local_camera_matrices,
                local_poses,
                self.points_3d[:len(observations)//len(image_indices)],
                observations
            )

            result = ba.optimize()

            if result['success']:
                # Update poses
                for i, img_idx in enumerate(image_indices):
                    self.poses[img_idx] = result['optimized_poses'][i]

                print(f"BA improved error from {result['initial_error']:.3f} to {result['final_error']:.3f}")

    def run_full_pipeline(self, images):
        """
        전체 SfM 파이프라인 실행

        Args:
            images: 입력 이미지 리스트

        Returns:
            dict: Reconstruction 결과
        """
        print("=" * 50)
        print("Starting Structure from Motion Pipeline")
        print("=" * 50)

        # Add all images
        for i, image in enumerate(images):
            self.add_image(image, i)

        # Match features between all pairs
        self.match_all_pairs()

        # Initialize with best image pair
        self.initialize_reconstruction()

        # Incremental reconstruction
        self.incremental_reconstruction()

        # Final global bundle adjustment
        if len(self.poses) > 2:
            print("Performing final global bundle adjustment...")
            all_indices = list(self.poses.keys())
            self.local_bundle_adjustment(all_indices)

        print("=" * 50)
        print("Structure from Motion Complete")
        print(f"Reconstructed {len(self.poses)} cameras")
        print(f"Reconstructed {len(self.points_3d)} 3D points")
        print("=" * 50)

        return {
            'poses': self.poses,
            'points_3d': self.points_3d,
            'point_colors': self.point_colors,
            'camera_matrix': self.camera_matrix
        }

# Visualization and utilities
def visualize_reconstruction(poses, points_3d, point_colors=None):
    """
    3D reconstruction 결과 시각화

    Args:
        poses: 카메라 포즈들
        points_3d: 3D 점들
        point_colors: 점들의 색상 (optional)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    if len(points_3d) > 0:
        colors = point_colors if point_colors is not None else 'blue'
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                  c=colors, s=1, alpha=0.6)

    # Plot camera poses
    for i, pose in poses.items():
        R = pose['R']
        t = pose['t'].flatten()

        # Camera center
        center = -R.T @ t
        ax.scatter(*center, color='red', s=100, marker='^')

        # Camera orientation (simplified)
        forward = R[:, 2] * 0.5  # Z-axis points forward
        ax.quiver(*center, *forward, color='red', alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Structure from Motion Reconstruction')

    # Equal aspect ratio
    max_range = np.array([points_3d[:, 0].max() - points_3d[:, 0].min(),
                         points_3d[:, 1].max() - points_3d[:, 1].min(),
                         points_3d[:, 2].max() - points_3d[:, 2].min()]).max() / 2.0
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

# Example usage
def create_synthetic_dataset():
    """
    SfM 테스트용 합성 데이터셋 생성

    실제 사용시에는:
    - 실제 이미지 로드
    - EXIF에서 카메라 정보 추출
    - 또는 calibration으로 카메라 파라미터 추정
    """
    # Create a simple 3D scene (cube)
    cube_points = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Front face
    ]) * 2.0

    # Camera parameters
    camera_matrix = np.array([
        [500, 0, 250],
        [0, 500, 250],
        [0, 0, 1]
    ], dtype=np.float32)

    # Generate images from different viewpoints
    images = []
    true_poses = []

    for i in range(8):
        # Circular camera trajectory
        angle = i * np.pi / 4
        camera_pos = np.array([3 * np.cos(angle), 1, 3 * np.sin(angle)])

        # Look at origin
        look_at = np.array([0, 0, 0])
        up = np.array([0, 1, 0])

        # Create rotation matrix (look-at)
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        R = np.column_stack([right, up, -forward])
        t = -R @ camera_pos

        true_poses.append({'R': R, 't': t.reshape(3, 1)})

        # Project 3D points to create synthetic image
        projected_points = []
        for pt_3d in cube_points:
            pt_cam = R @ pt_3d + t
            pt_2d = camera_matrix @ pt_cam
            pt_2d = pt_2d[:2] / pt_2d[2]
            projected_points.append(pt_2d)

        # Create a simple image with feature points
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        for pt_2d in projected_points:
            if 0 <= pt_2d[0] < 500 and 0 <= pt_2d[1] < 500:
                cv2.circle(image, tuple(pt_2d.astype(int)), 10, (255, 255, 255), -1)

        images.append(image)

    return images, camera_matrix, true_poses, cube_points

if __name__ == "__main__":
    print("Structure from Motion Example")
    print("This implementation demonstrates:")
    print("- Feature extraction and matching")
    print("- Two-view geometry and pose estimation")
    print("- Triangulation and 3D point reconstruction")
    print("- Bundle adjustment optimization")
    print("- Incremental reconstruction pipeline")

    # Create synthetic dataset
    print("\\nGenerating synthetic dataset...")
    images, camera_matrix, true_poses, true_points = create_synthetic_dataset()

    # Run SfM pipeline
    sfm = StructureFromMotion(camera_matrix)
    result = sfm.run_full_pipeline(images)

    print("\\nReconstruction completed!")
    print(f"Ground truth: {len(true_poses)} cameras, {len(true_points)} points")
    print(f"Reconstructed: {len(result['poses'])} cameras, {len(result['points_3d'])} points")

    # Visualize result (if matplotlib available)
    try:
        visualize_reconstruction(result['poses'], result['points_3d'], result['point_colors'])
    except Exception as e:
        print(f"Visualization not available: {e}")

    print("\\nNote: This is an educational implementation.")
    print("For production use, consider:")
    print("- COLMAP: State-of-the-art SfM pipeline")
    print("- OpenMVG: Open source MVG library")
    print("- Theia: C++ library for multiview geometry")
    print("- OpenCV's sfm module")