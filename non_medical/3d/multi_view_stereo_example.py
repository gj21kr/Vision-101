"""
Multi-View Stereo (MVS) Implementation

Multi-View Stereo는 여러 시점의 이미지와 카메라 포즈 정보를 이용해
dense 3D reconstruction을 수행하는 기술입니다. Structure from Motion이
sparse한 point cloud를 생성한다면, MVS는 조밀한 3D 모델을 만듭니다.

핵심 개념:

1. **Depth Map Estimation**:
   - 각 reference view에 대해 depth map 계산
   - Photo-consistency를 이용한 matching cost 계산
   - Winner-takes-all 또는 optimization 기반 depth 선택

2. **Photo-consistency Measures**:
   - NCC (Normalized Cross Correlation)
   - SAD (Sum of Absolute Differences)
   - Census Transform
   - Patch-based matching

3. **Multi-View Fusion**:
   - 여러 view의 depth map을 하나로 통합
   - Visibility reasoning으로 occlusion 처리
   - Outlier 제거 및 hole filling

4. **Volumetric Methods**:
   - Voxel-based representation
   - Level sets or signed distance functions
   - Graph cuts for global optimization

수학적 원리:

Photo-consistency:
- 동일한 3D 점이 여러 이미지에서 비슷한 색상을 가져야 함
- C(x,y,d) = similarity(I₁(x,y), I₂(π(x,y,d)), ..., Iₙ(π(x,y,d)))

Plane-sweep algorithm:
- 각 depth에 대해 모든 픽셀의 matching cost 계산
- d* = argmax_d C(x,y,d)

Epipolar geometry constraint:
- 대응점은 epipolar line 위에 존재
- Rectification으로 수평 탐색으로 단순화

장점:
- Dense reconstruction (모든 픽셀에 대해 depth)
- 높은 정확도 (subpixel precision 가능)
- 텍스처가 풍부한 영역에서 우수한 성능
- 잘 정립된 이론과 다양한 구현

단점:
- 텍스처가 부족한 영역에서 어려움
- Occlusion과 reflection 처리 복잡
- 계산 비용이 높음
- Lighting 변화에 민감

Reference:
- Seitz, S. M., et al. (2006).
  "A comparison and evaluation of multi-view stereo reconstruction algorithms."
  Conference on Computer Vision and Pattern Recognition (CVPR).
- Furukawa, Y. & Ponce, J. (2010).
  "Accurate, dense, and robust multiview stereopsis."
  IEEE Transactions on Pattern Analysis and Machine Intelligence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
import warnings

class PhotoConsistency:
    def __init__(self, method='ncc', patch_size=5):
        """
        Photo-consistency measure for multi-view matching

        다양한 similarity measures:
        - NCC: Normalized Cross Correlation (조명 변화에 강인)
        - SAD: Sum of Absolute Differences (간단하고 빠름)
        - SSD: Sum of Squared Differences (전통적 방법)
        - Census: Census Transform (robust to illumination)

        Args:
            method: Similarity measure 방법
            patch_size: Matching patch 크기 (홀수)
        """
        self.method = method
        self.patch_size = patch_size
        self.half_patch = patch_size // 2

    def compute_ncc(self, patch1, patch2):
        """
        Normalized Cross Correlation 계산

        NCC의 장점:
        - 조명 변화에 강인 (normalization 효과)
        - [-1, 1] 범위의 직관적 값
        - 1에 가까울수록 유사함

        수학적 정의:
        NCC = Σ((I₁ - μ₁)(I₂ - μ₂)) / √(Σ(I₁ - μ₁)² × Σ(I₂ - μ₂)²)

        Args:
            patch1, patch2: 비교할 패치들 [patch_size, patch_size]

        Returns:
            float: NCC 값 [-1, 1]
        """
        if patch1.size == 0 or patch2.size == 0:
            return -1.0

        # Convert to float and flatten
        p1 = patch1.astype(np.float32).flatten()
        p2 = patch2.astype(np.float32).flatten()

        # Compute means
        mean1 = np.mean(p1)
        mean2 = np.mean(p2)

        # Center the patches
        p1_centered = p1 - mean1
        p2_centered = p2 - mean2

        # Compute NCC
        numerator = np.sum(p1_centered * p2_centered)
        denominator = np.sqrt(np.sum(p1_centered**2) * np.sum(p2_centered**2))

        if denominator < 1e-8:
            return -1.0

        return numerator / denominator

    def compute_sad(self, patch1, patch2):
        """
        Sum of Absolute Differences 계산

        SAD의 특징:
        - 계산이 매우 빠름
        - 0에 가까울수록 유사함
        - 조명 변화에 민감
        """
        return np.sum(np.abs(patch1.astype(np.float32) - patch2.astype(np.float32)))

    def compute_census(self, patch1, patch2):
        """
        Census Transform 기반 similarity

        Census Transform:
        - 각 픽셀을 중심으로 주변 픽셀들과 비교
        - Binary string 생성 (brighter=1, darker=0)
        - Hamming distance로 similarity 계산

        장점:
        - 조명 변화에 매우 강인
        - Outlier에 robust
        """
        def census_transform(patch):
            center = patch[self.half_patch, self.half_patch]
            binary = (patch > center).astype(np.uint8)
            return binary.flatten()

        census1 = census_transform(patch1)
        census2 = census_transform(patch2)

        # Hamming distance
        hamming = np.sum(census1 != census2)
        return hamming / len(census1)  # Normalize to [0, 1]

    def compute_similarity(self, patch1, patch2):
        """
        선택된 방법으로 patch similarity 계산

        Returns:
            float: Similarity score (높을수록 유사함)
        """
        if self.method == 'ncc':
            return self.compute_ncc(patch1, patch2)
        elif self.method == 'sad':
            return -self.compute_sad(patch1, patch2)  # Negative for higher=better
        elif self.method == 'census':
            return 1.0 - self.compute_census(patch1, patch2)
        else:
            raise ValueError(f"Unknown method: {self.method}")

class PlaneSweep:
    def __init__(self, photo_measure='ncc', patch_size=5):
        """
        Plane Sweep Algorithm for depth estimation

        Plane Sweep의 원리:
        1. 일정한 간격의 depth planes 설정
        2. 각 plane에 reference image를 projection
        3. 다른 view들에서 해당 위치의 패치 추출
        4. Photo-consistency 계산
        5. 최적 depth 선택

        Args:
            photo_measure: Photo-consistency 방법
            patch_size: Matching patch 크기
        """
        self.photo_consistency = PhotoConsistency(photo_measure, patch_size)
        self.patch_size = patch_size
        self.half_patch = patch_size // 2

    def create_depth_planes(self, depth_min, depth_max, n_planes):
        """
        Uniform depth planes 생성

        Depth sampling strategies:
        - Uniform: 균등한 간격
        - Inverse: 1/d space에서 균등 (원근감 반영)
        - Adaptive: 텍스처 기반 adaptive sampling

        Args:
            depth_min, depth_max: Depth 범위
            n_planes: Plane 개수

        Returns:
            numpy.ndarray: Depth values [n_planes]
        """
        return np.linspace(depth_min, depth_max, n_planes)

    def project_to_camera(self, points_3d, camera_matrix, R, t):
        """
        3D 점들을 카메라 이미지 평면에 투영

        Camera projection:
        x = K[R|t]X

        Args:
            points_3d: 3D 점들 [N, 3]
            camera_matrix: 내부 파라미터 [3, 3]
            R: 회전 행렬 [3, 3]
            t: 변환 벡터 [3, 1]

        Returns:
            numpy.ndarray: 2D 투영 점들 [N, 2]
        """
        # Transform to camera coordinates
        points_cam = (R @ points_3d.T + t).T

        # Project to image plane
        points_2d_homo = (camera_matrix @ points_cam.T).T

        # Convert from homogeneous coordinates
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]

        return points_2d

    def compute_depth_map(self, ref_image, ref_camera, source_images, source_cameras,
                         depth_planes, verbose=False):
        """
        Reference view에 대한 depth map 계산

        Algorithm:
        1. 각 픽셀 (u,v)에 대해
        2. 각 depth d에 대해
           - 3D 점 X = K⁻¹[u,v,1]ᵀ × d 계산
           - X를 모든 source view에 투영
           - Photo-consistency 계산
        3. 최적 depth 선택

        Args:
            ref_image: Reference 이미지 [H, W, C]
            ref_camera: Reference 카메라 정보 dict{'K', 'R', 't'}
            source_images: Source 이미지들 list
            source_cameras: Source 카메라 정보들 list
            depth_planes: Depth plane 값들

        Returns:
            tuple: (depth_map, confidence_map)
        """
        if verbose:
            print(f"Computing depth map for {len(source_images)} source views...")

        height, width = ref_image.shape[:2]
        n_planes = len(depth_planes)

        # Initialize cost volume
        cost_volume = np.zeros((height, width, n_planes))

        # Camera parameters
        K_ref = ref_camera['K']
        K_ref_inv = np.linalg.inv(K_ref)
        R_ref = ref_camera['R']
        t_ref = ref_camera['t']

        # For each pixel in reference image
        for y in range(self.half_patch, height - self.half_patch):
            for x in range(self.half_patch, width - self.half_patch):
                # Reference patch
                ref_patch = ref_image[y-self.half_patch:y+self.half_patch+1,
                                    x-self.half_patch:x+self.half_patch+1]

                if len(ref_patch.shape) == 3:
                    ref_patch = cv2.cvtColor(ref_patch, cv2.COLOR_RGB2GRAY)

                # For each depth plane
                for d_idx, depth in enumerate(depth_planes):
                    # Compute 3D point
                    pixel_homo = np.array([x, y, 1.0])
                    ray_dir = K_ref_inv @ pixel_homo
                    point_3d = (R_ref.T @ (depth * ray_dir.reshape(-1, 1) - t_ref)).flatten()

                    # Accumulate photo-consistency across all source views
                    consistency_sum = 0.0
                    valid_views = 0

                    for src_idx, (src_image, src_camera) in enumerate(zip(source_images, source_cameras)):
                        # Project to source view
                        projected = self.project_to_camera(
                            point_3d.reshape(1, -1),
                            src_camera['K'],
                            src_camera['R'],
                            src_camera['t']
                        )[0]

                        px, py = projected.astype(int)

                        # Check if projection is within image bounds
                        if (self.half_patch <= px < width - self.half_patch and
                            self.half_patch <= py < height - self.half_patch):

                            # Extract source patch
                            src_patch = src_image[py-self.half_patch:py+self.half_patch+1,
                                                px-self.half_patch:px+self.half_patch+1]

                            if len(src_patch.shape) == 3:
                                src_patch = cv2.cvtColor(src_patch, cv2.COLOR_RGB2GRAY)

                            # Compute photo-consistency
                            similarity = self.photo_consistency.compute_similarity(ref_patch, src_patch)
                            consistency_sum += similarity
                            valid_views += 1

                    # Average consistency across valid views
                    if valid_views > 0:
                        cost_volume[y, x, d_idx] = consistency_sum / valid_views

            if verbose and y % 50 == 0:
                print(f"  Processed row {y}/{height}")

        # Winner-takes-all depth selection
        depth_map = np.zeros((height, width))
        confidence_map = np.zeros((height, width))

        for y in range(height):
            for x in range(width):
                costs = cost_volume[y, x, :]
                if np.max(costs) > -np.inf:
                    best_idx = np.argmax(costs)
                    depth_map[y, x] = depth_planes[best_idx]
                    confidence_map[y, x] = costs[best_idx]

        return depth_map, confidence_map

class DepthMapFusion:
    def __init__(self, consistency_threshold=0.6, depth_tolerance=0.02):
        """
        Multiple depth maps fusion into single dense point cloud

        Fusion strategies:
        1. Visibility-based: 각 3D 점이 보이는 view들만 고려
        2. Consistency check: 여러 view에서 일치하는 depth만 유지
        3. Outlier removal: 고립된 점들 제거
        4. Hole filling: 빈 영역 보간

        Args:
            consistency_threshold: 일관성 판정 임계값
            depth_tolerance: Depth 차이 허용 범위 (relative)
        """
        self.consistency_threshold = consistency_threshold
        self.depth_tolerance = depth_tolerance

    def fuse_depth_maps(self, depth_maps, confidence_maps, cameras, min_views=2):
        """
        Multiple depth maps를 하나의 3D point cloud로 융합

        Algorithm:
        1. 각 depth map에서 3D 점들 생성
        2. 같은 3D 위치에 대응하는 점들 그룹화
        3. Consistency check로 outlier 제거
        4. 최종 3D coordinates와 colors 결정

        Args:
            depth_maps: List of depth maps
            confidence_maps: List of confidence maps
            cameras: List of camera parameters
            min_views: 최소 관찰 view 수

        Returns:
            dict: Fused point cloud
                - points_3d: 3D coordinates [N, 3]
                - colors: RGB colors [N, 3]
                - confidences: Confidence scores [N]
        """
        print(f"Fusing {len(depth_maps)} depth maps...")

        all_points = []
        all_colors = []
        all_confidences = []
        all_view_ids = []

        # Convert each depth map to 3D points
        for view_idx, (depth_map, conf_map, camera) in enumerate(zip(depth_maps, confidence_maps, cameras)):
            points_3d, colors, confidences = self.depth_map_to_points(
                depth_map, conf_map, camera, view_idx
            )

            if len(points_3d) > 0:
                all_points.append(points_3d)
                all_colors.append(colors)
                all_confidences.append(confidences)
                all_view_ids.extend([view_idx] * len(points_3d))

        if not all_points:
            return {'points_3d': np.array([]), 'colors': np.array([]), 'confidences': np.array([])}

        # Concatenate all points
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        all_confidences = np.concatenate(all_confidences)
        all_view_ids = np.array(all_view_ids)

        print(f"Generated {len(all_points)} total points from all views")

        # Spatial clustering and consistency check
        fused_points, fused_colors, fused_confidences = self.spatial_consistency_check(
            all_points, all_colors, all_confidences, all_view_ids, min_views
        )

        print(f"After fusion and consistency check: {len(fused_points)} points")

        return {
            'points_3d': fused_points,
            'colors': fused_colors,
            'confidences': fused_confidences
        }

    def depth_map_to_points(self, depth_map, confidence_map, camera, view_idx):
        """
        Depth map을 3D point cloud로 변환

        Args:
            depth_map: Depth values [H, W]
            confidence_map: Confidence scores [H, W]
            camera: Camera parameters
            view_idx: View identifier

        Returns:
            tuple: (points_3d, colors, confidences)
        """
        height, width = depth_map.shape
        K = camera['K']
        K_inv = np.linalg.inv(K)
        R = camera['R']
        t = camera['t']

        points_3d = []
        colors = []
        confidences = []

        # Get reference image for colors (if available)
        ref_image = camera.get('image', np.ones((height, width, 3)) * 128)

        for y in range(height):
            for x in range(width):
                depth = depth_map[y, x]
                conf = confidence_map[y, x]

                # Skip invalid or low-confidence depths
                if depth <= 0 or conf < self.consistency_threshold:
                    continue

                # Convert to 3D point
                pixel_homo = np.array([x, y, 1.0])
                ray_dir = K_inv @ pixel_homo
                point_cam = depth * ray_dir

                # Transform to world coordinates
                point_world = R.T @ (point_cam.reshape(-1, 1) - t)
                points_3d.append(point_world.flatten())

                # Get color
                if len(ref_image.shape) == 3:
                    color = ref_image[y, x]
                else:
                    color = [ref_image[y, x]] * 3
                colors.append(color)

                confidences.append(conf)

        return np.array(points_3d), np.array(colors), np.array(confidences)

    def spatial_consistency_check(self, points, colors, confidences, view_ids, min_views):
        """
        Spatial consistency check for outlier removal

        Strategy:
        1. Spatial binning으로 nearby points 그룹화
        2. 각 bin에서 multiple view 관찰 여부 확인
        3. Depth consistency check
        4. Final point selection

        Args:
            points: All 3D points [N, 3]
            colors: Point colors [N, 3]
            confidences: Point confidences [N]
            view_ids: View identifiers [N]
            min_views: Minimum observation views

        Returns:
            tuple: (filtered_points, filtered_colors, filtered_confidences)
        """
        if len(points) == 0:
            return np.array([]), np.array([]), np.array([])

        # Spatial binning (simplified octree-like approach)
        # 실제 구현에서는 KD-tree나 octree 사용
        voxel_size = 0.05  # 5cm voxels

        # Quantize points to voxels
        voxel_coords = (points / voxel_size).astype(int)

        # Group points by voxel
        voxel_dict = {}
        for i, voxel in enumerate(voxel_coords):
            key = tuple(voxel)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(i)

        # Process each voxel
        fused_points = []
        fused_colors = []
        fused_confidences = []

        for voxel_key, point_indices in voxel_dict.items():
            if len(point_indices) < min_views:
                continue

            # Get unique views in this voxel
            unique_views = np.unique(view_ids[point_indices])

            if len(unique_views) >= min_views:
                # Compute average point (weighted by confidence)
                voxel_points = points[point_indices]
                voxel_colors = colors[point_indices]
                voxel_confs = confidences[point_indices]

                # Weighted average
                weights = voxel_confs / np.sum(voxel_confs)
                avg_point = np.average(voxel_points, axis=0, weights=weights)
                avg_color = np.average(voxel_colors, axis=0, weights=weights)
                avg_conf = np.mean(voxel_confs)

                fused_points.append(avg_point)
                fused_colors.append(avg_color)
                fused_confidences.append(avg_conf)

        return (np.array(fused_points) if fused_points else np.array([]),
                np.array(fused_colors) if fused_colors else np.array([]),
                np.array(fused_confidences) if fused_confidences else np.array([]))

class MultiViewStereo:
    def __init__(self, photo_measure='ncc', patch_size=7):
        """
        Complete Multi-View Stereo pipeline

        전체 파이프라인:
        1. Depth map estimation for each view
        2. Confidence-based filtering
        3. Multi-view fusion
        4. Outlier removal and smoothing
        5. Mesh generation (optional)

        Args:
            photo_measure: Photo-consistency measure
            patch_size: Matching patch size
        """
        self.plane_sweep = PlaneSweep(photo_measure, patch_size)
        self.fusion = DepthMapFusion()

    def estimate_depth_range(self, cameras, scene_bounds=None):
        """
        Estimate reasonable depth range for plane sweep

        방법들:
        1. Camera baseline 기반 추정
        2. SfM sparse points 기반 추정
        3. Manual specification
        4. Adaptive range estimation

        Args:
            cameras: List of camera parameters
            scene_bounds: Optional scene bounding box

        Returns:
            tuple: (depth_min, depth_max)
        """
        if scene_bounds is not None:
            # Use provided scene bounds
            return scene_bounds['depth_min'], scene_bounds['depth_max']

        # Estimate from camera positions
        camera_positions = []
        for camera in cameras:
            R, t = camera['R'], camera['t']
            camera_center = -R.T @ t
            camera_positions.append(camera_center.flatten())

        camera_positions = np.array(camera_positions)

        # Estimate scene depth based on camera distribution
        camera_distances = np.linalg.norm(camera_positions, axis=1)
        depth_min = np.min(camera_distances) * 0.1  # 10% of nearest camera
        depth_max = np.max(camera_distances) * 3.0   # 3x of farthest camera

        return max(depth_min, 0.1), min(depth_max, 100.0)

    def run_mvs_pipeline(self, images, cameras, n_depth_planes=64, scene_bounds=None):
        """
        Complete MVS pipeline execution

        Args:
            images: List of input images
            cameras: List of camera parameters
            n_depth_planes: Number of depth planes for plane sweep
            scene_bounds: Optional scene bounds for depth range

        Returns:
            dict: MVS reconstruction results
        """
        print("=" * 50)
        print("Starting Multi-View Stereo Pipeline")
        print("=" * 50)

        n_views = len(images)
        print(f"Processing {n_views} views with {n_depth_planes} depth planes")

        # Estimate depth range
        depth_min, depth_max = self.estimate_depth_range(cameras, scene_bounds)
        depth_planes = self.plane_sweep.create_depth_planes(depth_min, depth_max, n_depth_planes)

        print(f"Depth range: {depth_min:.2f} - {depth_max:.2f}")

        # Compute depth maps for each view
        depth_maps = []
        confidence_maps = []

        for ref_idx in range(n_views):
            print(f"\\nComputing depth map for view {ref_idx}/{n_views}...")

            # Select source views (all others)
            source_indices = [i for i in range(n_views) if i != ref_idx]
            source_images = [images[i] for i in source_indices]
            source_cameras = [cameras[i] for i in source_indices]

            # Add reference image to camera for color information
            cameras[ref_idx]['image'] = images[ref_idx]

            # Compute depth map
            depth_map, conf_map = self.plane_sweep.compute_depth_map(
                images[ref_idx],
                cameras[ref_idx],
                source_images,
                source_cameras,
                depth_planes,
                verbose=True
            )

            depth_maps.append(depth_map)
            confidence_maps.append(conf_map)

        # Fuse all depth maps
        print("\\nFusing depth maps...")
        fusion_result = self.fusion.fuse_depth_maps(
            depth_maps, confidence_maps, cameras, min_views=2
        )

        print("=" * 50)
        print("Multi-View Stereo Complete")
        print(f"Generated {len(fusion_result['points_3d'])} 3D points")
        print("=" * 50)

        return {
            'depth_maps': depth_maps,
            'confidence_maps': confidence_maps,
            'points_3d': fusion_result['points_3d'],
            'colors': fusion_result['colors'],
            'confidences': fusion_result['confidences']
        }

# Post-processing utilities
class MVSPostProcessing:
    def __init__(self):
        """
        MVS 결과 후처리

        포함되는 기능:
        - Outlier removal (statistical, spatial)
        - Hole filling
        - Surface smoothing
        - Mesh generation
        """
        pass

    def statistical_outlier_removal(self, points, k_neighbors=20, std_ratio=2.0):
        """
        Statistical outlier removal

        방법:
        1. 각 점의 k nearest neighbors 찾기
        2. Average distance 계산
        3. Gaussian distribution 가정하고 outlier 제거

        Args:
            points: 3D points [N, 3]
            k_neighbors: 이웃 점 개수
            std_ratio: Standard deviation threshold

        Returns:
            numpy.ndarray: Filtered point indices
        """
        if len(points) < k_neighbors:
            return np.arange(len(points))

        # 실제 구현에서는 KD-tree나 Ball-tree 사용
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1)  # +1 for self
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)

        # Average distance to neighbors (excluding self)
        avg_distances = np.mean(distances[:, 1:], axis=1)

        # Statistical filtering
        mean_dist = np.mean(avg_distances)
        std_dist = np.std(avg_distances)
        threshold = mean_dist + std_ratio * std_dist

        inlier_mask = avg_distances < threshold
        return np.where(inlier_mask)[0]

    def spatial_filtering(self, points, voxel_size=0.01, min_points=10):
        """
        Spatial filtering using voxel grid

        방법:
        1. Voxel grid로 공간 분할
        2. 각 voxel의 점 개수 확인
        3. 충분한 점이 있는 voxel만 유지

        Args:
            points: 3D points [N, 3]
            voxel_size: Voxel 크기
            min_points: Voxel당 최소 점 수

        Returns:
            numpy.ndarray: Filtered point indices
        """
        if len(points) == 0:
            return np.array([])

        # Voxelize
        voxel_coords = (points / voxel_size).astype(int)

        # Count points per voxel
        voxel_counts = {}
        point_to_voxel = {}

        for i, voxel in enumerate(voxel_coords):
            key = tuple(voxel)
            if key not in voxel_counts:
                voxel_counts[key] = 0
            voxel_counts[key] += 1
            point_to_voxel[i] = key

        # Filter points
        valid_indices = []
        for i, voxel_key in point_to_voxel.items():
            if voxel_counts[voxel_key] >= min_points:
                valid_indices.append(i)

        return np.array(valid_indices)

# Visualization utilities
def visualize_depth_map(depth_map, confidence_map=None, title="Depth Map"):
    """
    Depth map 시각화

    Args:
        depth_map: Depth values [H, W]
        confidence_map: Optional confidence map [H, W]
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2 if confidence_map is not None else 1, figsize=(12, 6))

    if confidence_map is None:
        axes = [axes]

    # Depth map
    valid_mask = depth_map > 0
    if np.any(valid_mask):
        depth_vis = depth_map.copy()
        depth_vis[~valid_mask] = np.nan

        im1 = axes[0].imshow(depth_vis, cmap='jet')
        axes[0].set_title(f"{title} - Depth")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], label='Depth')

    # Confidence map
    if confidence_map is not None:
        im2 = axes[1].imshow(confidence_map, cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title(f"{title} - Confidence")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], label='Confidence')

    plt.tight_layout()
    plt.show()

def visualize_point_cloud(points_3d, colors=None, title="MVS Point Cloud"):
    """
    3D point cloud 시각화

    Args:
        points_3d: 3D points [N, 3]
        colors: Point colors [N, 3] (optional)
        title: Plot title
    """
    if len(points_3d) == 0:
        print("No points to visualize")
        return

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    if colors is not None and len(colors) == len(points_3d):
        # Normalize colors to [0, 1]
        if colors.max() > 1:
            colors = colors / 255.0
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                  c=colors, s=1, alpha=0.6)
    else:
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                  c='blue', s=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

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

# Example synthetic dataset
def create_synthetic_mvs_dataset():
    """
    MVS 테스트용 synthetic dataset 생성

    실제 사용시에는:
    - SfM으로 카메라 포즈 추정 후 사용
    - 또는 calibrated camera setup
    - COLMAP, OpenMVG 등의 결과 활용
    """
    # Create synthetic scene (textured plane)
    plane_points = []
    plane_colors = []

    # Checkerboard pattern
    for x in np.linspace(-2, 2, 20):
        for z in np.linspace(-2, 2, 20):
            y = 0  # Ground plane
            plane_points.append([x, y, z])

            # Checkerboard color
            checker = ((int(x * 2) + int(z * 2)) % 2) * 255
            plane_colors.append([checker, checker, checker])

    plane_points = np.array(plane_points)
    plane_colors = np.array(plane_colors)

    # Camera parameters
    camera_matrix = np.array([
        [500, 0, 250],
        [0, 500, 250],
        [0, 0, 1]
    ], dtype=np.float32)

    # Generate synthetic images and camera poses
    images = []
    cameras = []

    for i in range(5):  # 5 views
        # Circular camera trajectory
        angle = i * np.pi / 6  # 30 degree steps
        camera_pos = np.array([2 * np.cos(angle), 1, 2 * np.sin(angle)])

        # Look at center
        look_at = np.array([0, 0, 0])
        up = np.array([0, 1, 0])

        # Create camera pose
        forward = look_at - camera_pos
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        R = np.column_stack([right, up, -forward])
        t = -R @ camera_pos.reshape(-1, 1)

        cameras.append({
            'K': camera_matrix,
            'R': R,
            't': t
        })

        # Render synthetic image
        image = np.ones((500, 500, 3), dtype=np.uint8) * 128  # Gray background

        for pt_3d, color in zip(plane_points, plane_colors):
            # Project to image
            pt_cam = R @ pt_3d + t.flatten()
            if pt_cam[2] > 0:  # In front of camera
                pt_2d = camera_matrix @ pt_cam
                pt_2d = pt_2d[:2] / pt_2d[2]

                px, py = pt_2d.astype(int)
                if 0 <= px < 500 and 0 <= py < 500:
                    # Draw small circle
                    cv2.circle(image, (px, py), 3, color.tolist(), -1)

        # Add some noise/texture
        noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        images.append(image)

    return images, cameras

if __name__ == "__main__":
    print("Multi-View Stereo Example")
    print("This implementation demonstrates:")
    print("- Plane sweep stereo algorithm")
    print("- Photo-consistency measures (NCC, SAD, Census)")
    print("- Multi-view depth map fusion")
    print("- Outlier removal and post-processing")

    # Create synthetic dataset
    print("\\nGenerating synthetic MVS dataset...")
    images, cameras = create_synthetic_mvs_dataset()

    # Run MVS pipeline
    mvs = MultiViewStereo(photo_measure='ncc', patch_size=7)

    # Define scene bounds for better depth estimation
    scene_bounds = {'depth_min': 0.5, 'depth_max': 5.0}

    result = mvs.run_mvs_pipeline(
        images, cameras,
        n_depth_planes=32,
        scene_bounds=scene_bounds
    )

    print(f"\\nMVS Results:")
    print(f"Generated {len(result['points_3d'])} 3D points")

    # Post-processing
    if len(result['points_3d']) > 0:
        post_processor = MVSPostProcessing()

        # Statistical outlier removal
        valid_indices = post_processor.statistical_outlier_removal(
            result['points_3d'], k_neighbors=10, std_ratio=2.0
        )

        filtered_points = result['points_3d'][valid_indices]
        filtered_colors = result['colors'][valid_indices] if len(result['colors']) > 0 else None

        print(f"After outlier removal: {len(filtered_points)} points")

        # Visualization
        try:
            if len(result['depth_maps']) > 0:
                visualize_depth_map(result['depth_maps'][0], result['confidence_maps'][0],
                                  "View 0 Depth Map")

            if len(filtered_points) > 0:
                visualize_point_cloud(filtered_points, filtered_colors, "MVS Reconstruction")

        except Exception as e:
            print(f"Visualization not available: {e}")

    print("\\nNote: This is an educational implementation.")
    print("For production use, consider:")
    print("- OpenMVS: Open source MVS library")
    print("- COLMAP: Complete SfM+MVS pipeline")
    print("- CMVS/PMVS: Classical MVS implementation")
    print("- Commercial solutions: Agisoft, RealityCapture")