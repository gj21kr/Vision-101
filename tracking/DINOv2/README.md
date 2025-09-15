# DINOv2: Self-Supervised Learning for Visual Features

This directory contains examples demonstrating the capabilities of DINOv2, a powerful self-supervised learning model developed by Meta AI. DINOv2 excels at creating robust, general-purpose visual features directly from images without needing explicit labels. These features can be effectively used for various downstream tasks like image matching, tracking, and semantic segmentation.

## Core Concept

DINOv2 uses a Vision Transformer (ViT) backbone and learns by comparing different augmented views of the same image (self-distillation). This process teaches the model to capture high-level semantic information, making its features highly versatile.

The key output from DINOv2 for our purposes is a dense feature map for an input image. Each vector in this map corresponds to a specific patch of the image and contains rich semantic information about that patch.

--- 

## Examples

### 1. Feature Matching (`feature_matching.py`)

-   **Algorithm:** This example demonstrates how to find corresponding points between two different images. 
    1.  It loads two images and a pre-trained DINOv2 model.
    2.  It extracts a dense feature map from both images.
    3.  For a selected keypoint in the first image, it calculates the cosine similarity between that point's feature vector and all feature vectors in the second image.
    4.  The point in the second image with the highest similarity is identified as the best match.
    5.  The script visualizes these matches by drawing lines between the corresponding points.

### 2. PCA Feature Matching (`pca_feature_matching.py`)

-   **Algorithm:** This example builds on the first one by applying Principal Component Analysis (PCA) to the DINOv2 features.
    1.  After extracting the high-dimensional features (e.g., 384 dimensions for DINOv2-ViT-S/14), PCA is used to reduce them to a much lower dimension (e.g., 3).
    2.  This dimensionality reduction helps to compress the features, reduce noise, and can sometimes improve matching performance by focusing on the most significant variations.
    3.  The matching is then performed using these reduced features.
    4.  Additionally, the script visualizes the first 3 principal components of the feature map as an RGB image. This visualization reveals the rich semantic segmentation implicitly learned by the model.

### 3. Feature Tracking (`feature_tracking.py`)

-   **Algorithm:** This example shows how to track a user-selected point across frames in a video.
    1.  The user selects a point to track in the first frame of the video.
    2.  The DINOv2 feature vector for the patch at this point is stored as the "target" feature.
    3.  For each subsequent frame, the script extracts a DINOv2 feature map.
    4.  It then searches for the patch in the new frame whose feature vector has the highest cosine similarity to the target feature.
    5.  This new patch location becomes the updated position of the tracked point.
    6.  The script generates an output video showing the tracked point and its trail.

## Setup

To run these examples, you need to install the required Python libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```
