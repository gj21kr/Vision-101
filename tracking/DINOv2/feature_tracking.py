# feature_tracking.py

import torch
import requests
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import os

# --- 1. Setup ---

# Check for GPU and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load DINOv2 model
print("Loading DINOv2 model (dinov2_vits14)...")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)

# --- 2. Video and Initial Point Setup ---

video_url = "https://videos.pexels.com/video-files/1572386/1572386-sd_640_360_30fps.mp4"
video_path = "tracking_input.mp4"
output_path = "feature_tracking_output.mp4"

# Download the video if it doesn't exist
if not os.path.exists(video_path):
    print(f"Downloading video from {video_url}...")
    response = requests.get(video_url, stream=True)
    response.raise_for_status()
    with open(video_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

# Open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Point to track in the first frame (x, y)
# This point is on the person's backpack in the sample video
point_to_track = (350, 180)
tracked_points = [point_to_track]

# --- 3. Feature Extraction and Tracking Loop ---

# DINOv2 transformation
transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

@torch.no_grad()
def extract_features_from_frame(model, frame):
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
    features_dict = model.forward_features(frame_tensor)
    return features_dict['x_norm_patchtokens'].squeeze(0)

# Extract target feature from the first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Could not read the first frame.")
    exit()

print("Extracting target feature from the first frame...")
features_first_frame = extract_features_from_frame(dinov2, first_frame)

num_patches = 16

def scale_pt(pt, orig_size, feature_size):
    return (int(pt[0] * (feature_size / orig_size[0])), int(pt[1] * (feature_size / orig_size[1])))

pt_scaled = scale_pt(point_to_track, (width, height), num_patches)
idx = pt_scaled[1] * num_patches + pt_scaled[0]
target_feature = features_first_frame[idx]

# Add first frame to output video (with the initial point)
cv2.circle(first_frame, point_to_track, 5, (0, 255, 0), -1)
out.write(first_frame)

print("Starting tracking loop...")
frame_count = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...", end='\r')

    # Extract features from the current frame
    features_current_frame = extract_features_from_frame(dinov2, frame)

    # Calculate similarity and find the best match
    similarities = torch.nn.functional.cosine_similarity(target_feature.unsqueeze(0), features_current_frame)
    best_match_idx = torch.argmax(similarities).item()

    # Convert 1D index to 2D and scale back to original frame size
    pt_scaled_y, pt_scaled_x = np.unravel_index(best_match_idx, (num_patches, num_patches))
    new_point = (int(pt_scaled_x * (width / num_patches)), int(pt_scaled_y * (height / num_patches)))
    
    tracked_points.append(new_point)

    # --- 4. Visualization ---

    # Draw the tracking trail
    for i in range(1, len(tracked_points)):
        cv2.line(frame, tracked_points[i-1], tracked_points[i], (0, 255, 255), 2)
    
    # Draw the current tracked point
    cv2.circle(frame, new_point, 5, (0, 0, 255), -1)

    # Write the frame to the output video
    out.write(frame)

# Release resources
cap.release()
out.release()

print(f"\n\nFeature tracking complete.")
print(f"Output video saved to: {output_path}")
