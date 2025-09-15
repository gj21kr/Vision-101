# feature_matching.py

import torch
import requests
from PIL import Image
import numpy as np
import torchvision.transforms as T
import cv2

# --- 1. Setup ---

# Check for GPU and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load DINOv2 model
# Using the smallest model for efficiency
print("Loading DINOv2 model (dinov2_vits14)...")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)

# --- 2. Image Loading and Preprocessing ---

# Image URLs
url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Florence_cathedral_from_Giottos_bell_tower_2.jpg/1280px-Florence_cathedral_from_Giottos_bell_tower_2.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Florence_Duomo_from_Michelangelo_hill.jpg/1280px-Florence_Duomo_from_Michelangelo_hill.jpg"

# Standard transformation for DINOv2
transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def load_and_transform_image(url):
    """Downloads, opens, and transforms an image from a URL."""
    try:
        image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        image_tensor = transform(image_pil).unsqueeze(0).to(device)
        return image_pil, image_tensor
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None, None

print("Loading and processing images...")
img1_pil, img1_tensor = load_and_transform_image(url1)
img2_pil, img2_tensor = load_and_transform_image(url2)

if img1_pil is None or img2_pil is None:
    exit()

# --- 3. Feature Extraction ---

@torch.no_grad()
def extract_features(model, image_tensor):
    """Extracts dense features from the model."""
    # The model outputs a dictionary, we are interested in the patch features
    features_dict = model.forward_features(image_tensor)
    features = features_dict['x_norm_patchtokens']
    return features.squeeze(0) # Remove batch dimension

print("Extracting features from both images...")
features1 = extract_features(dinov2, img1_tensor)
features2 = extract_features(dinov2, img2_tensor)

# DINOv2 uses a 14x14 grid of patches for a 224x224 image
num_patches = 16

# --- 4. Feature Matching ---

# Define some keypoints to match from image 1 (in pixel coordinates of the original image)
# We will scale them to the transformed image size
keypoints1_orig = [
    (670, 230),  # Top of the dome
    (670, 580),  # Base of the dome
    (850, 480),  # Top of the bell tower
]

# Scale keypoints to the feature map size (16x16 grid)
def scale_pt(pt, orig_size, feature_size):
    return (
        int(pt[0] * (feature_size / orig_size[0])),
        int(pt[1] * (feature_size / orig_size[1]))
    )

matches = []
print("\nFinding matches for keypoints...")

for pt1_orig in keypoints1_orig:
    # Scale the point
    pt1_scaled = scale_pt(pt1_orig, img1_pil.size, num_patches)
    
    # Get the feature vector for the keypoint in image 1
    # The feature map is flattened, so we convert 2D index to 1D
    idx1 = pt1_scaled[1] * num_patches + pt1_scaled[0]
    feature1 = features1[idx1]
    
    # Calculate cosine similarity with all features in image 2
    similarities = torch.nn.functional.cosine_similarity(feature1.unsqueeze(0), features2)
    
    # Find the index of the most similar patch in image 2
    best_match_idx = torch.argmax(similarities).item()
    
    # Convert the 1D index back to 2D coordinates
    pt2_scaled_y, pt2_scaled_x = np.unravel_index(best_match_idx, (num_patches, num_patches))
    
    # Scale the matched point back to the original image 2 size
    pt2_orig = (
        int(pt2_scaled_x * (img2_pil.width / num_patches)),
        int(pt2_scaled_y * (img2_pil.height / num_patches))
    )
    
    matches.append((pt1_orig, pt2_orig))
    print(f"- Match found for {pt1_orig} -> {pt2_orig}")

# --- 5. Visualization ---

print("\nVisualizing matches and saving to 'feature_matching_output.jpg'...")

# Combine the two images side-by-side
combined_width = img1_pil.width + img2_pil.width
combined_height = max(img1_pil.height, img2_pil.height)
combined_img = Image.new('RGB', (combined_width, combined_height))
combined_img.paste(img1_pil, (0, 0))
combined_img.paste(img2_pil, (img1_pil.width, 0))

draw = ImageDraw.Draw(combined_img)

# Draw lines for matches
for pt1, pt2 in matches:
    # Adjust pt2 coordinates for the combined image
    pt2_adj = (pt2[0] + img1_pil.width, pt2[1])
    draw.line([pt1, pt2_adj], fill="red", width=3)
    
    # Draw circles at keypoints
    draw.ellipse((pt1[0]-5, pt1[1]-5, pt1[0]+5, pt1[1]+5), fill="lime")
    draw.ellipse((pt2_adj[0]-5, pt2_adj[1]-5, pt2_adj[0]+5, pt2_adj[1]+5), fill="lime")

combined_img.save("feature_matching_output.jpg")
print("\nFeature matching example complete.")
