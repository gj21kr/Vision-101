# pca_feature_matching.py

import torch
import requests
from PIL import Image
import numpy as np
import torchvision.transforms as T
from sklearn.decomposition import PCA
import cv2

# --- 1. Setup ---

# Check for GPU and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load DINOv2 model
print("Loading DINOv2 model (dinov2_vits14)...")
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)

# --- 2. Image Loading and Preprocessing ---

url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Florence_cathedral_from_Giottos_bell_tower_2.jpg/1280px-Florence_cathedral_from_Giottos_bell_tower_2.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/32/Florence_Duomo_from_Michelangelo_hill.jpg/1280px-Florence_Duomo_from_Michelangelo_hill.jpg"

transform = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

def load_and_transform_image(url):
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
    features_dict = model.forward_features(image_tensor)
    return features_dict['x_norm_patchtokens'].squeeze(0)

print("Extracting features from both images...")
features1_raw = extract_features(dinov2, img1_tensor).cpu().numpy()
features2_raw = extract_features(dinov2, img2_tensor).cpu().numpy()

# --- 4. PCA Transformation ---

N_COMPONENTS = 3 # Reduce to 3 dimensions for visualization
print(f"\nApplying PCA to reduce feature dimensions to {N_COMPONENTS}...")

pca = PCA(n_components=N_COMPONENTS)

# Fit PCA on the features of the first image and transform both
pca.fit(features1_raw)
features1_pca = pca.transform(features1_raw)
features2_pca = pca.transform(features2_raw)

# --- 5. Visualize PCA Components ---

print("Visualizing PCA components and saving to 'pca_visualization.jpg'...")

# Reshape the PCA features into an image grid
num_patches = 16
pca_image = features1_pca.reshape(num_patches, num_patches, N_COMPONENTS)

# Normalize the components to be in the 0-255 range for visualization
pca_image_normalized = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
pca_image_uint8 = (pca_image_normalized * 255).astype(np.uint8)

# Resize to original image size for a clear view
pca_pil = Image.fromarray(pca_image_uint8).resize(img1_pil.size, Image.NEAREST)
pca_pil.save("pca_visualization.jpg")
print("- PCA visualization saved. This image shows the semantic segmentation learned by the model.")

# --- 6. Feature Matching with PCA Features ---

keypoints1_orig = [
    (670, 230),  # Top of the dome
    (670, 580),  # Base of the dome
    (850, 480),  # Top of the bell tower
]

def scale_pt(pt, orig_size, feature_size):
    return (int(pt[0] * (feature_size / orig_size[0])), int(pt[1] * (feature_size / orig_size[1])))

matches = []
print("\nFinding matches using PCA features...")

for pt1_orig in keypoints1_orig:
    pt1_scaled = scale_pt(pt1_orig, img1_pil.size, num_patches)
    idx1 = pt1_scaled[1] * num_patches + pt1_scaled[0]
    feature1 = features1_pca[idx1]
    
    # Calculate L2 distance in the PCA space
    distances = np.linalg.norm(features2_pca - feature1, axis=1)
    best_match_idx = np.argmin(distances)
    
    pt2_scaled_y, pt2_scaled_x = np.unravel_index(best_match_idx, (num_patches, num_patches))
    pt2_orig = (int(pt2_scaled_x * (img2_pil.width / num_patches)), int(pt2_scaled_y * (img2_pil.height / num_patches)))
    
    matches.append((pt1_orig, pt2_orig))
    print(f"- Match found for {pt1_orig} -> {pt2_orig}")

# --- 7. Visualization ---

print("\nVisualizing matches and saving to 'pca_feature_matching_output.jpg'...")

combined_width = img1_pil.width + img2_pil.width
combined_height = max(img1_pil.height, img2_pil.height)
combined_img = Image.new('RGB', (combined_width, combined_height))
combined_img.paste(img1_pil, (0, 0))
combined_img.paste(img2_pil, (img1_pil.width, 0))

from PIL import ImageDraw
draw = ImageDraw.Draw(combined_img)

for pt1, pt2 in matches:
    pt2_adj = (pt2[0] + img1_pil.width, pt2[1])
    draw.line([pt1, pt2_adj], fill="cyan", width=3)
    draw.ellipse((pt1[0]-5, pt1[1]-5, pt1[0]+5, pt1[1]+5), fill="lime")
    draw.ellipse((pt2_adj[0]-5, pt2_adj[1]-5, pt2_adj[0]+5, pt2_adj[1]+5), fill="lime")

combined_img.save("pca_feature_matching_output.jpg")
print("\nPCA feature matching example complete.")
