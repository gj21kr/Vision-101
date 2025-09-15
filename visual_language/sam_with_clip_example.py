# sam_with_clip_example.py

import requests
from PIL import Image
import torch
import numpy as np
import supervision as sv
import os

# --- 1. Load Models ---

# Check for GPU and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP
from transformers import CLIPProcessor, CLIPModel

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load SAM
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

print("Loading Segment Anything Model (SAM)...")
SAM_ENCODER_VERSION = "vit_b"
SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_b_01ec64.pth")

# Download the SAM checkpoint if it doesn't exist
if not os.path.exists(SAM_CHECKPOINT_PATH):
    os.makedirs(os.path.dirname(SAM_CHECKPOINT_PATH), exist_ok=True)
    checkpoint_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    print(f"Downloading SAM checkpoint from {checkpoint_url}...")
    response = requests.get(checkpoint_url)
    response.raise_for_status()
    with open(SAM_CHECKPOINT_PATH, 'wb') as f:
        f.write(response.content)
    print("Download complete.")

sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device)
sam_predictor = SamPredictor(sam)

# --- 2. Load Image and Define Prompt ---

img_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
output_path = "sam_clip_output.jpg"

print(f"Loading image from: {img_url}")
try:
    image_pil = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    image_np = np.array(image_pil)
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Text prompt for segmentation
text_prompt = "the two cats"

# --- 3. Use CLIP to Find the Target Point ---

print(f"Using CLIP to find the region for prompt: '{text_prompt}'")

# Preprocess the image and text for CLIP
inputs = clip_processor(text=[text_prompt], images=image_pil, return_tensors="pt").to(device)

# Get the features from CLIP
with torch.no_grad():
    outputs = clip_model(**inputs, output_hidden_states=True)

# The vision model output has the patch embeddings
# We find the patch that is most similar to the text embedding
vision_outputs = outputs.vision_model_output
patch_embeddings = vision_outputs.hidden_states[-1][:, 1:, :] # Removed unnecessary escaping
text_embedding = outputs.text_model_output.pooler_output

# Normalize embeddings
patch_embeddings /= patch_embeddings.norm(dim=-1, keepdim=True)
text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

# Calculate similarity
similarity = (patch_embeddings @ text_embedding.T).squeeze(0).cpu().numpy()

# Find the most similar patch
most_similar_patch_idx = np.argmax(similarity)

# The patches are in a flattened grid. We need to find the 2D coordinate.
# For ViT-B/32, the grid size is (image_size / patch_size)^2 = (224/32)^2 = 7x7
grid_size = int(np.sqrt(patch_embeddings.shape[1]))

y, x = np.unravel_index(most_similar_patch_idx, (grid_size, grid_size))

# Scale the coordinate to the original image size
original_h, original_w, _ = image_np.shape
input_point = np.array([
    (x + 0.5) * (original_w / grid_size),
    (y + 0.5) * (original_h / grid_size)
])

print(f"Most relevant point found at coordinates: {input_point.astype(int)}")

# --- 4. Use SAM to Generate Mask ---

print("Using SAM to generate mask from the point prompt...")

# Set the image for SAM predictor
sam_predictor.set_image(image_np)

# Use the point found by CLIP as a prompt for SAM
input_label = np.array([1]) # 1 indicates a foreground point

masks, scores, logits = sam_predictor.predict(
    point_coords=input_point[None, :], # SAM expects a batch of points
    point_labels=input_label[None, :],
    multimask_output=True,
)

# The model returns multiple masks; we choose the one with the highest score.
best_mask_idx = np.argmax(scores)
final_mask = masks[best_mask_idx]

print(f"Mask generated with score: {scores[best_mask_idx]:.4f}")

# --- 5. Annotation ---

print(f"Annotating image and saving to '{output_path}'...")

# Use supervision to overlay the mask on the image
annotator = sv.MaskAnnotator(color=sv.Color.red())
detections = sv.Detections(
    xyxy=sv.mask_to_xyxy(masks=np.array([final_mask]))
)
annotated_image = annotator.annotate(scene=image_np.copy(), detections=detections)

# Save the annotated image
sv.plot_image(annotated_image, (16, 16), save_path=output_path)

print("\nSAM with CLIP example complete.")
print(f"Check the output image at: {output_path}")
