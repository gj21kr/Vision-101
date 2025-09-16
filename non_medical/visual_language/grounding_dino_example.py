# grounding_dino_example.py

import requests
from PIL import Image, ImageDraw, ImageFont
import torch
import supervision as sv
from groundingdino.util.inference import load_model, load_image, predict, annotate
import os

# --- Configuration ---
# Check for GPU and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Path to the GroundingDINO config and weights
# The model will be downloaded automatically if not present
CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = os.path.join("weights", WEIGHTS_NAME)

# --- Model Loading ---
print("Loading GroundingDINO model...")
# This function handles downloading the model weights as well
model = load_model(CONFIG_PATH, WEIGHTS_PATH, device=device)

# --- Image and Text Prompt ---
img_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image_path = "grounding_dino_input.jpg"
output_path = "grounding_dino_output.jpg"

print(f"Loading image from: {img_url}")
try:
    # Download and save the image locally
    response = requests.get(img_url)
    response.raise_for_status()
    with open(image_path, 'wb') as f:
        f.write(response.content)
    
    # Load image for the model and for annotation
    image_source, image = load_image(image_path)

except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Define the text prompt for object detection
# Use '.' to separate different objects you want to detect.
TEXT_PROMPT = "cat . couch . remote control"

# --- Inference ---
# Set thresholds for detection
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

print(f"Detecting objects with prompt: '{TEXT_PROMPT}'")

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=device
)

print(f"\nDetected {len(boxes)} objects:")
for i in range(len(boxes)):
    print(f"- Object: '{phrases[i]}', Confidence: {logits[i]:.4f}")

# --- Annotation ---
print(f"\nAnnotating image and saving to '{output_path}'...")

# Use the supervision library to draw the bounding boxes and labels
annotator = sv.BoxAnnotator()
labels = [
    f"{phrase} {logit:.2f}"
    for phrase, logit in zip(phrases, logits)
]
annotated_image = annotator.annotate(scene=image_source.copy(), detections=sv.Detections(xyxy=boxes), labels=labels)

# Save the annotated image
sv.plot_image(annotated_image, (16, 16), save_path=output_path)

print("\nGrounding DINO example complete.")
print(f"Check the output image at: {output_path}")
