# image_captioning_example.py

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Check if a GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pre-trained processor and model
# The processor prepares the image for the model, and the model generates the caption.
print("Loading BLIP model for Image Captioning...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(device)

# URL of the sample image
img_url = 'http://images.cocodataset.org/val2017/000000039769.jpg' 
print(f"Loading image from: {img_url}")

# Download the image and open it
try:
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

print("Image loaded successfully. Generating caption...")

# --- Conditional Captioning ---
# You can provide text as a condition to guide the caption generation.
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to(device)
out = model.generate(**inputs)
caption_conditional = processor.decode(out[0], skip_special_tokens=True)
print(f"\nConditional Caption: {caption_conditional}")

# --- Unconditional Captioning ---
# The model can also generate a caption without any text prompt.
inputs = processor(raw_image, return_tensors="pt").to(device)
out = model.generate(**inputs)
caption_unconditional = processor.decode(out[0], skip_special_tokens=True)
print(f"Unconditional Caption: {caption_unconditional}")

print("\nCaptioning complete.")
