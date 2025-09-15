# clip_zero_shot_example.py

import requests
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

# Check if a GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pre-trained CLIP model and processor
# CLIP learns a shared embedding space for images and text.
print("Loading CLIP model for Zero-Shot Classification...")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# URL of the sample image (same as the other examples)
img_url = 'http://images.cocodataset.org/val2017/000000039769.jpg' 
print(f"Loading image from: {img_url}")

# Download the image and open it
try:
    image = Image.open(requests.get(img_url, stream=True).raw)
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# Define a set of candidate labels for classification.
# The model has not been trained on these specific labels.
labels = ["a photo of two cats on a couch", "a photo of a dog playing fetch", "a photo of a city skyline", "a photo of a delicious pizza"]
print(f"\nCandidate labels: {labels}")

# Preprocess the image and the text labels
# The processor prepares them for the model.
inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

# The model will output "logits" representing the similarity between the image and each text label.
with torch.no_grad():
    outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

# We use the softmax function to convert these scores into probabilities.
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

print("\n--- Classification Results ---")
# Print the probability for each label
for i, label in enumerate(labels):
    print(f"- Label: '{label}' | Probability: {probs[0][i]:.4f}")

# Find the label with the highest probability
best_label_idx = probs.argmax(-1).item()
print(f"\n=> Best match: '{labels[best_label_idx]}' (Probability: {probs[0][best_label_idx]:.4f})")

print("\nZero-shot classification complete.")
