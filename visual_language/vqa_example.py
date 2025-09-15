# vqa_example.py

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch

# Check if a GPU is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the pre-trained processor and model for VQA
print("Loading BLIP model for Visual Question Answering...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# URL of the sample image (same as the captioning example)
img_url = 'http://images.cocodataset.org/val2017/000000039769.jpg' 
print(f"Loading image from: {img_url}")

# Download the image and open it
try:
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
except Exception as e:
    print(f"Error loading image: {e}")
    exit()

# The question to ask about the image
question = "How many cats are in the picture?"
print(f"Question: {question}")

# Preprocess the image and question
# The model takes both the image and the text question as input.
inputs = processor(raw_image, question, return_tensors="pt").to(device)

# Generate the answer
out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

print(f"Answer: {answer}")

# --- Example with another question ---
question_2 = "What are the cats doing?"
print(f"\nQuestion: {question_2}")

inputs_2 = processor(raw_image, question_2, return_tensors="pt").to(device)
out_2 = model.generate(**inputs_2)
answer_2 = processor.decode(out_2[0], skip_special_tokens=True)

print(f"Answer: {answer_2}")

print("\nVQA complete.")
