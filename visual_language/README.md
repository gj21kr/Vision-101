# Vision-Language Models Examples

This directory contains example code for various vision-language models, which are designed to understand and process information from both images and text.

## Examples

1.  **`image_captioning_example.py`**: 
    This script demonstrates how to use a pre-trained model (BLIP) to generate a descriptive caption for a given image. Image captioning is a fundamental task that bridges computer vision and natural language processing.

2.  **`vqa_example.py`**: 
    This script shows how to perform Visual Question Answering (VQA). It uses a pre-trained model (BLIP-VQA) to answer a natural language question about the contents of an image.

3.  **`clip_zero_shot_example.py`**: 
    This script showcases the power of CLIP (Contrastive Language-Image Pre-training) for zero-shot image classification. It classifies an image among a set of arbitrary text labels without being explicitly trained on those categories.

4.  **`grounding_dino_example.py`**: 
    This script provides an example of language-guided object detection using Grounding DINO. It uses a free-text prompt (e.g., "cat . couch") to detect specific objects in an image and saves the annotated result.

5.  **`sam_with_clip_example.py`**: 
    This script demonstrates how to combine two powerful models: CLIP and the Segment Anything Model (SAM). It uses CLIP to find the most relevant area in an image for a given text prompt, and then uses that location as a point prompt for SAM to generate a precise segmentation mask.

## Setup

To run these examples, you need to install the required Python libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```

**Note on `segment-anything` installation:**

The `sam_with_clip_example.py` script depends on Meta's Segment Anything Model. If you encounter issues installing it via `requirements.txt`, you may need to install it directly from the source:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```