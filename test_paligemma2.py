from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import torch
from PIL import Image
import requests

# Load the model and processor
model_id = "google/paligemma2-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
processor = PaliGemmaProcessor.from_pretrained(model_id)

# Load an image from a URL
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTWzViCqnqdmqUhe2lFzCHFjdFPlBPp5lYRCQ&s"
image = Image.open(requests.get(image_url, stream=True).raw)

# Define the prompt for image captioning
prompt = "answer en {is this pig in distress}"

# Process the image and prompt
inputs = processor(images=image, text=prompt, return_tensors="pt")

# Generate the output
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=50)

# Decode the generated output
output_text = processor.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
