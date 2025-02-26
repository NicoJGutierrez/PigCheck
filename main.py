import openai
import base64
from PIL import Image
from io import BytesIO
import os

# Load and encode the image
image_path = "/home/hgrafe/Downloads/BMC-slat-floor-1.jpg"
with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# Initialize OpenAI client
# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define the question and request format
question = "question en Estimate the amount of area in square meters the pigs can evolve in, and the number of pigs. The last line must be your best estimate of the area as a float without anything else than the number, and the line before that must be an int without anything else than the number"
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ],
    max_tokens=300
)

# Extract and print the answer
answer = response.choices[0].message.content
print("Answer:", answer)
