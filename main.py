import openai
import base64
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()

# Load and encode the image
image_path = os.path.join(os.path.dirname(__file__), "images/image.png")
with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

# Initialize OpenAI client
# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define the answer format


class Answer(BaseModel):
    amount_of_pigs: int
    how_cramped_is_each_pig: list[str]
    pig_happiness: list[int]
    explanation: str


# Define the question and request format
question = "Estimate the amount of pigs in the image and how cramped each pig is. Also, estimate the happiness of each pig on a scale from 1 to 10. Provide an explanation for your answer."
response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
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
    response_format=Answer,
    max_tokens=300
)

# Extract and print the answer
answer = response.choices[0].message.parsed
print("Answer:", answer)
