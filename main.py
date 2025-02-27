import openai
import base64
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


class Answer(BaseModel):
    # Define the answer format
    amount_of_pigs: int
    estimated_age: list[int]
    positive_behaviours: list[str]
    negative_behaviours: list[str]
    health_issues: list[str]
    posture: list[str]


# Define the question
question = """You are a welfare biologist labelling images of pigs to be used in a dataset for classifing pig welfare. 
For each image, return succinct notes on these dimensions: number of pigs, positive behaviors, negative behaviors, health issues, estimated age, and posture. Below are examples of some of those dimensions. Only turn concise labels on what you do observe in the image for each dimension.
Positive behaviors: panting, wallowing, digging, rolling around
Negative behavior: pinned-back ears, worried or anxious expression on the face, biting other pig on the flank or tail (as indicated by having their mouth open in the direction of another pig).
Health issues: open wounds, dolphin-shaped nose, bulging eyes, wrinkles to the mouth, dry “crispy” gums, glazed eyes, drooling from the mouth, bright red skin around the eyes and tongue, bloody tears.
Posture: dog-sitting position, lying down with legs stretched out, lying down without legs stretched out, feeding, standing, in motion"""

# Iterate through all image files in the images folder
image_folder = os.path.join(os.path.dirname(__file__), "images")
image_files = [f for f in os.listdir(
    image_folder) if os.path.isfile(os.path.join(image_folder, f))]

answers = []

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    with open(image_path, "rb") as img_file:
        image_base64 = base64.b64encode(img_file.read()).decode("utf-8")

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

    answer = response.choices[0].message.parsed
    answers.append(answer)

# Print all answers
for idx, answer in enumerate(answers):
    print(f"Answer for {image_files[idx]}:", answer)
