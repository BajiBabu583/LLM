import os

from google import genai
from dotenv import load_dotenv

load_dotenv()

api_key=os.getenv("API")
Model=os.getenv("MODEL1")

gemini = genai.Client(api_key=api_key)

response = gemini.models.generate_content(
    model=Model,
    contents="I want to know the history of India"
)

print(response.text)