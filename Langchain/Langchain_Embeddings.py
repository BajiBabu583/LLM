import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
 
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
 
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
 
outputs = embeddings.embed_query("This is an example for embeddings")
 
print(outputs)