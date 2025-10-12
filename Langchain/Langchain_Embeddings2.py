from langchain_google_genai import GoogleGenerativeAIEmbeddings

llm=GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

Query=llm.embed_query("India is the 4th largest Economic country in the world")

print(Query)