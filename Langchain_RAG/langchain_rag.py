from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from langchain.chains import RetrievalQA

# Sample Text
text = [
    "Baji is a graduate from Chebrolu Engineering College from CSE background.",
    "Vivek is an IT employee, working at Zennial Pro Private Limited. Baji, Shiva(playboy) and Vivek are colleagues.",
    "Dhanuj is an ex-TCS employee. Vivek and Dhanuj have been working in the same companies since their previous company till now."
]

# Splitting the text
text_splitter = CharacterTextSplitter(chunk_size=201)
docs = text_splitter.create_documents(text)

# Embeddings and Chroma Vector DB
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key="AIzaSyA-SsBef1O30YyaHmG6jigNjnTX2raliqA"
)
vector_store = Chroma.from_documents(docs, embeddings)

# Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key="AIzaSyA-SsBef1O30YyaHmG6jigNjnTX2raliqA"
)

# RAG Pipeline
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Test
query = "who is shiva"
ans = rag_chain.invoke({"query":query})

print(ans)
