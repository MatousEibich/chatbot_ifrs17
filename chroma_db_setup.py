# Import necessary modules
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import chromadb
from langchain.vectorstores import Chroma
import os
import openai
from dotenv import load_dotenv, find_dotenv

# Load .env file to get the OpenAI API key
load_dotenv(find_dotenv())

# Set the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]


# Create an instance of TextLoader to load your markdown file
loader = PyPDFDirectoryLoader("abc/")
# loader = PyPDFLoader("IASB_IFRS17_Standard_june2020_amendments (1).pdf")
data = loader.load()

# Create an instance of RecursiveCharacterTextSplitter to split your text into chunks
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=600
)

# Split the data into chunks
chunks = r_splitter.split_documents(data)

# Create an instance of OpenAIEmbeddings to compute embeddings for your chunks
embedding = OpenAIEmbeddings()

# Connect to your ChromaDB
client = chromadb.PersistentClient(path="./db")

# Create a Chroma vector store from your chunks
vector_db = Chroma.from_documents(
    client=client,
    documents=chunks,
    embedding=embedding,
    persist_directory="abc"
)

# Persist the vector store for later use
vector_db.persist()

# Print the number of documents loaded into the vector store
print(f"Documents Loaded: {vector_db._collection.count()}")
