from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load PDF
loader = PyPDFLoader("Bhagavad-gita_As_It_Is english.pdf")
pages = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(pages)

# Use open-source embedding model
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store into Chroma DB
vectordb = Chroma.from_documents(splits, embedding=embedding, persist_directory="chroma_db")
vectordb.persist()

print("✅ Geeta embedded and stored.")

