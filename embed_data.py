import time
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

start_time = time.time()
print("📥 Loading raw documents from data.txt...")
loader = TextLoader("data.txt")
raw_docs = loader.load()
print(f"✅ Loaded {len(raw_docs)} document(s).")

print("🧩 Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = splitter.split_documents(raw_docs)
print(f"✅ Split into {len(docs)} chunks.")

print("🧠 Initializing HuggingFace Embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("✅ Embedding model loaded.")

print("📦 Saving to Milvus vector store...")
Milvus.from_documents(
    docs,
    embedding=embeddings,
    collection_name="langchain_docs",
    connection_args={"host": "192.168.130.9", "port": "19530"},
    drop_old=True
)
print("✅ Data embedded and saved to Milvus.")

end_time = time.time()
print(f"⏱ Done in {end_time - start_time:.2f} seconds.")
