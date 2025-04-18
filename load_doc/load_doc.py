import os
import shutil

# --- Enable local inference for Unstructured ---
os.environ["UNSTRUCTURED_LOCAL_INFERENCE_ENABLED"] = "true"
print("🔍 UNSTRUCTURED_LOCAL_INFERENCE_ENABLED =", os.environ.get("UNSTRUCTURED_LOCAL_INFERENCE_ENABLED"))

# --- Imports ---
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredPDFLoader,
    UnstructuredMarkdownLoader,
    PDFPlumberLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus

# --- Configuration ---
DOCS_PATH = "./docs"
LOADED_DIR = os.path.join(DOCS_PATH, "loaded")
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "langchain_docs"

os.makedirs(LOADED_DIR, exist_ok=True)

# --- Custom loader selector ---
def get_loader_for_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return TextLoader(path)
    elif ext == ".md":
        return UnstructuredMarkdownLoader(path)
    elif ext == ".pdf":
        return PDFPlumberLoader(path)
    else:
        return None

# --- Load and chunk documents ---
def load_and_chunk_documents(folder_path, vector_db):
    file_chunk_map = {}
    for root, _, files in os.walk(folder_path):
        for file in files:
            if root == LOADED_DIR:
                continue  # Skip already loaded files
            full_path = os.path.join(root, file)
            filename = os.path.basename(full_path)
            source_tag = f"load_data_{filename}"

            # --- Check for duplicates ---
            try:
                existing = vector_db.similarity_search("test", k=1, filter={"source": source_tag})
                if existing:
                    print(f"🚫 Skipping already uploaded file: {filename}")
                    continue
            except Exception as e:
                print(f"⚠️ Error checking for duplicates in Milvus: {e}")

            loader = get_loader_for_file(full_path)
            if loader:
                print(f"📄 Loading: {file} using {loader.__class__.__name__}")
                try:
                    raw_docs = loader.load()
                    for doc in raw_docs:
                        doc.metadata["source"] = filename
                    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
                    chunks = splitter.split_documents(raw_docs)
                    file_chunk_map[full_path] = chunks
                except Exception as e:
                    print(f"❌ Failed to load {file}: {e}")
            else:
                print(f"⚠️ Skipping unsupported file: {file}")
    return file_chunk_map

# --- Store into Milvus ---
def store_in_milvus(file_chunk_map, vector_db, embeddings):
    print("📤 Uploading to Milvus...")

    for file_path, chunks in file_chunk_map.items():
        try:
            filename = os.path.basename(file_path)
            source_tag = f"load_data_{filename}"
            for chunk in chunks:
                raw_text = chunk.page_content
                embeddings.embed_query(raw_text)  # Check embeddings
                vector_db.add_documents([Document(page_content=raw_text, metadata={"source": source_tag})])

            shutil.move(file_path, os.path.join(LOADED_DIR, filename))
            print(f"📦 Moved {filename} to {LOADED_DIR}")

        except Exception as e:
            print(f"❌ Failed to store chunks for {file_path}: {e}")
            print("⏭️ Skipping file move due to upload failure.")

# --- Main ---
if __name__ == "__main__":
    print("📥 Scanning folder and loading documents...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Milvus(
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        auto_id=True
    )

    file_chunks = load_and_chunk_documents(DOCS_PATH, vector_db)

    if file_chunks:
        store_in_milvus(file_chunks, vector_db, embeddings)
        print("✅ Done uploading all chunks.")
    else:
        print("⚠️ No new files to upload.")
