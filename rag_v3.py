import streamlit as st
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.llms import HuggingFaceEndpoint
from pydantic import Field
from typing import List

# --- Env Fix ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_NO_PYTHON_EXTENSION_WARNING"] = "1"

st.set_page_config(page_title="RAG Streaming Demo", layout="wide")
st.title("🧠 Local RAG Q&A - MapReduce + Streaming + Milvus")

# --- Radius Filtered Retriever ---
class RadiusFilteredRetriever(BaseRetriever):
    retriever: BaseRetriever
    threshold: float = Field(...)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self.retriever.vectorstore.similarity_search_with_score(query, k=10)
        filtered = [(doc, score) for doc, score in results if score <= self.threshold]
        top_k = sorted(filtered, key=lambda x: x[1])[:3]
        return [doc for doc, _ in top_k]

# --- Text Cleaning ---
def clean_text(text: str) -> str:
    text = re.sub(r"\|\s*\[.*?\]", "", text)
    text = re.sub(r"[|─]+", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# --- Load Docs ---
@st.cache_resource(show_spinner=False)
def load_docs():
    loader = TextLoader("data.txt")
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    return splitter.split_documents(raw_docs)

docs = load_docs()

# --- Milvus & Embeddings ---
@st.cache_resource(show_spinner=False)
def connect_milvus():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Milvus(
        embedding_function=embeddings,
        collection_name="langchain_docs",
        connection_args={"host": "localhost", "port": "19530"}
    )
    return vector_db, embeddings

vector_db, embeddings = connect_milvus()

# --- Setup Retriever ---
dense_retriever_raw = vector_db.as_retriever(search_kwargs={"k": 10})
radius = st.slider("🔯 Set max distance (radius) for dense retrieval:", 0.0, 2.0, 1.5, 0.01)
filtered_dense = RadiusFilteredRetriever(retriever=dense_retriever_raw, threshold=radius)

bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 3

retriever = EnsembleRetriever(
    retrievers=[filtered_dense, bm25_retriever],
    weights=[0.7, 0.3]
)

# --- Prompt & LLM ---
prompt = PromptTemplate.from_template(
    """You are a helpful, friendly assistant. Answer the user's question naturally based on the context below.
If you don't know the answer from the context, say "I'm not sure based on what I know so far."

Context:
{context}

Question: {question}
Answer:"""
)

llm = HuggingFaceEndpoint(
    endpoint_url="http://localhost:8085/generate_stream",
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.02,
    streaming=True,
    stop_sequences=["<|eot_id|>"]
)

map_chain = LLMChain(llm=llm, prompt=prompt)
reduce_chain = StuffDocumentsChain(llm_chain=map_chain, document_variable_name="context")
combine_docs_chain = MapReduceDocumentsChain(
    llm_chain=map_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name="context"
)
qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=combine_docs_chain)

# --- Ask Question & Stream Answer ---
st.subheader("💬 Ask a Question")
query = st.text_input("Enter your question:")

if query:
    start = time.time()
    with st.spinner("🌀 Generating answer (streaming)..."):
        final_prompt = prompt.format(context="\n".join([doc.page_content for doc in filtered_dense.get_relevant_documents(query)]), question=query)
        response_placeholder = st.empty()
        streamed_text = ""

        for chunk in llm.stream(final_prompt):
            streamed_text += chunk  # Assume `chunk` is a string
            response_placeholder.markdown(f"**🧠 Answer:**\n\n{streamed_text}")


    st.caption(f"⏱️ Response streamed in {time.time() - start:.2f}s")

# --- Teach Mode ---
st.subheader("✍️ Teach the AI (add to knowledge base)")
new_knowledge = st.text_area("Enter new fact or information:")
if st.button("🧠 Save this new information to vector DB") and new_knowledge.strip():
    try:
        cleaned_text = clean_text(new_knowledge)
        new_embedding = embeddings.embed_query(cleaned_text)
        vector_db.col.insert([
            ["manual_prompt"],
            [cleaned_text],
            [new_embedding]
        ])
        st.success("✅ Your new knowledge was saved!")
    except Exception as e:
        st.error(f"❌ Failed to save: {e}")

# --- t-SNE Visualization ---
if st.checkbox("🧬 Show t-SNE Visualization"):
    st.subheader("🧬 Embedding Visualization (t-SNE)")
    if len(docs) >= 2 and query:
        st.write("📊 Computing embeddings for t-SNE...")
        start = time.time()
        doc_vecs = embeddings.embed_documents([doc.page_content for doc in docs])
        query_vec = embeddings.embed_query(query)

        all_embeddings = np.vstack([doc_vecs, query_vec])
        tsne = TSNE(n_components=2, perplexity=min(30, len(all_embeddings) - 1), random_state=42)
        tsne_result = tsne.fit_transform(all_embeddings)

        doc_points = tsne_result[:-1]
        query_point = tsne_result[-1]

        st.write(f"✅ t-SNE completed in {time.time() - start:.2f}s")

        fig, ax = plt.subplots()
        ax.scatter(doc_points[:, 0], doc_points[:, 1], c='blue', label='doc', s=100, alpha=0.6)
        ax.scatter(query_point[0], query_point[1], c='red', label='query', s=120, marker='x')
        ax.set_title("t-SNE of Document Embeddings + Query")
        ax.legend()
        st.pyplot(fig)
    elif not query:
        st.info("💡 Ask a question to visualize where the query appears among the documents.")
    else:
        st.warning("⚠️ You need at least 2 chunks to show t-SNE.")

