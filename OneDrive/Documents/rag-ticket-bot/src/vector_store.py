from langchain_community.vectorstores import FAISS
from src.loader import load_tickets
from src.embeddings import get_embeddings

INDEX_PATH = "faiss_index"

def create_vector_store():
    docs = load_tickets()
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)

def load_vector_store():
    embeddings = get_embeddings()
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)