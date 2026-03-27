import src.config
import streamlit as st
from src.vector_store import create_vector_store
from src.rag_chain import get_rag_chain

st.title("🎯 RAG Ticket Resolution Bot")

if st.button("Create Vector DB"):
    create_vector_store()
    st.success("Vector DB Created")

query = st.text_input("Enter Support Ticket")

if query:
    chain = get_rag_chain()

    # ✅ NEW WAY
    result = chain.invoke(query)

    st.subheader("✅ Suggested Solution")
    st.write(result)