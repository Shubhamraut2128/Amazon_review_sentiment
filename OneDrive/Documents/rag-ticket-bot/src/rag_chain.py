from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

from src.vector_store import load_vector_store

load_dotenv()


def get_rag_chain():

    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

    prompt = ChatPromptTemplate.from_template(
        """
You are a customer support assistant.

Use the following past ticket solutions to answer the user query.

Context:
{context}

User Ticket:
{question}

Give clear resolution suggestion.
If no relevant ticket found say: Escalate to human agent.
"""
    )

    parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | parser
    )

    return chain