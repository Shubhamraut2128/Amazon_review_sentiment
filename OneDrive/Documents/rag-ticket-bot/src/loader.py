import pandas as pd
from langchain_core.documents import Document

def load_tickets():
    df = pd.read_csv("data/tickets.csv")

    docs = []

    for _, row in df.iterrows():
        content = f"Issue: {row['issue']}\nSolution: {row['solution']}"
        docs.append(Document(page_content=content))

    return docs