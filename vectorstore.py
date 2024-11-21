from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import time
from helper import embd
import json
from langchain_core.documents import Document
import streamlit as st

load_dotenv()

with open("summary.json", "r") as file:
    loaded_documents = json.load(file)
summary_docs = [
    Document(page_content=doc["page_content"], metadata=doc["metadata"])
    for doc in loaded_documents
]


pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])


index_name = "doordash"  # change if desired

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

vectorstore = PineconeVectorStore(index=index, embedding=embd)
vectorstore.add_documents(documents=summary_docs)
