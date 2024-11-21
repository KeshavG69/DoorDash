import pandas as pd
import numpy as np
import streamlit as st

import uuid
from typing import List

from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field
from typing import List,Dict
import cohere
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_cohere import CohereEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import json
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from langchain_together import ChatTogether
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()

llm = ChatGroq(
    temperature=0.1,
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-70b-versatile",
    streaming=True,
)
llm90 = ChatGroq(
    temperature=0.1,
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.2-90b-vision-preview",
    streaming=True,
)

llm_together = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    temperature=0.1,
    api_key=st.secrets["TOGETHER_API_KEY"],
    streaming=True,
)


co = cohere.Client(st.secrets["COHERE_API_KEY"])

embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=st.secrets["COHERE_API_KEY"],
)

embd = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def summary_llm(articles, links, headings, path, df):
    summaries = []
    template = """"
<Introduction>
You are an expert summarizer trained to distill large pieces of information into concise, precise, and retrieval-optimized summaries. Your task is to generate a structured and contextually relevant summary of the given article. This summary will be used in a vector database for similarity search to answer customer care queries.
</Introduction>

<Rules>
The summary should:

Highlight the main topic or purpose of the article in one sentence.
Identify key points, features, or solutions addressed in the article.
Provide any relevant details that distinguish the article from others in similar domains.
Avoid unnecessary jargon or excessive details—focus on actionable information.
Make sure to give more emphasise on the last question ie the question that comes in the end of the list .
</Rules>

<Document>
Article:{doc}
</Document>

"""
    summarise_chain = (
        ChatPromptTemplate.from_template(template) | llm | StrOutputParser()
    )

    for i, article in enumerate(articles):

        summaries.append(summarise_chain.invoke({"doc": article}))
    cleaned_summaries = [text.lstrip("**Summary:**\n") for text in summaries]

    doc_ids = [str(uuid.uuid4()) for _ in articles]
    summary_docs = [
        Document(page_content=s, metadata={"doc_id": doc_ids[i]})
        for i, s in enumerate(cleaned_summaries)
    ]
    articles_docs = [
        Document(
            page_content=article,
            metadata={"source": links[i], "doc_id": doc_ids[i], "heading": headings[i]},
        )
        for i, article in enumerate(articles)
    ]
    summary_documents = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in summary_docs
    ]
    articles_documents = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in articles_docs
    ]

    df["Sub Article Summaries"] = cleaned_summaries
    with open("summary.json", "w") as file:
        json.dump(summary_documents, file, indent=4)
    with open("articles.json", "w") as file:
        json.dump(articles_documents, file, indent=4)

    df.to_csv(path, index=False)


def compute_cosine_similarity(v1, v2):

    dot_product = np.dot(v1, v2)

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    similarity = dot_product / (norm_v1 * norm_v2)

    return similarity


def compute_overall_similarity(response, kb_chunks):
    """
    Compute the overall similarity (average) between the response (LLM's output) and KB chunks.
    This checks how much the response is grounded in the KB content.
    """

    response_embedding = embeddings.embed_query(response)

    similarities = []
    for chunk in kb_chunks:
        chunk_embedding = embeddings.embed_query(chunk.page_content)

        similarity = compute_cosine_similarity(response_embedding, chunk_embedding)
        similarities.append(similarity)

    overall_similarity = np.mean(similarities)

    return overall_similarity
