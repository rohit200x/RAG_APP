import os
import logging
import uuid
import time

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.http import models

from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import cohere


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("api_key"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

gemini = genai.GenerativeModel("models/gemini-2.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

collection_name = "rag_docs"

if not qdrant.collection_exists(collection_name):
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=384,
            distance=models.Distance.COSINE
        )
    )


def read_pdf_return_emb(pdf_path):
    qdrant.delete(
        collection_name=collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[])
        )
    )

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    points = []

    for doc in docs:
        splits = text_splitter.split_text(doc.page_content)
        for chunk in splits:
            vector = embedder.encode(chunk).tolist()
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"text": chunk}
                )
            )

    qdrant.upsert(
        collection_name=collection_name,
        points=points
    )


def retrieve(query, top_k=10):
    try:
        query_vec = embedder.encode(query).tolist()

        results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vec,
            limit=top_k
        )

        return results

    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []


def answer_query(query):
    retrieved = retrieve(query, top_k=10)

    if not retrieved:
        return "No relevant results found", []

    docs = [r.payload["text"] for r in retrieved]

    rerank = co.rerank(
        query=query,
        documents=docs,
        top_n=3,
        model="rerank-english-v3.0"
    )

    reranked_docs = [docs[r.index] for r in rerank.results]
    context = "\n".join(reranked_docs)

    prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}

Answer only from the context.
"""

    response = gemini.generate_content(prompt)
    return response.text, reranked_docs


st.title("RAG with Gemini + Qdrant")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    read_pdf_return_emb("temp.pdf")
    st.success("PDF uploaded and indexed successfully")

query = st.text_input("Ask a question")

if st.button("Submit Query") and query:
    start = time.time()
    answer, retrieved_docs = answer_query(query)
    end = time.time()

    st.subheader("Answer")
    st.write(answer)
    st.write(f"Response time: {end - start:.2f} seconds")
    st.write(f"Estimated tokens: {(len(query) + len(' '.join(retrieved_docs)) + len(answer)) // 4}")
