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


# ---------------- LOAD ENV ----------------
load_dotenv()


# ---------------- LOGGING SETUP ----------------

LOG_FILE = "rag_app.log"

logger = logging.getLogger("rag_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
    )

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# --------- DEBUG ENV (VERY IMPORTANT) ---------

qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"QDRANT_URL = {qdrant_url}")

if qdrant_key:
    logger.info(f"QDRANT_API_KEY starts with: {qdrant_key[:6]}")
else:
    logger.error("QDRANT_API_KEY NOT FOUND!")

logger.info("Application started.")


# ---------------- CLIENTS ----------------

genai.configure(api_key=os.getenv("api_key"))

logger.info("Initializing Qdrant client...")

qdrant = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_key,
    timeout=60
)

gemini = genai.GenerativeModel("models/gemini-2.5-flash")
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
co = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

collection_name = "10_feb"


# ---------------- COLLECTION CHECK ----------------

def ensure_collection():
    logger.info("Checking if collection exists...")

    if not qdrant.collection_exists(collection_name):
        logger.info("Collection not found. Creating.")

        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

        logger.info("Collection created.")


# ---------------- PDF INGESTION ----------------

def read_pdf_return_emb(pdf_path):
    try:
        logger.info(f"Reading and indexing PDF: {pdf_path}")

        qdrant.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[])
            )
        )

        logger.info("Old vectors deleted.")

        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} pages.")

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

        logger.info(f"Total chunks created: {len(points)}")

        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )

        logger.info("Embeddings stored successfully.")

    except Exception:
        logger.exception("Error while indexing PDF.")


# ---------------- RETRIEVAL ----------------

def retrieve(query, top_k=10):
    try:
        logger.info(f"Retrieving docs for query: {query}")

        query_vec = embedder.encode(query).tolist()

        results = qdrant.search(
            collection_name=collection_name,
            query_vector=query_vec,
            limit=top_k
        )

        logger.info(f"{len(results)} docs retrieved.")

        return results

    except Exception:
        logger.exception("Retrieval error.")
        return []


# ---------------- QA ----------------

def answer_query(query):
    try:
        retrieved = retrieve(query, top_k=10)

        if not retrieved:
            logger.warning("No relevant results.")
            return "No relevant results found", []

        docs = [r.payload["text"] for r in retrieved]

        logger.info("Reranking with Cohere.")

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

        logger.info("Sending prompt to Gemini.")

        start = time.time()
        response = gemini.generate_content(prompt)

        logger.info(f"LLM latency: {time.time() - start:.2f}s")

        return response.text, reranked_docs

    except Exception:
        logger.exception("Answer generation failed.")
        return "An error occurred.", []


# ---------------- STREAMLIT UI ----------------

st.title("RAG with Gemini + Qdrant")

ensure_collection()

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    logger.info(f"File uploaded: {uploaded_file.name}")

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    read_pdf_return_emb("temp.pdf")

    st.success("PDF uploaded and indexed successfully")


query = st.text_input("Ask a question")

if st.button("Submit Query") and query:
    logger.info(f"User query: {query}")

    start = time.time()
    answer, retrieved_docs = answer_query(query)
    end = time.time()

    st.subheader("Answer")
    st.write(answer)

    response_time = end - start
    tokens_est = (len(query) + len(' '.join(retrieved_docs)) + len(answer)) // 4

    logger.info(f"Response time: {response_time:.2f}s | Tokens: {tokens_est}")

    st.write(f"Response time: {response_time:.2f} seconds")
    st.write(f"Estimated tokens: {tokens_est}")
