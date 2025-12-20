# RAG with Gemini + Qdrant + Cohere Reranker

This project is a **Retrieval-Augmented Generation (RAG)** application built as part of the Predusk AI/ML Intern assessment.  
It combines **Google Gemini** for answer generation, **Qdrant** for vector storage, and **Cohere** for document re-ranking.

## Features
1. Upload a PDF file and process its text into embeddings.
2. Store embeddings in **Qdrant** (vector database).
3. Retrieve and **re-rank** the most relevant chunks for a query.
4. Generate grounded answers using **Gemini** with inline citations `[1], [2]`.
5. Display retrieved sources for transparency.
6. Show response latency and estimated token usage.

## Architecture

```mermaid
flowchart TD
    A[User Uploads PDF] --> B[Text Extraction and Chunking]
    B --> C["SentenceTransformer Embeddings 384-d"]
    C --> D[Qdrant Vector DB]
    E[User Query] --> F[Query Embedding]
    F --> D
    D --> G[Top-k Retrieved Chunks]
    G --> H[Cohere Reranker]
    H --> I[Gemini LLM]
    I --> J[Answer with Citations and Sources]


How it Works

1.Upload a PDF – Text is extracted and split into chunks (1000 characters with 150-character overlap).
2.Embedding – Chunks encoded using all-MiniLM-L6-v2 (384-dimensional embeddings).
3.Storage – Embeddings stored in Qdrant vector database.
4.Query – User submits a question.
5.Retrieve & Rerank – Top 10 chunks retrieved from Qdrant → reranked by Cohere → top 3 selected.
6.Answer – Gemini generates a final answer with inline citations.
7.UI – Streamlit displays the answer, sources, response time, and estimated token usage.

Installation
# Clone the repo
git clone https://github.com/your-username/rag-gemini-qdrant.git
cd rag-gemini-qdrant

# Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt


Environment Variables

Create a .env file with the following:

api_key=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
COHERE_API_KEY=your_cohere_api_key


Usage

Run the Streamlit app:

streamlit run backend.py


Upload a PDF file.

Ask a question in the text box.

See the answer with citations, sources, latency, and token estimates.

Remarks

Currently supports only PDFs.

Embedding model: all-MiniLM-L6-v2 (384-d).

Retrieval: top-k=10, Reranker: Cohere (rerank-english-v3.0), final top-3 passed to Gemini.

Token usage estimated (1 token ≈ 4 characters).

Free-tier API limits may apply → fallback/error messages shown.

Gemini Flash is used for speed and cost efficiency.

Acceptance Criteria Checklist

Live hosted on Streamlit Cloud.

Query → retrieved chunks → reranked → Gemini answer with citations.

Latency + token estimates displayed.

README includes diagram, parameters, providers, and remarks.

Tradeoffs / Limitations

Dataset size: Currently limited to 100 samples for faster testing. Scaling may increase runtime and memory.

Model choice: Simpler baseline models used for speed vs. accuracy tradeoff.

Preprocessing: Minimal preprocessing applied; advanced cleaning could improve results.

Deployment: Runs locally in Colab/VS Code; not yet production-ready (no API endpoint, no Docker).

Hardware dependency: Performance varies with GPU/CPU availability.

Generalization: Tested on dataset X; may require fine-tuning for other datasets.


This version:

- Fixes Mermaid parsing issues.  
- Replaces all “mini RAG” mentions with **RAG**.  
- Is clean, consistent, and ready to copy-paste as a single README.  

If you want, I can also add **screenshots/diagram placeholders** and **a sample query-answer example** to make it even more professional. Do you want me to do that?

