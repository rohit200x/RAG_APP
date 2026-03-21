import phoenix as px
from phoenix.otel import register
from opentelemetry import trace
import uuid, time, os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import cohere
from groq import Groq

from phoenix.evals import run_evals, HallucinationEvaluator, QAEvaluator, RelevanceEvaluator
from phoenix.evals.utils import to_annotation_dataframe
from phoenix.session.evaluation import get_qa_with_reference

load_dotenv()
os.environ["PHOENIX_PORT"] = "6006"


# ---------------- OBSERVABILITY ----------------
class ObservabilityManage:
    def __init__(self):
        register(project_name="rag-advanced", set_global_tracer_provider=False)
        self.session = px.launch_app()
        print("Phoenix UI:", self.session.url)
        self.tracer = trace.get_tracer(__name__)

    def get_tracer(self):
        return self.tracer


@st.cache_resource
def init_obs():
    return ObservabilityManage()

obs = init_obs()
tracer = obs.get_tracer()


# ---------------- DECORATORS ----------------
def trace_span(name):
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            with self.tracer.start_as_current_span(name) as span:
                start = time.time()
                result = func(self, *args, **kwargs)
                span.set_attribute("latency_ms", (time.time() - start) * 1000)
                return result
        return wrapper
    return decorator


# ---------------- CORE ----------------
class PDFPRocessor:
    def __init__(self, tracer):
        self.tracer = tracer

    @trace_span("pdf_read")
    def read_pdf(self, path):
        return PyMuPDFLoader(path).load()

    @trace_span("chunking")
    def chunk_doc(self, docs):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return [c for d in docs for c in splitter.split_text(d.page_content)]


class VectorStore:
    def __init__(self, qdrant, embedder, tracer):
        self.qdrant = qdrant
        self.embedder = embedder
        self.tracer = tracer

    @trace_span("embedding_store")
    def store(self, chunks):
        points = []
        for chunk in chunks:
            vec = self.embedder.encode(chunk).tolist()
            points.append(models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"text": chunk}
            ))
        self.qdrant.upsert(collection_name="askan", points=points)


class Retriver:
    def __init__(self, embedder, qdrant, tracer):
        self.embedder = embedder
        self.qdrant = qdrant
        self.tracer = tracer

    @trace_span("retrieval")
    def retrieve(self, query, top_k):
        vec = self.embedder.encode(query).tolist()
        result = self.qdrant.query_points(
            collection_name="askan",
            query=vec,
            limit=top_k
        )
        return [r.payload["text"] for r in result.points]


class Reranker:
    def __init__(self, co, tracer):
        self.co = co
        self.tracer = tracer

    @trace_span("rerank")
    def rerank(self, query, docs):
        result = self.co.rerank(query=query, documents=docs, top_n=3)
        return [docs[r.index] for r in result.results]


class LLM:
    def __init__(self, key, tracer):
        self.client = Groq(api_key=key)
        self.tracer = tracer

    @trace_span("llm")
    def generate(self, prompt):
        resp = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return resp.choices[0].message.content


class GroqEvalModel:
    def __init__(self, key):
        self.client = Groq(api_key=key)

    def __call__(self, prompt, **kwargs):
        resp = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return resp.choices[0].message.content


# ---------------- PIPELINE ----------------
class RAGPipeline:
    def __init__(self, retriever, reranker, llm, tracer):
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.tracer = tracer

    def run(self, query, top_k):
        docs = self.retriever.retrieve(query, top_k)
        docs = [d for d in docs if len(d.strip()) > 30]

        reranked = self.reranker.rerank(query, docs)
        context = "\n\n---\n\n".join(reranked)

        # 🔥 MAIN TRACE (Phoenix expects this structure)
        with self.tracer.start_as_current_span("rag_pipeline") as span:
            prompt = f"""
Use ONLY the context.
If not found, say "Not found in document".

Context:
{context}

Question:
{query}

Answer:
"""
            answer = self.llm.generate(prompt)

            # 🔥 REQUIRED FOR PHOENIX
            span.set_attribute("input.value", query)
            span.set_attribute("output.value", answer)
            span.set_attribute("retrieval.documents", str(docs))

        return {"answer": answer, "context": context}


# ---------------- INIT ----------------
qdrant = QdrantClient(
    url=os.getenv("Url"),
    api_key=os.getenv("QDRANT_API_KEY")
)

if not qdrant.collection_exists("askan"):
    qdrant.create_collection(
        collection_name="askan",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

embedder = SentenceTransformer("all-MiniLM-L6-v2")

pdf = PDFPRocessor(tracer)
vs = VectorStore(qdrant, embedder, tracer)
retriever = Retriver(embedder, qdrant, tracer)
reranker = Reranker(cohere.Client(os.getenv("COHERE_API_KEY")), tracer)
llm = LLM(os.getenv("GROQ_API_KEY"), tracer)

rag = RAGPipeline(retriever, reranker, llm, tracer)


# ---------------- UI ----------------
st.title("📊 RAG + Phoenix Observability")

file = st.file_uploader("Upload PDF")

if file:
    if qdrant.collection_exists("askan"):
        qdrant.delete_collection("askan")
    qdrant.create_collection(
        collection_name="askan",
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
    )

    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    docs = pdf.read_pdf("temp.pdf")
    chunks = pdf.chunk_doc(docs)
    vs.store(chunks)

    st.success(f"Indexed {len(chunks)} chunks")

top_k = st.slider("top_k", 1, 10, 3)
query = st.text_input("Ask")

if st.button("Run") and query:
    res = rag.run(query, top_k)

    st.write("Answer:", res["answer"])
    st.write("Context:", res["context"])

    # 🔥 WAIT FOR TRACE INGESTION
    time.sleep(2)

    px_client = px.Client()
    df = get_qa_with_reference(px_client)

    if df is not None and len(df) > 0:
        eval_model = GroqEvalModel(os.getenv("GROQ_API_KEY"))

        results = run_evals(
            dataframe=df,
            evaluators=[
                HallucinationEvaluator(model=eval_model),
                QAEvaluator(model=eval_model),
                RelevanceEvaluator(model=eval_model)
            ]
        )

        px_client.spans.log_span_annotations_dataframe(
            dataframe=to_annotation_dataframe(results),
            annotator_kind="LLM"
        )

        st.write(results.tail(1))
    else:
        st.warning("Run again to see evals")