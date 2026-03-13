# ============================================================
# app.py - Agentic Document Intelligence System v2
# Stack: LangGraph + LLaMA (Ollama) + FAISS RAG + RoBERTa
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import io
import time
import json
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypedDict, Optional

# PDF + OCR
import fitz
import pytesseract
from PIL import Image

# LangChain + LangGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langgraph.graph import StateGraph, END

# Ollama (LLaMA)
import ollama

# HuggingFace (RoBERTa)
from transformers import pipeline, logging
logging.set_verbosity_error()

import torch

# ============================================================
# FLASK SETUP
# ============================================================
app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIG
# ============================================================
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

OLLAMA_MODEL    = "llama3.2"
CHUNK_SIZE      = 10000      # bigger chunks = fewer LLM calls
CHUNK_OVERLAP   = 500
MAX_WORKERS     = 6          # parallel workers
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_CACHE_DIR = "faiss_cache"
os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

device = 0 if torch.cuda.is_available() else -1

# ============================================================
# LOAD ROBERTA (once at startup - small model)
# ============================================================
print("Loading RoBERTa QA model...")
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=device
)
print("RoBERTa loaded!")

# ============================================================
# LOAD EMBEDDINGS (once at startup)
# ============================================================
print("Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
print("Embeddings loaded!")

# ============================================================
# LANGGRAPH STATE
# ============================================================
class DocState(TypedDict):
    pdf_path:         str
    question:         str
    extracted_text:   str
    chunks:           list
    faiss_index:      object
    answer:           str
    metrics:          dict
    doc_type:         str
    query_type:       str
    retry_count:      int
    start_time:       float
    page_count:       int
    char_count:       int

# ============================================================
# PROMPTS
# ============================================================
SUMMARY_PROMPTS = {
    "academic": (
        "You are an academic summarizer. Extract: key concepts, "
        "methodology, findings, conclusions. Use bullet points. "
        "Be concise.\n\nText:\n{text}\n\nBullet point summary:"
    ),
    "legal": (
        "You are a legal document analyst. Extract: parties, "
        "key clauses, obligations, dates. Be precise.\n\n"
        "Text:\n{text}\n\nKey points:"
    ),
    "technical": (
        "You are a technical writer. Extract: key concepts, "
        "processes, specifications. Use bullet points.\n\n"
        "Text:\n{text}\n\nTechnical summary:"
    ),
    "general": (
        "Summarize the key points of this text in clear "
        "bullet points. Be concise.\n\nText:\n{text}\n\nSummary:"
    )
}

MULTIPART_PROMPT = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "List ALL items mentioned in the context that answer "
    "this question. Number each one. Do not miss any.\n\n"
    "Complete answer:"
)

REASONING_PROMPT = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Think step by step using only the context above. "
    "Provide a reasoned explanation.\n\n"
    "Answer:"
)

MERGE_PROMPT = (
    "Merge these summaries into one coherent final summary. "
    "Remove repetition. Keep all key points. "
    "Use clear bullet points.\n\n"
    "Summaries:\n{summaries}\n\n"
    "Final merged summary:"
)

# ============================================================
# UTILITY: PDF HASH (for FAISS cache)
# ============================================================
def get_pdf_hash(pdf_path: str) -> str:
    h = hashlib.md5()
    with open(pdf_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

# ============================================================
# UTILITY: FORMAT TIME
# ============================================================
def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} sec"
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins} min {secs:.2f} sec"

# ============================================================
# UTILITY: DETECT QUERY TYPE
# ============================================================
def detect_query_type(question: str) -> str:
    q = question.lower().strip()

    # Summary triggers
    summary_keywords = ["summary", "summarize", "overview",
                        "brief", "outline", "gist", "tldr"]
    if any(k in q for k in summary_keywords):
        return "FULL_SUMMARY"

    # Multi-part triggers
    multipart_keywords = ["list", "all", "every", "what are",
                          "how many", "enumerate", "types of",
                          "examples of", "mention"]
    if any(k in q for k in multipart_keywords):
        return "MULTIPART_QA"

    # Reasoning triggers
    reasoning_keywords = ["why", "how does", "explain",
                          "what would", "compare", "difference",
                          "relationship", "impact", "effect"]
    if any(k in q for k in reasoning_keywords):
        return "REASONING_QA"

    # Default: factual
    return "FACTUAL_QA"

# ============================================================
# UTILITY: DETECT DOCUMENT TYPE
# ============================================================
def detect_doc_type(text: str) -> str:
    sample = text[:1000].lower()
    if any(w in sample for w in ["abstract", "methodology",
           "conclusion", "research", "university", "lecture"]):
        return "academic"
    if any(w in sample for w in ["whereas", "clause", "agreement",
           "party", "hereby", "legal", "contract"]):
        return "legal"
    if any(w in sample for w in ["api", "function", "install",
           "configure", "technical", "specification"]):
        return "technical"
    return "general"

# ============================================================
# CORE: EXTRACT SINGLE PAGE
# ============================================================
def extract_page(args):
    page, page_num = args
    try:
        text = page.get_text().strip()
        if text:
            return page_num, text
        # OCR fallback for scanned pages
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB",
              [pix.width, pix.height], pix.samples)
        ocr_text = pytesseract.image_to_string(img).strip()
        return page_num, ocr_text
    except Exception:
        return page_num, ""

# ============================================================
# CORE: PARALLEL PDF EXTRACTION
# ============================================================
def extract_pdf_parallel(pdf_path: str):
    doc = fitz.open(pdf_path)
    page_count = len(doc)
    pages = [(doc[i], i) for i in range(page_count)]

    results = {}
    workers = min(MAX_WORKERS, page_count)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(extract_page, p): p[1] for p in pages}
        for future in as_completed(futures):
            page_num, text = future.result()
            results[page_num] = text

    doc.close()

    # Join in page order
    full_text = " ".join(
        results[i] for i in sorted(results.keys())
    )
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    return full_text, page_count

# ============================================================
# CORE: SEMANTIC CHUNKING
# ============================================================
def semantic_chunk(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

# ============================================================
# CORE: LLaMA CALL (via Ollama)
# ============================================================
def call_llama(prompt: str, num_ctx: int = 4096) -> str:
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            options={"num_ctx": num_ctx},
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"LLaMA error: {str(e)}"

# ============================================================
# CORE: TF-IDF EXTRACTIVE SUMMARY (replaces LLaMA for map phase)
# Extracts top sentences by importance — no LLM needed = instant!
# ============================================================
def extractive_summary(chunk: str, top_n: int = 5) -> str:
    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

    if not sentences:
        return chunk[:1000]
    if len(sentences) <= top_n:
        return " ".join(sentences)

    try:
        # TF-IDF score each sentence
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)
        scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        # Pick top_n sentences in original order
        top_indices = sorted(
            np.argsort(scores)[-top_n:].tolist()
        )
        return " ".join(sentences[i] for i in top_indices)
    except Exception:
        return " ".join(sentences[:top_n])

# ============================================================
# CORE: RAPTOR HIERARCHICAL SUMMARY
# MAP    = TF-IDF extractive (parallel, instant, no LLM)
# REDUCE = smart batching — LLaMA sees entire document!
#
# Strategy:
#   1. TF-IDF: extract 3 key sentences per chunk (instant, parallel)
#   2. Batch extracted text into 15000-char batches
#   3. Each batch → LLaMA → partial summary
#   4. If >1 batch: merge all partial summaries → final
#
# 363 pages = 23 chunks × 3 sentences = ~3 batches
# Best case: 2 LLaMA calls × 120sec = ~4 mins
# ============================================================
def raptor_summarize(chunks: list, doc_type: str):

    if not chunks:
        return "No content to summarize.", 0, 0

    # ---- MAP PHASE: TF-IDF extractive — truly parallel, no LLM ----
    map_start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(extractive_summary, chunk, 3)
                   for chunk in chunks]
        mini_summaries = [f.result() for f in futures
                          if f.result().strip()]

    map_time = time.time() - map_start
    print(f"[RAPTOR] Map (TF-IDF): {len(mini_summaries)} chunks in {map_time:.1f}s")

    if not mini_summaries:
        return "Could not generate summary.", map_time, 0

    # ---- REDUCE PHASE: smart batching ----
    reduce_start = time.time()

    # Split into 15000-char batches (3 sentences/chunk × 23 chunks = ~3 batches)
    BATCH_CHARS = 2500
    batches = []
    current_batch = ""

    for summary in mini_summaries:
        if len(current_batch) + len(summary) > BATCH_CHARS and current_batch:
            batches.append(current_batch.strip())
            current_batch = summary + "\n\n"
        else:
            current_batch += summary + "\n\n"

    if current_batch.strip():
        batches.append(current_batch.strip())

    print(f"[RAPTOR] Reduce: {len(batches)} batches → LLaMA")

    # Each batch → LLaMA partial summary
    partial_summaries = []
    for i, batch in enumerate(batches):
        prompt = (
            f"Extract the key points from this text section "
            f"in clear bullet points. Be concise.\n\n"
            f"Text:\n{batch}\n\nKey points:"
        )
        partial = call_llama(prompt)
        partial_summaries.append(partial)
        print(f"[RAPTOR] Batch {i+1}/{len(batches)} done")

    # Final merge — combine all partial summaries
    if len(partial_summaries) == 1:
        final_summary = partial_summaries[0]
    else:
        merged = "\n\n---\n\n".join(partial_summaries)
        final_prompt = (
            "Merge these section summaries into one comprehensive "
            "final summary. Remove repetition. Keep ALL key points. "
            "Use clear organized bullet points.\n\n"
            f"Section summaries:\n{merged[:6000]}\n\n"
            "Final comprehensive summary:"
        )
        final_summary = call_llama(final_prompt)

    reduce_time = time.time() - reduce_start
    total_calls = len(batches) + (1 if len(partial_summaries) > 1 else 0)
    print(f"[RAPTOR] Reduce done: {reduce_time:.1f}s | {total_calls} LLaMA calls | Total: {map_time+reduce_time:.1f}s")

    return final_summary, map_time, reduce_time

# ============================================================
# CORE: BUILD FAISS INDEX
# ============================================================
def build_faiss_index(chunks: list, pdf_hash: str):
    cache_path = os.path.join(FAISS_CACHE_DIR, pdf_hash)

    # Try loading from cache first
    if os.path.exists(cache_path):
        try:
            return FAISS.load_local(
                cache_path,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        except Exception:
            pass

    # Build new index
    docs = [Document(page_content=c,
            metadata={"chunk_id": i})
            for i, c in enumerate(chunks)]

    index = FAISS.from_documents(docs, embedding_model)

    # Save to cache
    try:
        index.save_local(cache_path)
    except Exception:
        pass

    return index

# ============================================================
# CORE: COMPUTE CONFIDENCE
# ============================================================
def compute_confidence(question: str, docs: list) -> float:
    try:
        q_embed = embedding_model.embed_query(question)
        sims = []
        for d in docs:
            d_embed = embedding_model.embed_query(
                d.page_content[:300])
            sim = np.dot(q_embed, d_embed) / (
                np.linalg.norm(q_embed) *
                np.linalg.norm(d_embed) + 1e-8
            )
            sims.append(sim)
        return round(float(np.mean(sims)) * 100, 2)
    except Exception:
        return 0.0

# ============================================================
# CORE: ROBERTA QA
# ============================================================
def roberta_qa(question: str, chunks: list):
    best_answer = ""
    best_score = 0.0

    for chunk in chunks:
        try:
            result = qa_pipeline(
                question=question,
                context=chunk[:2000]
            )
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result['answer']
        except Exception:
            continue

    return best_answer, round(best_score, 4)

# ============================================================
# LANGGRAPH NODES
# ============================================================

# Node 1: Extract PDF
def node_extract(state: DocState) -> DocState:
    extract_start = time.time()
    text, page_count = extract_pdf_parallel(state["pdf_path"])
    extract_time = time.time() - extract_start

    state["extracted_text"] = text
    state["page_count"] = page_count
    state["char_count"] = len(text)
    state["metrics"]["extraction_time_sec"] = round(
        extract_time, 2)
    state["metrics"]["pages_processed"] = page_count
    state["metrics"]["characters_processed"] = len(text)
    state["metrics"]["words_processed"] = len(text.split())
    return state

# Node 2: Detect type + chunk
def node_chunk(state: DocState) -> DocState:
    text = state["extracted_text"]
    state["doc_type"] = detect_doc_type(text)
    state["query_type"] = detect_query_type(state["question"])
    state["chunks"] = semantic_chunk(text)
    state["metrics"]["chunks_created"] = len(state["chunks"])
    state["metrics"]["doc_type"] = state["doc_type"]
    state["metrics"]["query_type"] = state["query_type"]
    return state

# Node 3: Summary path
def node_summarize(state: DocState) -> DocState:
    summary_start = time.time()
    chunks = state["chunks"]
    doc_type = state["doc_type"]

    summary, map_time, reduce_time = raptor_summarize(chunks, doc_type)
    summary_time = time.time() - summary_start

    state["answer"] = summary
    state["metrics"]["summary_time_sec"] = round(summary_time, 2)
    state["metrics"]["summary_length_words"] = len(summary.split())
    state["metrics"]["parallel_workers"] = min(MAX_WORKERS, len(chunks))
    state["metrics"]["map_time_sec"] = round(map_time, 2)
    state["metrics"]["reduce_time_sec"] = round(reduce_time, 2)
    state["metrics"]["type"] = "summary"
    return state

# Node 4: QA path
def node_qa(state: DocState) -> DocState:
    qa_start = time.time()
    question = state["question"]
    query_type = state["query_type"]
    chunks = state["chunks"]

    # Build or load FAISS index
    pdf_hash = get_pdf_hash(state["pdf_path"])
    faiss_index = build_faiss_index(chunks, pdf_hash)

    # Retrieve relevant chunks
    k = 8 if query_type in ["MULTIPART_QA",
                             "REASONING_QA"] else 5
    retrieved = faiss_index.similarity_search(question, k=k)
    retrieved_texts = [d.page_content for d in retrieved]

    # Confidence score
    confidence = compute_confidence(question, retrieved)

    answer = ""
    model_used = ""

    if query_type == "FACTUAL_QA":
        # Try RoBERTa first
        rob_answer, rob_score = roberta_qa(
            question, retrieved_texts)

        if rob_score > 0.3 and rob_answer.strip():
            answer = rob_answer
            model_used = "roberta"
            confidence = round(rob_score * 100, 2)
        else:
            # Fallback to LLaMA
            context = "\n\n".join(retrieved_texts)
            prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Give a short exact answer from "
                f"the context only:\n"
            )
            answer = call_llama(prompt)
            model_used = "llama_fallback"

    elif query_type == "MULTIPART_QA":
        context = "\n\n".join(retrieved_texts)
        prompt = MULTIPART_PROMPT.format(
            context=context,
            question=question
        )
        answer = call_llama(prompt)
        model_used = "llama"

    elif query_type == "REASONING_QA":
        context = "\n\n".join(retrieved_texts)
        prompt = REASONING_PROMPT.format(
            context=context,
            question=question
        )
        answer = call_llama(prompt)
        model_used = "llama"

    else:
        # Default factual
        rob_answer, rob_score = roberta_qa(
            question, retrieved_texts)
        answer = rob_answer if rob_answer else "Not found."
        model_used = "roberta"

    qa_time = time.time() - qa_start

    if not answer.strip():
        answer = "Could not find a relevant answer in the PDF."

    state["answer"] = answer
    state["metrics"]["qa_time_sec"] = round(qa_time, 2)
    state["metrics"]["confidence_score"] = confidence
    state["metrics"]["model_used"] = model_used
    state["metrics"]["chunks_retrieved"] = k
    state["metrics"]["type"] = "qa"
    return state

# Node 5: Validate output
def node_validate(state: DocState) -> DocState:
    answer = state["answer"]
    retry = state.get("retry_count", 0)

    # Basic quality checks
    if len(answer.strip()) < 10 and retry < 2:
        state["retry_count"] = retry + 1
        state["answer"] = ""
        return state

    # ---- FINAL TIMING ----
    total_time = time.time() - state["start_time"]

    # ---- TOKENS PER SECOND ----
    output_words  = len(answer.split())
    output_tokens = output_words * 1.3
    tps = round(output_tokens / total_time, 2) if total_time > 0 else 0

    m = state["metrics"]

    # ============================================================
    # V1 METRICS (old - kept for benchmarking comparison)
    # These match exactly what v1 app.py returned
    # ============================================================
    m["response_time_sec"]    = round(total_time, 2)
    m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
    m["pages_processed"]      = state.get("page_count", 0)
    m["characters_processed"] = state.get("char_count", 0)
    m["words_processed"]      = len(state.get("extracted_text", "").split())

    # v1 summary specific
    if m.get("type") == "summary":
        m["summary_time_sec"]      = m.get("summary_time_sec", 0)
        m["summary_length_words"]  = len(answer.split())

    # v1 QA specific
    if m.get("type") == "qa":
        m["qa_time_sec"]       = m.get("qa_time_sec", 0)
        m["confidence_score"]  = m.get("confidence_score", 0)

    # ============================================================
    # V2 METRICS (new - industry standard)
    # ============================================================
    m["ttft_sec"]          = round(total_time, 2)   # time to first token
    m["tps"]               = tps                     # tokens per second
    m["doc_type"]          = state.get("doc_type", "general")
    m["query_type"]        = state.get("query_type", "")
    m["chunks_created"]    = m.get("chunks_created", 0)
    m["retry_count"]       = retry
    m["model_used"]        = m.get("model_used", "roberta")

    # v2 summary specific
    if m.get("type") == "summary":
        m["parallel_workers"]  = m.get("parallel_workers", 0)
        m["map_time_sec"]      = m.get("map_time_sec", 0)
        m["reduce_time_sec"]   = m.get("reduce_time_sec", 0)

    # v2 QA specific
    if m.get("type") == "qa":
        m["chunks_retrieved"]  = m.get("chunks_retrieved", 0)

    state["metrics"] = m
    return state

# ============================================================
# LANGGRAPH ROUTER
# ============================================================
def route_query(state: DocState) -> str:
    if state["query_type"] == "FULL_SUMMARY":
        return "summarize"
    return "qa"

def route_validate(state: DocState) -> str:
    if not state["answer"].strip() and \
       state.get("retry_count", 0) < 2:
        return "retry"
    return "done"

# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================
def build_workflow():
    workflow = StateGraph(DocState)

    # Add nodes
    workflow.add_node("extract", node_extract)
    workflow.add_node("chunk", node_chunk)
    workflow.add_node("summarize", node_summarize)
    workflow.add_node("qa", node_qa)
    workflow.add_node("validate", node_validate)

    # Entry point
    workflow.set_entry_point("extract")

    # Linear flow: extract → chunk → route
    workflow.add_edge("extract", "chunk")

    # Conditional routing after chunk
    workflow.add_conditional_edges(
        "chunk",
        route_query,
        {
            "summarize": "summarize",
            "qa": "qa"
        }
    )

    # Both paths go to validate
    workflow.add_edge("summarize", "validate")
    workflow.add_edge("qa", "validate")

    # Validate: done or retry
    workflow.add_conditional_edges(
        "validate",
        route_validate,
        {
            "retry": "qa",
            "done": END
        }
    )

    return workflow.compile()

# Build workflow once at startup
print("Building LangGraph workflow...")
workflow = build_workflow()
print("Workflow ready!")

# ============================================================
# FLASK ROUTE
# ============================================================
@app.route("/process", methods=["POST"])
def process():
    data = request.json
    pdf_path = data.get("pdf_path")
    question = data.get("question")

    if not pdf_path or not question:
        return jsonify({"answer": "Missing PDF path or question."})
    if not os.path.exists(pdf_path):
        return jsonify({"answer": "File not found."})

    # Initial state
    initial_state: DocState = {
        "pdf_path":       pdf_path,
        "question":       question,
        "extracted_text": "",
        "chunks":         [],
        "faiss_index":    None,
        "answer":         "",
        "metrics":        {},
        "doc_type":       "general",
        "query_type":     "",
        "retry_count":    0,
        "start_time":     time.time(),
        "page_count":     0,
        "char_count":     0
    }

    try:
        # Run LangGraph workflow
        result = workflow.invoke(initial_state)
        m = result["metrics"]

        # ---- V1 METRICS (exact same fields as old app.py benchmark) ----
        v1 = {
            "type":                  m.get("type", ""),
            "response_time_sec":     round(m.get("response_time_sec", 0), 2),
            "extraction_time_sec":   round(m.get("extraction_time_sec", 0), 2),
            "pages_processed":       m.get("pages_processed", 0),
            "characters_processed":  m.get("characters_processed", 0),
            "words_processed":       m.get("words_processed", 0),
        }
        if m.get("type") == "summary":
            v1["summary_time_sec"]     = round(m.get("summary_time_sec", 0), 2)
            v1["summary_length_words"] = m.get("summary_length_words", 0)
        if m.get("type") == "qa":
            v1["qa_time_sec"]      = round(m.get("qa_time_sec", 0), 2)
            # confidence_score: store as raw float (0.937)
            # index.html multiplies by 100 to show as % (93.7%)
            # same behavior as old app.py
            v1["confidence_score"] = round(m.get("confidence_score", 0) / 100, 4) \
                if m.get("confidence_score", 0) > 1 \
                else round(m.get("confidence_score", 0), 4)

        # ---- V2 METRICS (new fields) ----
        v2 = {
            "ttft_sec":        round(m.get("ttft_sec", 0), 2),
            "tps":             m.get("tps", 0),
            "doc_type":        m.get("doc_type", "general"),
            "query_type":      m.get("query_type", ""),
            "model_used":      m.get("model_used", ""),
            "chunks_created":  m.get("chunks_created", 0),
            "retry_count":     m.get("retry_count", 0),
        }
        if m.get("type") == "summary":
            v2["parallel_workers"] = m.get("parallel_workers", 0)
            v2["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
            v2["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
        if m.get("type") == "qa":
            v2["chunks_retrieved"] = m.get("chunks_retrieved", 0)

        return jsonify({
            "answer":     result["answer"],
            "metrics":    {**v1, **v2},   # merged for frontend
            "metrics_v1": v1,             # old metrics isolated
            "metrics_v2": v2,             # new metrics isolated
        })

    except Exception as e:
        return jsonify({
            "answer":     f"Error: {str(e)}",
            "metrics":    {},
            "metrics_v1": {},
            "metrics_v2": {}
        })

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("Flask server running on http://127.0.0.1:5000")
    app.run(port=5000, threaded=True)




















#  # this do not go for full pdf 
# ============================================================
# # app.py - Agentic Document Intelligence System v2
# # Stack: LangGraph + LLaMA (Ollama) + FAISS RAG + RoBERTa
# # ============================================================

# import warnings
# warnings.filterwarnings("ignore")

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import re
# import io
# import time
# import json
# import hashlib
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import TypedDict, Optional

# # PDF + OCR
# import fitz
# import pytesseract
# from PIL import Image

# # LangChain + LangGraph
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langgraph.graph import StateGraph, END

# # Ollama (LLaMA)
# import ollama

# # HuggingFace (RoBERTa)
# from transformers import pipeline, logging
# logging.set_verbosity_error()

# import torch

# # ============================================================
# # FLASK SETUP
# # ============================================================
# app = Flask(__name__)
# CORS(app)

# # ============================================================
# # CONFIG
# # ============================================================
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# OLLAMA_MODEL    = "llama3.2"
# CHUNK_SIZE      = 10000      # bigger chunks = fewer LLM calls
# CHUNK_OVERLAP   = 500
# MAX_WORKERS     = 6          # parallel workers
# EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
# FAISS_CACHE_DIR = "faiss_cache"
# os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

# device = 0 if torch.cuda.is_available() else -1

# # ============================================================
# # LOAD ROBERTA (once at startup - small model)
# # ============================================================
# print("Loading RoBERTa QA model...")
# qa_pipeline = pipeline(
#     "question-answering",
#     model="deepset/roberta-base-squad2",
#     device=device
# )
# print("RoBERTa loaded!")

# # ============================================================
# # LOAD EMBEDDINGS (once at startup)
# # ============================================================
# print("Loading embedding model...")
# embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# print("Embeddings loaded!")

# # ============================================================
# # LANGGRAPH STATE
# # ============================================================
# class DocState(TypedDict):
#     pdf_path:         str
#     question:         str
#     extracted_text:   str
#     chunks:           list
#     faiss_index:      object
#     answer:           str
#     metrics:          dict
#     doc_type:         str
#     query_type:       str
#     retry_count:      int
#     start_time:       float
#     page_count:       int
#     char_count:       int

# # ============================================================
# # PROMPTS
# # ============================================================
# SUMMARY_PROMPTS = {
#     "academic": (
#         "You are an academic summarizer. Extract: key concepts, "
#         "methodology, findings, conclusions. Use bullet points. "
#         "Be concise.\n\nText:\n{text}\n\nBullet point summary:"
#     ),
#     "legal": (
#         "You are a legal document analyst. Extract: parties, "
#         "key clauses, obligations, dates. Be precise.\n\n"
#         "Text:\n{text}\n\nKey points:"
#     ),
#     "technical": (
#         "You are a technical writer. Extract: key concepts, "
#         "processes, specifications. Use bullet points.\n\n"
#         "Text:\n{text}\n\nTechnical summary:"
#     ),
#     "general": (
#         "Summarize the key points of this text in clear "
#         "bullet points. Be concise.\n\nText:\n{text}\n\nSummary:"
#     )
# }

# MULTIPART_PROMPT = (
#     "Context:\n{context}\n\n"
#     "Question: {question}\n\n"
#     "List ALL items mentioned in the context that answer "
#     "this question. Number each one. Do not miss any.\n\n"
#     "Complete answer:"
# )

# REASONING_PROMPT = (
#     "Context:\n{context}\n\n"
#     "Question: {question}\n\n"
#     "Think step by step using only the context above. "
#     "Provide a reasoned explanation.\n\n"
#     "Answer:"
# )

# MERGE_PROMPT = (
#     "Merge these summaries into one coherent final summary. "
#     "Remove repetition. Keep all key points. "
#     "Use clear bullet points.\n\n"
#     "Summaries:\n{summaries}\n\n"
#     "Final merged summary:"
# )

# # ============================================================
# # UTILITY: PDF HASH (for FAISS cache)
# # ============================================================
# def get_pdf_hash(pdf_path: str) -> str:
#     h = hashlib.md5()
#     with open(pdf_path, "rb") as f:
#         h.update(f.read())
#     return h.hexdigest()

# # ============================================================
# # UTILITY: FORMAT TIME
# # ============================================================
# def format_time(seconds: float) -> str:
#     if seconds < 60:
#         return f"{seconds:.2f} sec"
#     mins = int(seconds // 60)
#     secs = seconds % 60
#     return f"{mins} min {secs:.2f} sec"

# # ============================================================
# # UTILITY: DETECT QUERY TYPE
# # ============================================================
# def detect_query_type(question: str) -> str:
#     q = question.lower().strip()

#     # Summary triggers
#     summary_keywords = ["summary", "summarize", "overview",
#                         "brief", "outline", "gist", "tldr"]
#     if any(k in q for k in summary_keywords):
#         return "FULL_SUMMARY"

#     # Multi-part triggers
#     multipart_keywords = ["list", "all", "every", "what are",
#                           "how many", "enumerate", "types of",
#                           "examples of", "mention"]
#     if any(k in q for k in multipart_keywords):
#         return "MULTIPART_QA"

#     # Reasoning triggers
#     reasoning_keywords = ["why", "how does", "explain",
#                           "what would", "compare", "difference",
#                           "relationship", "impact", "effect"]
#     if any(k in q for k in reasoning_keywords):
#         return "REASONING_QA"

#     # Default: factual
#     return "FACTUAL_QA"

# # ============================================================
# # UTILITY: DETECT DOCUMENT TYPE
# # ============================================================
# def detect_doc_type(text: str) -> str:
#     sample = text[:1000].lower()
#     if any(w in sample for w in ["abstract", "methodology",
#            "conclusion", "research", "university", "lecture"]):
#         return "academic"
#     if any(w in sample for w in ["whereas", "clause", "agreement",
#            "party", "hereby", "legal", "contract"]):
#         return "legal"
#     if any(w in sample for w in ["api", "function", "install",
#            "configure", "technical", "specification"]):
#         return "technical"
#     return "general"

# # ============================================================
# # CORE: EXTRACT SINGLE PAGE
# # ============================================================
# def extract_page(args):
#     page, page_num = args
#     try:
#         text = page.get_text().strip()
#         if text:
#             return page_num, text
#         # OCR fallback for scanned pages
#         pix = page.get_pixmap(dpi=200)
#         img = Image.frombytes("RGB",
#               [pix.width, pix.height], pix.samples)
#         ocr_text = pytesseract.image_to_string(img).strip()
#         return page_num, ocr_text
#     except Exception:
#         return page_num, ""

# # ============================================================
# # CORE: PARALLEL PDF EXTRACTION
# # ============================================================
# def extract_pdf_parallel(pdf_path: str):
#     doc = fitz.open(pdf_path)
#     page_count = len(doc)
#     pages = [(doc[i], i) for i in range(page_count)]

#     results = {}
#     workers = min(MAX_WORKERS, page_count)

#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         futures = {ex.submit(extract_page, p): p[1] for p in pages}
#         for future in as_completed(futures):
#             page_num, text = future.result()
#             results[page_num] = text

#     doc.close()

#     # Join in page order
#     full_text = " ".join(
#         results[i] for i in sorted(results.keys())
#     )
#     full_text = re.sub(r'\s+', ' ', full_text).strip()
#     return full_text, page_count

# # ============================================================
# # CORE: SEMANTIC CHUNKING
# # ============================================================
# def semantic_chunk(text: str) -> list:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     return splitter.split_text(text)

# # ============================================================
# # CORE: LLaMA CALL (via Ollama)
# # ============================================================
# def call_llama(prompt: str) -> str:
#     try:
#         response = ollama.chat(
#             model=OLLAMA_MODEL,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response['message']['content'].strip()
#     except Exception as e:
#         return f"LLaMA error: {str(e)}"

# # ============================================================
# # CORE: TF-IDF EXTRACTIVE SUMMARY (replaces LLaMA for map phase)
# # Extracts top sentences by importance — no LLM needed = instant!
# # ============================================================
# def extractive_summary(chunk: str, top_n: int = 8) -> str:
#     # Split into sentences
#     sentences = re.split(r"(?<=[.!?])\s+", chunk.strip())
#     sentences = [s.strip() for s in sentences if len(s.strip()) > 30]

#     if not sentences:
#         return chunk[:1000]
#     if len(sentences) <= top_n:
#         return " ".join(sentences)

#     try:
#         # TF-IDF score each sentence
#         vectorizer = TfidfVectorizer(stop_words="english")
#         tfidf_matrix = vectorizer.fit_transform(sentences)
#         scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

#         # Pick top_n sentences in original order
#         top_indices = sorted(
#             np.argsort(scores)[-top_n:].tolist()
#         )
#         return " ".join(sentences[i] for i in top_indices)
#     except Exception:
#         return " ".join(sentences[:top_n])

# # ============================================================
# # CORE: RAPTOR HIERARCHICAL SUMMARY
# # MAP  = TF-IDF extractive (parallel, instant, no LLM)
# # REDUCE = single LLaMA call for final coherent summary
# # ============================================================
# def raptor_summarize(chunks: list, doc_type: str):

#     if not chunks:
#         return "No content to summarize.", 0, 0

#     # ---- MAP PHASE: TF-IDF extractive — truly parallel, no LLM ----
#     map_start = time.time()

#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
#         futures = [ex.submit(extractive_summary, chunk, 8)
#                    for chunk in chunks]
#         mini_summaries = [f.result() for f in futures
#                           if f.result().strip()]

#     map_time = time.time() - map_start
#     print(f"[RAPTOR] Map (TF-IDF): {len(mini_summaries)} chunks in {map_time:.1f}s")

#     if not mini_summaries:
#         return "Could not generate summary.", map_time, 0

#     # ---- REDUCE PHASE: single LLaMA call ----
#     reduce_start = time.time()

#     # Combine all extracted key sentences
#     combined = "\n\n---\n\n".join(mini_summaries)

#     # If too long, do one quick mid-level merge first
#     if len(combined) > 12000:
#         batch_size = 6
#         mid = []
#         for i in range(0, len(mini_summaries), batch_size):
#             batch = mini_summaries[i:i + batch_size]
#             mid.append(call_llama(
#                 MERGE_PROMPT.format(
#                     summaries="\n\n---\n\n".join(batch)
#                 )
#             ))
#         combined = "\n\n---\n\n".join(mid)

#     # Final LLaMA call — 1 call total for whole document!
#     final_prompt = (
#         "You are summarizing a document. Below are key extracted "
#         "points from all sections.\n\n"
#         f"{combined[:8000]}\n\n"
#         "Write a comprehensive, well-structured final summary "
#         "with clear bullet points covering all major topics:\n"
#     )
#     final_summary = call_llama(final_prompt)

#     reduce_time = time.time() - reduce_start
#     print(f"[RAPTOR] Reduce (LLaMA): {reduce_time:.1f}s  |  Total: {map_time+reduce_time:.1f}s")

#     return final_summary, map_time, reduce_time

# # ============================================================
# # CORE: BUILD FAISS INDEX
# # ============================================================
# def build_faiss_index(chunks: list, pdf_hash: str):
#     cache_path = os.path.join(FAISS_CACHE_DIR, pdf_hash)

#     # Try loading from cache first
#     if os.path.exists(cache_path):
#         try:
#             return FAISS.load_local(
#                 cache_path,
#                 embedding_model,
#                 allow_dangerous_deserialization=True
#             )
#         except Exception:
#             pass

#     # Build new index
#     docs = [Document(page_content=c,
#             metadata={"chunk_id": i})
#             for i, c in enumerate(chunks)]

#     index = FAISS.from_documents(docs, embedding_model)

#     # Save to cache
#     try:
#         index.save_local(cache_path)
#     except Exception:
#         pass

#     return index

# # ============================================================
# # CORE: COMPUTE CONFIDENCE
# # ============================================================
# def compute_confidence(question: str, docs: list) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(
#                 d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) *
#                 np.linalg.norm(d_embed) + 1e-8
#             )
#             sims.append(sim)
#         return round(float(np.mean(sims)) * 100, 2)
#     except Exception:
#         return 0.0

# # ============================================================
# # CORE: ROBERTA QA
# # ============================================================
# def roberta_qa(question: str, chunks: list):
#     best_answer = ""
#     best_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(
#                 question=question,
#                 context=chunk[:2000]
#             )
#             if result['score'] > best_score:
#                 best_score = result['score']
#                 best_answer = result['answer']
#         except Exception:
#             continue

#     return best_answer, round(best_score, 4)

# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# # Node 1: Extract PDF
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time = time.time() - extract_start

#     state["extracted_text"] = text
#     state["page_count"] = page_count
#     state["char_count"] = len(text)
#     state["metrics"]["extraction_time_sec"] = round(
#         extract_time, 2)
#     state["metrics"]["pages_processed"] = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"] = len(text.split())
#     return state

# # Node 2: Detect type + chunk
# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)
#     state["query_type"] = detect_query_type(state["question"])
#     state["chunks"] = semantic_chunk(text)
#     state["metrics"]["chunks_created"] = len(state["chunks"])
#     state["metrics"]["doc_type"] = state["doc_type"]
#     state["metrics"]["query_type"] = state["query_type"]
#     return state

# # Node 3: Summary path
# def node_summarize(state: DocState) -> DocState:
#     summary_start = time.time()
#     chunks = state["chunks"]
#     doc_type = state["doc_type"]

#     summary, map_time, reduce_time = raptor_summarize(chunks, doc_type)
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     state["metrics"]["summary_time_sec"] = round(summary_time, 2)
#     state["metrics"]["summary_length_words"] = len(summary.split())
#     state["metrics"]["parallel_workers"] = min(MAX_WORKERS, len(chunks))
#     state["metrics"]["map_time_sec"] = round(map_time, 2)
#     state["metrics"]["reduce_time_sec"] = round(reduce_time, 2)
#     state["metrics"]["type"] = "summary"
#     return state

# # Node 4: QA path
# def node_qa(state: DocState) -> DocState:
#     qa_start = time.time()
#     question = state["question"]
#     query_type = state["query_type"]
#     chunks = state["chunks"]

#     # Build or load FAISS index
#     pdf_hash = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(chunks, pdf_hash)

#     # Retrieve relevant chunks
#     k = 8 if query_type in ["MULTIPART_QA",
#                              "REASONING_QA"] else 5
#     retrieved = faiss_index.similarity_search(question, k=k)
#     retrieved_texts = [d.page_content for d in retrieved]

#     # Confidence score
#     confidence = compute_confidence(question, retrieved)

#     answer = ""
#     model_used = ""

#     if query_type == "FACTUAL_QA":
#         # Try RoBERTa first
#         rob_answer, rob_score = roberta_qa(
#             question, retrieved_texts)

#         if rob_score > 0.3 and rob_answer.strip():
#             answer = rob_answer
#             model_used = "roberta"
#             confidence = round(rob_score * 100, 2)
#         else:
#             # Fallback to LLaMA
#             context = "\n\n".join(retrieved_texts)
#             prompt = (
#                 f"Context:\n{context}\n\n"
#                 f"Question: {question}\n\n"
#                 f"Give a short exact answer from "
#                 f"the context only:\n"
#             )
#             answer = call_llama(prompt)
#             model_used = "llama_fallback"

#     elif query_type == "MULTIPART_QA":
#         context = "\n\n".join(retrieved_texts)
#         prompt = MULTIPART_PROMPT.format(
#             context=context,
#             question=question
#         )
#         answer = call_llama(prompt)
#         model_used = "llama"

#     elif query_type == "REASONING_QA":
#         context = "\n\n".join(retrieved_texts)
#         prompt = REASONING_PROMPT.format(
#             context=context,
#             question=question
#         )
#         answer = call_llama(prompt)
#         model_used = "llama"

#     else:
#         # Default factual
#         rob_answer, rob_score = roberta_qa(
#             question, retrieved_texts)
#         answer = rob_answer if rob_answer else "Not found."
#         model_used = "roberta"

#     qa_time = time.time() - qa_start

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."

#     state["answer"] = answer
#     state["metrics"]["qa_time_sec"] = round(qa_time, 2)
#     state["metrics"]["confidence_score"] = confidence
#     state["metrics"]["model_used"] = model_used
#     state["metrics"]["chunks_retrieved"] = k
#     state["metrics"]["type"] = "qa"
#     return state

# # Node 5: Validate output
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry = state.get("retry_count", 0)

#     # Basic quality checks
#     if len(answer.strip()) < 10 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"] = ""
#         return state

#     # ---- FINAL TIMING ----
#     total_time = time.time() - state["start_time"]

#     # ---- TOKENS PER SECOND ----
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     tps = round(output_tokens / total_time, 2) if total_time > 0 else 0

#     m = state["metrics"]

#     # ============================================================
#     # V1 METRICS (old - kept for benchmarking comparison)
#     # These match exactly what v1 app.py returned
#     # ============================================================
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # v1 summary specific
#     if m.get("type") == "summary":
#         m["summary_time_sec"]      = m.get("summary_time_sec", 0)
#         m["summary_length_words"]  = len(answer.split())

#     # v1 QA specific
#     if m.get("type") == "qa":
#         m["qa_time_sec"]       = m.get("qa_time_sec", 0)
#         m["confidence_score"]  = m.get("confidence_score", 0)

#     # ============================================================
#     # V2 METRICS (new - industry standard)
#     # ============================================================
#     m["ttft_sec"]          = round(total_time, 2)   # time to first token
#     m["tps"]               = tps                     # tokens per second
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "roberta")

#     # v2 summary specific
#     if m.get("type") == "summary":
#         m["parallel_workers"]  = m.get("parallel_workers", 0)
#         m["map_time_sec"]      = m.get("map_time_sec", 0)
#         m["reduce_time_sec"]   = m.get("reduce_time_sec", 0)

#     # v2 QA specific
#     if m.get("type") == "qa":
#         m["chunks_retrieved"]  = m.get("chunks_retrieved", 0)

#     state["metrics"] = m
#     return state

# # ============================================================
# # LANGGRAPH ROUTER
# # ============================================================
# def route_query(state: DocState) -> str:
#     if state["query_type"] == "FULL_SUMMARY":
#         return "summarize"
#     return "qa"

# def route_validate(state: DocState) -> str:
#     if not state["answer"].strip() and \
#        state.get("retry_count", 0) < 2:
#         return "retry"
#     return "done"

# # ============================================================
# # BUILD LANGGRAPH WORKFLOW
# # ============================================================
# def build_workflow():
#     workflow = StateGraph(DocState)

#     # Add nodes
#     workflow.add_node("extract", node_extract)
#     workflow.add_node("chunk", node_chunk)
#     workflow.add_node("summarize", node_summarize)
#     workflow.add_node("qa", node_qa)
#     workflow.add_node("validate", node_validate)

#     # Entry point
#     workflow.set_entry_point("extract")

#     # Linear flow: extract → chunk → route
#     workflow.add_edge("extract", "chunk")

#     # Conditional routing after chunk
#     workflow.add_conditional_edges(
#         "chunk",
#         route_query,
#         {
#             "summarize": "summarize",
#             "qa": "qa"
#         }
#     )

#     # Both paths go to validate
#     workflow.add_edge("summarize", "validate")
#     workflow.add_edge("qa", "validate")

#     # Validate: done or retry
#     workflow.add_conditional_edges(
#         "validate",
#         route_validate,
#         {
#             "retry": "qa",
#             "done": END
#         }
#     )

#     return workflow.compile()

# # Build workflow once at startup
# print("Building LangGraph workflow...")
# workflow = build_workflow()
# print("Workflow ready!")

# # ============================================================
# # FLASK ROUTE
# # ============================================================
# @app.route("/process", methods=["POST"])
# def process():
#     data = request.json
#     pdf_path = data.get("pdf_path")
#     question = data.get("question")

#     if not pdf_path or not question:
#         return jsonify({"answer": "Missing PDF path or question."})
#     if not os.path.exists(pdf_path):
#         return jsonify({"answer": "File not found."})

#     # Initial state
#     initial_state: DocState = {
#         "pdf_path":       pdf_path,
#         "question":       question,
#         "extracted_text": "",
#         "chunks":         [],
#         "faiss_index":    None,
#         "answer":         "",
#         "metrics":        {},
#         "doc_type":       "general",
#         "query_type":     "",
#         "retry_count":    0,
#         "start_time":     time.time(),
#         "page_count":     0,
#         "char_count":     0
#     }

#     try:
#         # Run LangGraph workflow
#         result = workflow.invoke(initial_state)
#         m = result["metrics"]

#         # ---- V1 METRICS (exact same fields as old app.py benchmark) ----
#         v1 = {
#             "type":                  m.get("type", ""),
#             "response_time_sec":     round(m.get("response_time_sec", 0), 2),
#             "extraction_time_sec":   round(m.get("extraction_time_sec", 0), 2),
#             "pages_processed":       m.get("pages_processed", 0),
#             "characters_processed":  m.get("characters_processed", 0),
#             "words_processed":       m.get("words_processed", 0),
#         }
#         if m.get("type") == "summary":
#             v1["summary_time_sec"]     = round(m.get("summary_time_sec", 0), 2)
#             v1["summary_length_words"] = m.get("summary_length_words", 0)
#         if m.get("type") == "qa":
#             v1["qa_time_sec"]      = round(m.get("qa_time_sec", 0), 2)
#             # confidence_score: store as raw float (0.937)
#             # index.html multiplies by 100 to show as % (93.7%)
#             # same behavior as old app.py
#             v1["confidence_score"] = round(m.get("confidence_score", 0) / 100, 4) \
#                 if m.get("confidence_score", 0) > 1 \
#                 else round(m.get("confidence_score", 0), 4)

#         # ---- V2 METRICS (new fields) ----
#         v2 = {
#             "ttft_sec":        round(m.get("ttft_sec", 0), 2),
#             "tps":             m.get("tps", 0),
#             "doc_type":        m.get("doc_type", "general"),
#             "query_type":      m.get("query_type", ""),
#             "model_used":      m.get("model_used", ""),
#             "chunks_created":  m.get("chunks_created", 0),
#             "retry_count":     m.get("retry_count", 0),
#         }
#         if m.get("type") == "summary":
#             v2["parallel_workers"] = m.get("parallel_workers", 0)
#             v2["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#             v2["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#         if m.get("type") == "qa":
#             v2["chunks_retrieved"] = m.get("chunks_retrieved", 0)

#         return jsonify({
#             "answer":     result["answer"],
#             "metrics":    {**v1, **v2},   # merged for frontend
#             "metrics_v1": v1,             # old metrics isolated
#             "metrics_v2": v2,             # new metrics isolated
#         })

#     except Exception as e:
#         return jsonify({
#             "answer":     f"Error: {str(e)}",
#             "metrics":    {},
#             "metrics_v1": {},
#             "metrics_v2": {}
#         })

# # ============================================================
# # RUN
# # ============================================================
# if __name__ == "__main__":
#     print("Flask server running on http://127.0.0.1:5000")
#     app.run(port=5000, threaded=True)






























# # ============================================================
# # app.py - Agentic Document Intelligence System v2
# # Stack: LangGraph + LLaMA (Ollama) + FAISS RAG + RoBERTa
# # ============================================================


# #got this for this code i mean sometimes 10 mins sometimes 22 mins it gave 
# #(venv-genai+qa) PS C:\xampp\htdocs\genai-summary+qa> python -c "
## >> import time
# #>> import ollama
## >> from concurrent.futures import ThreadPoolExecutor
# #>> 
# #>> def call_llama(i):
# #>>     start = time.time()
# #>>     r = ollama.chat(model='llama3.2:1b', messages=[{'role':'user','content':'say hello in one word'}])
# #>>     t = round(time.time()-start, 2)
# #>>     print(f'Worker {i} done in {t}s')
# #>>     return t
# #>>
# #>> print('Testing SEQUENTIAL (one by one):')
# #>> start = time.time()
# #>> for i in range(3):
# # >>     call_llama(i)
# # >> print(f'Sequential total: {round(time.time()-start,2)}s')
# # >>
# # >> print()
# # >> print('Testing PARALLEL (all at once):')
# # >> start = time.time()
# # >> with ThreadPoolExecutor(max_workers=3) as ex:
# # >>     list(ex.map(call_llama, range(3)))
# # >> print(f'Parallel total: {round(time.time()-start,2)}s')
# # >> "
# # Testing SEQUENTIAL (one by one):
# # Worker 0 done in 4.65s
# # Worker 1 done in 0.39s
# # Worker 2 done in 0.35s
# # Sequential total: 5.39s

# # Testing PARALLEL (all at once):
# # Worker 1 done in 0.39s
# # Worker 0 done in 0.52s
# # Worker 2 done in 0.6s
# # Parallel total: 0.6s
# # import warnings
# # warnings.filterwarnings("ignore")

# # from flask import Flask, request, jsonify
# # from flask_cors import CORS
#  import os
# import re
# import io
# import time
# import json
# import hashlib
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import TypedDict, Optional

# # PDF + OCR
# import fitz
# import pytesseract
# from PIL import Image

# # LangChain + LangGraph
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langgraph.graph import StateGraph, END

# # Ollama (LLaMA)
# import ollama

# # HuggingFace (RoBERTa)
# from transformers import pipeline, logging
# logging.set_verbosity_error()

# import torch

# # ============================================================
# # FLASK SETUP
# # ============================================================
# app = Flask(__name__)
# CORS(app)

# # ============================================================
# # CONFIG
# # ============================================================
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# # OLLAMA_MODEL    = "llama3.2"
# OLLAMA_MODEL    = "llama3.2:1b"
# CHUNK_SIZE      = 10000      # bigger chunks = fewer LLM calls
# CHUNK_OVERLAP   = 500
# MAX_WORKERS     = 6          # parallel workers
# EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
# FAISS_CACHE_DIR = "faiss_cache"
# os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

# device = 0 if torch.cuda.is_available() else -1

# # ============================================================
# # LOAD ROBERTA (once at startup - small model)
# # ============================================================
# print("Loading RoBERTa QA model...")
# qa_pipeline = pipeline(
#     "question-answering",
#     model="deepset/roberta-base-squad2",
#     device=device
# )
# print("RoBERTa loaded!")

# # ============================================================
# # LOAD EMBEDDINGS (once at startup)
# # ============================================================
# print("Loading embedding model...")
# embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# print("Embeddings loaded!")

# # ============================================================
# # LANGGRAPH STATE
# # ============================================================
# class DocState(TypedDict):
#     pdf_path:         str
#     question:         str
#     extracted_text:   str
#     chunks:           list
#     faiss_index:      object
#     answer:           str
#     metrics:          dict
#     doc_type:         str
#     query_type:       str
#     retry_count:      int
#     start_time:       float
#     page_count:       int
#     char_count:       int

# # ============================================================
# # PROMPTS
# # ============================================================
# SUMMARY_PROMPTS = {
#     "academic": (
#         "You are an academic summarizer. Extract: key concepts, "
#         "methodology, findings, conclusions. Use bullet points. "
#         "Be concise.\n\nText:\n{text}\n\nBullet point summary:"
#     ),
#     "legal": (
#         "You are a legal document analyst. Extract: parties, "
#         "key clauses, obligations, dates. Be precise.\n\n"
#         "Text:\n{text}\n\nKey points:"
#     ),
#     "technical": (
#         "You are a technical writer. Extract: key concepts, "
#         "processes, specifications. Use bullet points.\n\n"
#         "Text:\n{text}\n\nTechnical summary:"
#     ),
#     "general": (
#         "Summarize the key points of this text in clear "
#         "bullet points. Be concise.\n\nText:\n{text}\n\nSummary:"
#     )
# }

# MULTIPART_PROMPT = (
#     "Context:\n{context}\n\n"
#     "Question: {question}\n\n"
#     "List ALL items mentioned in the context that answer "
#     "this question. Number each one. Do not miss any.\n\n"
#     "Complete answer:"
# )

# REASONING_PROMPT = (
#     "Context:\n{context}\n\n"
#     "Question: {question}\n\n"
#     "Think step by step using only the context above. "
#     "Provide a reasoned explanation.\n\n"
#     "Answer:"
# )

# MERGE_PROMPT = (
#     "Merge these summaries into one coherent final summary. "
#     "Remove repetition. Keep all key points. "
#     "Use clear bullet points.\n\n"
#     "Summaries:\n{summaries}\n\n"
#     "Final merged summary:"
# )

# # ============================================================
# # UTILITY: PDF HASH (for FAISS cache)
# # ============================================================
# def get_pdf_hash(pdf_path: str) -> str:
#     h = hashlib.md5()
#     with open(pdf_path, "rb") as f:
#         h.update(f.read())
#     return h.hexdigest()

# # ============================================================
# # UTILITY: FORMAT TIME
# # ============================================================
# def format_time(seconds: float) -> str:
#     if seconds < 60:
#         return f"{seconds:.2f} sec"
#     mins = int(seconds // 60)
#     secs = seconds % 60
#     return f"{mins} min {secs:.2f} sec"

# # ============================================================
# # UTILITY: DETECT QUERY TYPE
# # ============================================================
# def detect_query_type(question: str) -> str:
#     q = question.lower().strip()

#     # Summary triggers
#     summary_keywords = ["summary", "summarize", "overview",
#                         "brief", "outline", "gist", "tldr"]
#     if any(k in q for k in summary_keywords):
#         return "FULL_SUMMARY"

#     # Multi-part triggers
#     multipart_keywords = ["list", "all", "every", "what are",
#                           "how many", "enumerate", "types of",
#                           "examples of", "mention"]
#     if any(k in q for k in multipart_keywords):
#         return "MULTIPART_QA"

#     # Reasoning triggers
#     reasoning_keywords = ["why", "how does", "explain",
#                           "what would", "compare", "difference",
#                           "relationship", "impact", "effect"]
#     if any(k in q for k in reasoning_keywords):
#         return "REASONING_QA"

#     # Default: factual
#     return "FACTUAL_QA"

# # ============================================================
# # UTILITY: DETECT DOCUMENT TYPE
# # ============================================================
# def detect_doc_type(text: str) -> str:
#     sample = text[:1000].lower()
#     if any(w in sample for w in ["abstract", "methodology",
#            "conclusion", "research", "university", "lecture"]):
#         return "academic"
#     if any(w in sample for w in ["whereas", "clause", "agreement",
#            "party", "hereby", "legal", "contract"]):
#         return "legal"
#     if any(w in sample for w in ["api", "function", "install",
#            "configure", "technical", "specification"]):
#         return "technical"
#     return "general"

# # ============================================================
# # CORE: EXTRACT SINGLE PAGE
# # ============================================================
# def extract_page(args):
#     page, page_num = args
#     try:
#         text = page.get_text().strip()
#         if text:
#             return page_num, text
#         # OCR fallback for scanned pages
#         pix = page.get_pixmap(dpi=200)
#         img = Image.frombytes("RGB",
#               [pix.width, pix.height], pix.samples)
#         ocr_text = pytesseract.image_to_string(img).strip()
#         return page_num, ocr_text
#     except Exception:
#         return page_num, ""

# # ============================================================
# # CORE: PARALLEL PDF EXTRACTION
# # ============================================================
# def extract_pdf_parallel(pdf_path: str):
#     doc = fitz.open(pdf_path)
#     page_count = len(doc)
#     pages = [(doc[i], i) for i in range(page_count)]

#     results = {}
#     workers = min(MAX_WORKERS, page_count)

#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         futures = {ex.submit(extract_page, p): p[1] for p in pages}
#         for future in as_completed(futures):
#             page_num, text = future.result()
#             results[page_num] = text

#     doc.close()

#     # Join in page order
#     full_text = " ".join(
#         results[i] for i in sorted(results.keys())
#     )
#     full_text = re.sub(r'\s+', ' ', full_text).strip()
#     return full_text, page_count

# # ============================================================
# # CORE: SEMANTIC CHUNKING
# # ============================================================
# def semantic_chunk(text: str) -> list:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     return splitter.split_text(text)

# # ============================================================
# # CORE: LLaMA CALL (via Ollama)
# # ============================================================
# def call_llama(prompt: str) -> str:
#     try:
#         response = ollama.chat(
#             model=OLLAMA_MODEL,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response['message']['content'].strip()
#     except Exception as e:
#         return f"LLaMA error: {str(e)}"

# # ============================================================
# # CORE: SUMMARIZE SINGLE CHUNK (for parallel map)
# # ============================================================
# def summarize_chunk(args):
#     chunk, doc_type = args
#     prompt_template = SUMMARY_PROMPTS.get(doc_type,
#                       SUMMARY_PROMPTS["general"])
#     prompt = prompt_template.format(text=chunk[:3000])
#     return call_llama(prompt)

# # ============================================================
# # CORE: RAPTOR HIERARCHICAL SUMMARY
# # ============================================================
# def raptor_summarize(chunks: list, doc_type: str) -> str:

#     if not chunks:
#         return "No content to summarize."

#     # ---- MAP PHASE: parallel chunk summarization ----
#     map_start = time.time()
#     workers = min(MAX_WORKERS, len(chunks))
#     args = [(chunk, doc_type) for chunk in chunks]
#     mini_summaries = []

#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         futures = list(ex.map(summarize_chunk, args))
#         mini_summaries = [f for f in futures if f and
#                          "error" not in f.lower()]

#     map_time = time.time() - map_start

#     if not mini_summaries:
#         return "Could not generate summary."

#     # ---- REDUCE PHASE: hierarchical merge ----
#     reduce_start = time.time()

#     # Group mini summaries into batches of 10
#     batch_size = 10
#     current_level = mini_summaries

#     while len(current_level) > 1:
#         next_level = []
#         for i in range(0, len(current_level), batch_size):
#             batch = current_level[i:i + batch_size]
#             if len(batch) == 1:
#                 next_level.append(batch[0])
#                 continue
#             combined = "\n\n---\n\n".join(batch)
#             prompt = MERGE_PROMPT.format(summaries=combined)
#             merged = call_llama(prompt)
#             next_level.append(merged)
#         current_level = next_level

#     reduce_time = time.time() - reduce_start
#     final_summary = current_level[0]
#     print(f"[RAPTOR] map={map_time:.1f}s reduce={reduce_time:.1f}s")

#     return final_summary, map_time, reduce_time

# # ============================================================
# # CORE: BUILD FAISS INDEX
# # ============================================================
# def build_faiss_index(chunks: list, pdf_hash: str):
#     cache_path = os.path.join(FAISS_CACHE_DIR, pdf_hash)

#     # Try loading from cache first
#     if os.path.exists(cache_path):
#         try:
#             return FAISS.load_local(
#                 cache_path,
#                 embedding_model,
#                 allow_dangerous_deserialization=True
#             )
#         except Exception:
#             pass

#     # Build new index
#     docs = [Document(page_content=c,
#             metadata={"chunk_id": i})
#             for i, c in enumerate(chunks)]

#     index = FAISS.from_documents(docs, embedding_model)

#     # Save to cache
#     try:
#         index.save_local(cache_path)
#     except Exception:
#         pass

#     return index

# # ============================================================
# # CORE: COMPUTE CONFIDENCE
# # ============================================================
# def compute_confidence(question: str, docs: list) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(
#                 d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) *
#                 np.linalg.norm(d_embed) + 1e-8
#             )
#             sims.append(sim)
#         return round(float(np.mean(sims)) * 100, 2)
#     except Exception:
#         return 0.0

# # ============================================================
# # CORE: ROBERTA QA
# # ============================================================
# def roberta_qa(question: str, chunks: list):
#     best_answer = ""
#     best_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(
#                 question=question,
#                 context=chunk[:2000]
#             )
#             if result['score'] > best_score:
#                 best_score = result['score']
#                 best_answer = result['answer']
#         except Exception:
#             continue

#     return best_answer, round(best_score, 4)

# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# # Node 1: Extract PDF
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time = time.time() - extract_start

#     state["extracted_text"] = text
#     state["page_count"] = page_count
#     state["char_count"] = len(text)
#     state["metrics"]["extraction_time_sec"] = round(
#         extract_time, 2)
#     state["metrics"]["pages_processed"] = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"] = len(text.split())
#     return state

# # Node 2: Detect type + chunk
# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)
#     state["query_type"] = detect_query_type(state["question"])
#     state["chunks"] = semantic_chunk(text)
#     state["metrics"]["chunks_created"] = len(state["chunks"])
#     state["metrics"]["doc_type"] = state["doc_type"]
#     state["metrics"]["query_type"] = state["query_type"]
#     return state

# # Node 3: Summary path
# def node_summarize(state: DocState) -> DocState:
#     summary_start = time.time()
#     chunks = state["chunks"]
#     doc_type = state["doc_type"]

#     summary, map_time, reduce_time = raptor_summarize(chunks, doc_type)
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     state["metrics"]["summary_time_sec"] = round(summary_time, 2)
#     state["metrics"]["summary_length_words"] = len(summary.split())
#     state["metrics"]["parallel_workers"] = min(MAX_WORKERS, len(chunks))
#     state["metrics"]["map_time_sec"] = round(map_time, 2)
#     state["metrics"]["reduce_time_sec"] = round(reduce_time, 2)
#     state["metrics"]["type"] = "summary"
#     return state

# # Node 4: QA path
# def node_qa(state: DocState) -> DocState:
#     qa_start = time.time()
#     question = state["question"]
#     query_type = state["query_type"]
#     chunks = state["chunks"]

#     # Build or load FAISS index
#     pdf_hash = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(chunks, pdf_hash)

#     # Retrieve relevant chunks
#     k = 8 if query_type in ["MULTIPART_QA",
#                              "REASONING_QA"] else 5
#     retrieved = faiss_index.similarity_search(question, k=k)
#     retrieved_texts = [d.page_content for d in retrieved]

#     # Confidence score
#     confidence = compute_confidence(question, retrieved)

#     answer = ""
#     model_used = ""

#     if query_type == "FACTUAL_QA":
#         # Try RoBERTa first
#         rob_answer, rob_score = roberta_qa(
#             question, retrieved_texts)

#         if rob_score > 0.3 and rob_answer.strip():
#             answer = rob_answer
#             model_used = "roberta"
#             confidence = round(rob_score * 100, 2)
#         else:
#             # Fallback to LLaMA
#             context = "\n\n".join(retrieved_texts)
#             prompt = (
#                 f"Context:\n{context}\n\n"
#                 f"Question: {question}\n\n"
#                 f"Give a short exact answer from "
#                 f"the context only:\n"
#             )
#             answer = call_llama(prompt)
#             model_used = "llama_fallback"

#     elif query_type == "MULTIPART_QA":
#         context = "\n\n".join(retrieved_texts)
#         prompt = MULTIPART_PROMPT.format(
#             context=context,
#             question=question
#         )
#         answer = call_llama(prompt)
#         model_used = "llama"

#     elif query_type == "REASONING_QA":
#         context = "\n\n".join(retrieved_texts)
#         prompt = REASONING_PROMPT.format(
#             context=context,
#             question=question
#         )
#         answer = call_llama(prompt)
#         model_used = "llama"

#     else:
#         # Default factual
#         rob_answer, rob_score = roberta_qa(
#             question, retrieved_texts)
#         answer = rob_answer if rob_answer else "Not found."
#         model_used = "roberta"

#     qa_time = time.time() - qa_start

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."

#     state["answer"] = answer
#     state["metrics"]["qa_time_sec"] = round(qa_time, 2)
#     state["metrics"]["confidence_score"] = confidence
#     state["metrics"]["model_used"] = model_used
#     state["metrics"]["chunks_retrieved"] = k
#     state["metrics"]["type"] = "qa"
#     return state

# # Node 5: Validate output
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry = state.get("retry_count", 0)

#     # Basic quality checks
#     if len(answer.strip()) < 10 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"] = ""
#         return state

#     # ---- FINAL TIMING ----
#     total_time = time.time() - state["start_time"]

#     # ---- TOKENS PER SECOND ----
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     tps = round(output_tokens / total_time, 2) if total_time > 0 else 0

#     m = state["metrics"]

#     # ============================================================
#     # V1 METRICS (old - kept for benchmarking comparison)
#     # These match exactly what v1 app.py returned
#     # ============================================================
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # v1 summary specific
#     if m.get("type") == "summary":
#         m["summary_time_sec"]      = m.get("summary_time_sec", 0)
#         m["summary_length_words"]  = len(answer.split())

#     # v1 QA specific
#     if m.get("type") == "qa":
#         m["qa_time_sec"]       = m.get("qa_time_sec", 0)
#         m["confidence_score"]  = m.get("confidence_score", 0)

#     # ============================================================
#     # V2 METRICS (new - industry standard)
#     # ============================================================
#     m["ttft_sec"]          = round(total_time, 2)   # time to first token
#     m["tps"]               = tps                     # tokens per second
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "roberta")

#     # v2 summary specific
#     if m.get("type") == "summary":
#         m["parallel_workers"]  = m.get("parallel_workers", 0)
#         m["map_time_sec"]      = m.get("map_time_sec", 0)
#         m["reduce_time_sec"]   = m.get("reduce_time_sec", 0)

#     # v2 QA specific
#     if m.get("type") == "qa":
#         m["chunks_retrieved"]  = m.get("chunks_retrieved", 0)

#     state["metrics"] = m
#     return state

# # ============================================================
# # LANGGRAPH ROUTER
# # ============================================================
# def route_query(state: DocState) -> str:
#     if state["query_type"] == "FULL_SUMMARY":
#         return "summarize"
#     return "qa"

# def route_validate(state: DocState) -> str:
#     if not state["answer"].strip() and \
#        state.get("retry_count", 0) < 2:
#         return "retry"
#     return "done"

# # ============================================================
# # BUILD LANGGRAPH WORKFLOW
# # ============================================================
# def build_workflow():
#     workflow = StateGraph(DocState)

#     # Add nodes
#     workflow.add_node("extract", node_extract)
#     workflow.add_node("chunk", node_chunk)
#     workflow.add_node("summarize", node_summarize)
#     workflow.add_node("qa", node_qa)
#     workflow.add_node("validate", node_validate)

#     # Entry point
#     workflow.set_entry_point("extract")

#     # Linear flow: extract → chunk → route
#     workflow.add_edge("extract", "chunk")

#     # Conditional routing after chunk
#     workflow.add_conditional_edges(
#         "chunk",
#         route_query,
#         {
#             "summarize": "summarize",
#             "qa": "qa"
#         }
#     )

#     # Both paths go to validate
#     workflow.add_edge("summarize", "validate")
#     workflow.add_edge("qa", "validate")

#     # Validate: done or retry
#     workflow.add_conditional_edges(
#         "validate",
#         route_validate,
#         {
#             "retry": "qa",
#             "done": END
#         }
#     )

#     return workflow.compile()

# # Build workflow once at startup
# print("Building LangGraph workflow...")
# workflow = build_workflow()
# print("Workflow ready!")

# # ============================================================
# # FLASK ROUTE
# # ============================================================
# @app.route("/process", methods=["POST"])
# def process():
#     data = request.json
#     pdf_path = data.get("pdf_path")
#     question = data.get("question")

#     if not pdf_path or not question:
#         return jsonify({"answer": "Missing PDF path or question."})
#     if not os.path.exists(pdf_path):
#         return jsonify({"answer": "File not found."})

#     # Initial state
#     initial_state: DocState = {
#         "pdf_path":       pdf_path,
#         "question":       question,
#         "extracted_text": "",
#         "chunks":         [],
#         "faiss_index":    None,
#         "answer":         "",
#         "metrics":        {},
#         "doc_type":       "general",
#         "query_type":     "",
#         "retry_count":    0,
#         "start_time":     time.time(),
#         "page_count":     0,
#         "char_count":     0
#     }

#     try:
#         # Run LangGraph workflow
#         result = workflow.invoke(initial_state)
#         m = result["metrics"]

#         # ---- V1 METRICS (exact same fields as old app.py benchmark) ----
#         v1 = {
#             "type":                  m.get("type", ""),
#             "response_time_sec":     round(m.get("response_time_sec", 0), 2),
#             "extraction_time_sec":   round(m.get("extraction_time_sec", 0), 2),
#             "pages_processed":       m.get("pages_processed", 0),
#             "characters_processed":  m.get("characters_processed", 0),
#             "words_processed":       m.get("words_processed", 0),
#         }
#         if m.get("type") == "summary":
#             v1["summary_time_sec"]     = round(m.get("summary_time_sec", 0), 2)
#             v1["summary_length_words"] = m.get("summary_length_words", 0)
#         if m.get("type") == "qa":
#             v1["qa_time_sec"]      = round(m.get("qa_time_sec", 0), 2)
#             # confidence_score: store as raw float (0.937)
#             # index.html multiplies by 100 to show as % (93.7%)
#             # same behavior as old app.py
#             v1["confidence_score"] = round(m.get("confidence_score", 0) / 100, 4) \
#                 if m.get("confidence_score", 0) > 1 \
#                 else round(m.get("confidence_score", 0), 4)

#         # ---- V2 METRICS (new fields) ----
#         v2 = {
#             "ttft_sec":        round(m.get("ttft_sec", 0), 2),
#             "tps":             m.get("tps", 0),
#             "doc_type":        m.get("doc_type", "general"),
#             "query_type":      m.get("query_type", ""),
#             "model_used":      m.get("model_used", ""),
#             "chunks_created":  m.get("chunks_created", 0),
#             "retry_count":     m.get("retry_count", 0),
#         }
#         if m.get("type") == "summary":
#             v2["parallel_workers"] = m.get("parallel_workers", 0)
#             v2["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#             v2["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#         if m.get("type") == "qa":
#             v2["chunks_retrieved"] = m.get("chunks_retrieved", 0)

#         return jsonify({
#             "answer":     result["answer"],
#             "metrics":    {**v1, **v2},   # merged for frontend
#             "metrics_v1": v1,             # old metrics isolated
#             "metrics_v2": v2,             # new metrics isolated
#         })

#     except Exception as e:
#         return jsonify({
#             "answer":     f"Error: {str(e)}",
#             "metrics":    {},
#             "metrics_v1": {},
#             "metrics_v2": {}
#         })

# # ============================================================
# # RUN
# # ============================================================
# if __name__ == "__main__":
#     print("Flask server running on http://127.0.0.1:5000")
#     app.run(port=5000, threaded=True)










# worked but slow# ============================================================
# # app.py - Agentic Document Intelligence System v2
# # Stack: LangGraph + LLaMA (Ollama) + FAISS RAG + RoBERTa
# # ============================================================

# import warnings
# warnings.filterwarnings("ignore")

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import re
# import io
# import time
# import json
# import hashlib
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import TypedDict, Optional

# # PDF + OCR
# import fitz
# import pytesseract
# from PIL import Image

# # LangChain + LangGraph
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langgraph.graph import StateGraph, END

# # Ollama (LLaMA)
# import ollama

# # HuggingFace (RoBERTa)
# from transformers import pipeline, logging
# logging.set_verbosity_error()

# import torch

# # ============================================================
# # FLASK SETUP
# # ============================================================
# app = Flask(__name__)
# CORS(app)

# # ============================================================
# # CONFIG
# # ============================================================
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# OLLAMA_MODEL    = "llama3.2"
# CHUNK_SIZE      = 4000       # bigger chunks = fewer LLM calls
# CHUNK_OVERLAP   = 200
# MAX_WORKERS     = 6          # parallel workers
# EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
# FAISS_CACHE_DIR = "faiss_cache"
# os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

# device = 0 if torch.cuda.is_available() else -1

# # ============================================================
# # LOAD ROBERTA (once at startup - small model)
# # ============================================================
# print("Loading RoBERTa QA model...")
# qa_pipeline = pipeline(
#     "question-answering",
#     model="deepset/roberta-base-squad2",
#     device=device
# )
# print("RoBERTa loaded!")

# # ============================================================
# # LOAD EMBEDDINGS (once at startup)
# # ============================================================
# print("Loading embedding model...")
# embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# print("Embeddings loaded!")

# # ============================================================
# # LANGGRAPH STATE
# # ============================================================
# class DocState(TypedDict):
#     pdf_path:         str
#     question:         str
#     extracted_text:   str
#     chunks:           list
#     faiss_index:      object
#     answer:           str
#     metrics:          dict
#     doc_type:         str
#     query_type:       str
#     retry_count:      int
#     start_time:       float
#     page_count:       int
#     char_count:       int

# # ============================================================
# # PROMPTS
# # ============================================================
# SUMMARY_PROMPTS = {
#     "academic": (
#         "You are an academic summarizer. Extract: key concepts, "
#         "methodology, findings, conclusions. Use bullet points. "
#         "Be concise.\n\nText:\n{text}\n\nBullet point summary:"
#     ),
#     "legal": (
#         "You are a legal document analyst. Extract: parties, "
#         "key clauses, obligations, dates. Be precise.\n\n"
#         "Text:\n{text}\n\nKey points:"
#     ),
#     "technical": (
#         "You are a technical writer. Extract: key concepts, "
#         "processes, specifications. Use bullet points.\n\n"
#         "Text:\n{text}\n\nTechnical summary:"
#     ),
#     "general": (
#         "Summarize the key points of this text in clear "
#         "bullet points. Be concise.\n\nText:\n{text}\n\nSummary:"
#     )
# }

# MULTIPART_PROMPT = (
#     "Context:\n{context}\n\n"
#     "Question: {question}\n\n"
#     "List ALL items mentioned in the context that answer "
#     "this question. Number each one. Do not miss any.\n\n"
#     "Complete answer:"
# )

# REASONING_PROMPT = (
#     "Context:\n{context}\n\n"
#     "Question: {question}\n\n"
#     "Think step by step using only the context above. "
#     "Provide a reasoned explanation.\n\n"
#     "Answer:"
# )

# MERGE_PROMPT = (
#     "Merge these summaries into one coherent final summary. "
#     "Remove repetition. Keep all key points. "
#     "Use clear bullet points.\n\n"
#     "Summaries:\n{summaries}\n\n"
#     "Final merged summary:"
# )

# # ============================================================
# # UTILITY: PDF HASH (for FAISS cache)
# # ============================================================
# def get_pdf_hash(pdf_path: str) -> str:
#     h = hashlib.md5()
#     with open(pdf_path, "rb") as f:
#         h.update(f.read())
#     return h.hexdigest()

# # ============================================================
# # UTILITY: FORMAT TIME
# # ============================================================
# def format_time(seconds: float) -> str:
#     if seconds < 60:
#         return f"{seconds:.2f} sec"
#     mins = int(seconds // 60)
#     secs = seconds % 60
#     return f"{mins} min {secs:.2f} sec"

# # ============================================================
# # UTILITY: DETECT QUERY TYPE
# # ============================================================
# def detect_query_type(question: str) -> str:
#     q = question.lower().strip()

#     # Summary triggers
#     summary_keywords = ["summary", "summarize", "overview",
#                         "brief", "outline", "gist", "tldr"]
#     if any(k in q for k in summary_keywords):
#         return "FULL_SUMMARY"

#     # Multi-part triggers
#     multipart_keywords = ["list", "all", "every", "what are",
#                           "how many", "enumerate", "types of",
#                           "examples of", "mention"]
#     if any(k in q for k in multipart_keywords):
#         return "MULTIPART_QA"

#     # Reasoning triggers
#     reasoning_keywords = ["why", "how does", "explain",
#                           "what would", "compare", "difference",
#                           "relationship", "impact", "effect"]
#     if any(k in q for k in reasoning_keywords):
#         return "REASONING_QA"

#     # Default: factual
#     return "FACTUAL_QA"

# # ============================================================
# # UTILITY: DETECT DOCUMENT TYPE
# # ============================================================
# def detect_doc_type(text: str) -> str:
#     sample = text[:1000].lower()
#     if any(w in sample for w in ["abstract", "methodology",
#            "conclusion", "research", "university", "lecture"]):
#         return "academic"
#     if any(w in sample for w in ["whereas", "clause", "agreement",
#            "party", "hereby", "legal", "contract"]):
#         return "legal"
#     if any(w in sample for w in ["api", "function", "install",
#            "configure", "technical", "specification"]):
#         return "technical"
#     return "general"

# # ============================================================
# # CORE: EXTRACT SINGLE PAGE
# # ============================================================
# def extract_page(args):
#     page, page_num = args
#     try:
#         text = page.get_text().strip()
#         if text:
#             return page_num, text
#         # OCR fallback for scanned pages
#         pix = page.get_pixmap(dpi=200)
#         img = Image.frombytes("RGB",
#               [pix.width, pix.height], pix.samples)
#         ocr_text = pytesseract.image_to_string(img).strip()
#         return page_num, ocr_text
#     except Exception:
#         return page_num, ""

# # ============================================================
# # CORE: PARALLEL PDF EXTRACTION
# # ============================================================
# def extract_pdf_parallel(pdf_path: str):
#     doc = fitz.open(pdf_path)
#     page_count = len(doc)
#     pages = [(doc[i], i) for i in range(page_count)]

#     results = {}
#     workers = min(MAX_WORKERS, page_count)

#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         futures = {ex.submit(extract_page, p): p[1] for p in pages}
#         for future in as_completed(futures):
#             page_num, text = future.result()
#             results[page_num] = text

#     doc.close()

#     # Join in page order
#     full_text = " ".join(
#         results[i] for i in sorted(results.keys())
#     )
#     full_text = re.sub(r'\s+', ' ', full_text).strip()
#     return full_text, page_count

# # ============================================================
# # CORE: SEMANTIC CHUNKING
# # ============================================================
# def semantic_chunk(text: str) -> list:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     return splitter.split_text(text)

# # ============================================================
# # CORE: LLaMA CALL (via Ollama)
# # ============================================================
# def call_llama(prompt: str) -> str:
#     try:
#         response = ollama.chat(
#             model=OLLAMA_MODEL,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response['message']['content'].strip()
#     except Exception as e:
#         return f"LLaMA error: {str(e)}"

# # ============================================================
# # CORE: SUMMARIZE SINGLE CHUNK (for parallel map)
# # ============================================================
# def summarize_chunk(args):
#     chunk, doc_type = args
#     prompt_template = SUMMARY_PROMPTS.get(doc_type,
#                       SUMMARY_PROMPTS["general"])
#     prompt = prompt_template.format(text=chunk[:3000])
#     return call_llama(prompt)

# # ============================================================
# # CORE: RAPTOR HIERARCHICAL SUMMARY
# # ============================================================
# def raptor_summarize(chunks: list, doc_type: str) -> str:

#     if not chunks:
#         return "No content to summarize."

#     # ---- MAP PHASE: parallel chunk summarization ----
#     map_start = time.time()
#     workers = min(MAX_WORKERS, len(chunks))
#     args = [(chunk, doc_type) for chunk in chunks]
#     mini_summaries = []

#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         futures = list(ex.map(summarize_chunk, args))
#         mini_summaries = [f for f in futures if f and
#                          "error" not in f.lower()]

#     map_time = time.time() - map_start

#     if not mini_summaries:
#         return "Could not generate summary."

#     # ---- REDUCE PHASE: hierarchical merge ----
#     reduce_start = time.time()

#     # Group mini summaries into batches of 6
#     batch_size = 6
#     current_level = mini_summaries

#     while len(current_level) > 1:
#         next_level = []
#         for i in range(0, len(current_level), batch_size):
#             batch = current_level[i:i + batch_size]
#             if len(batch) == 1:
#                 next_level.append(batch[0])
#                 continue
#             combined = "\n\n---\n\n".join(batch)
#             prompt = MERGE_PROMPT.format(summaries=combined)
#             merged = call_llama(prompt)
#             next_level.append(merged)
#         current_level = next_level

#     reduce_time = time.time() - reduce_start
#     final_summary = current_level[0]

#     return final_summary

# # ============================================================
# # CORE: BUILD FAISS INDEX
# # ============================================================
# def build_faiss_index(chunks: list, pdf_hash: str):
#     cache_path = os.path.join(FAISS_CACHE_DIR, pdf_hash)

#     # Try loading from cache first
#     if os.path.exists(cache_path):
#         try:
#             return FAISS.load_local(
#                 cache_path,
#                 embedding_model,
#                 allow_dangerous_deserialization=True
#             )
#         except Exception:
#             pass

#     # Build new index
#     docs = [Document(page_content=c,
#             metadata={"chunk_id": i})
#             for i, c in enumerate(chunks)]

#     index = FAISS.from_documents(docs, embedding_model)

#     # Save to cache
#     try:
#         index.save_local(cache_path)
#     except Exception:
#         pass

#     return index

# # ============================================================
# # CORE: COMPUTE CONFIDENCE
# # ============================================================
# def compute_confidence(question: str, docs: list) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(
#                 d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) *
#                 np.linalg.norm(d_embed) + 1e-8
#             )
#             sims.append(sim)
#         return round(float(np.mean(sims)) * 100, 2)
#     except Exception:
#         return 0.0

# # ============================================================
# # CORE: ROBERTA QA
# # ============================================================
# def roberta_qa(question: str, chunks: list):
#     best_answer = ""
#     best_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(
#                 question=question,
#                 context=chunk[:2000]
#             )
#             if result['score'] > best_score:
#                 best_score = result['score']
#                 best_answer = result['answer']
#         except Exception:
#             continue

#     return best_answer, round(best_score, 4)

# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# # Node 1: Extract PDF
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time = time.time() - extract_start

#     state["extracted_text"] = text
#     state["page_count"] = page_count
#     state["char_count"] = len(text)
#     state["metrics"]["extraction_time_sec"] = round(
#         extract_time, 2)
#     state["metrics"]["pages_processed"] = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"] = len(text.split())
#     return state

# # Node 2: Detect type + chunk
# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)
#     state["query_type"] = detect_query_type(state["question"])
#     state["chunks"] = semantic_chunk(text)
#     state["metrics"]["chunks_created"] = len(state["chunks"])
#     state["metrics"]["doc_type"] = state["doc_type"]
#     state["metrics"]["query_type"] = state["query_type"]
#     return state

# # Node 3: Summary path
# def node_summarize(state: DocState) -> DocState:
#     summary_start = time.time()
#     chunks = state["chunks"]
#     doc_type = state["doc_type"]

#     summary = raptor_summarize(chunks, doc_type)
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     state["metrics"]["summary_time_sec"] = round(summary_time, 2)
#     state["metrics"]["summary_length_words"] = len(
#         summary.split())
#     state["metrics"]["parallel_workers"] = min(
#         MAX_WORKERS, len(chunks))
#     state["metrics"]["type"] = "summary"
#     return state

# # Node 4: QA path
# def node_qa(state: DocState) -> DocState:
#     qa_start = time.time()
#     question = state["question"]
#     query_type = state["query_type"]
#     chunks = state["chunks"]

#     # Build or load FAISS index
#     pdf_hash = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(chunks, pdf_hash)

#     # Retrieve relevant chunks
#     k = 8 if query_type in ["MULTIPART_QA",
#                              "REASONING_QA"] else 5
#     retrieved = faiss_index.similarity_search(question, k=k)
#     retrieved_texts = [d.page_content for d in retrieved]

#     # Confidence score
#     confidence = compute_confidence(question, retrieved)

#     answer = ""
#     model_used = ""

#     if query_type == "FACTUAL_QA":
#         # Try RoBERTa first
#         rob_answer, rob_score = roberta_qa(
#             question, retrieved_texts)

#         if rob_score > 0.3 and rob_answer.strip():
#             answer = rob_answer
#             model_used = "roberta"
#             confidence = round(rob_score * 100, 2)
#         else:
#             # Fallback to LLaMA
#             context = "\n\n".join(retrieved_texts)
#             prompt = (
#                 f"Context:\n{context}\n\n"
#                 f"Question: {question}\n\n"
#                 f"Give a short exact answer from "
#                 f"the context only:\n"
#             )
#             answer = call_llama(prompt)
#             model_used = "llama_fallback"

#     elif query_type == "MULTIPART_QA":
#         context = "\n\n".join(retrieved_texts)
#         prompt = MULTIPART_PROMPT.format(
#             context=context,
#             question=question
#         )
#         answer = call_llama(prompt)
#         model_used = "llama"

#     elif query_type == "REASONING_QA":
#         context = "\n\n".join(retrieved_texts)
#         prompt = REASONING_PROMPT.format(
#             context=context,
#             question=question
#         )
#         answer = call_llama(prompt)
#         model_used = "llama"

#     else:
#         # Default factual
#         rob_answer, rob_score = roberta_qa(
#             question, retrieved_texts)
#         answer = rob_answer if rob_answer else "Not found."
#         model_used = "roberta"

#     qa_time = time.time() - qa_start

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."

#     state["answer"] = answer
#     state["metrics"]["qa_time_sec"] = round(qa_time, 2)
#     state["metrics"]["confidence_score"] = confidence
#     state["metrics"]["model_used"] = model_used
#     state["metrics"]["chunks_retrieved"] = k
#     state["metrics"]["type"] = "qa"
#     return state

# # Node 5: Validate output
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry = state.get("retry_count", 0)

#     # Basic quality checks
#     if len(answer.strip()) < 10 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"] = ""
#         return state

#     # ---- FINAL TIMING ----
#     total_time = time.time() - state["start_time"]

#     # ---- TOKENS PER SECOND ----
#     output_words  = len(answer.split())
#     output_tokens = output_words * 1.3
#     tps = round(output_tokens / total_time, 2) if total_time > 0 else 0

#     m = state["metrics"]

#     # ============================================================
#     # V1 METRICS (old - kept for benchmarking comparison)
#     # These match exactly what v1 app.py returned
#     # ============================================================
#     m["response_time_sec"]    = round(total_time, 2)
#     m["extraction_time_sec"]  = m.get("extraction_time_sec", 0)
#     m["pages_processed"]      = state.get("page_count", 0)
#     m["characters_processed"] = state.get("char_count", 0)
#     m["words_processed"]      = len(state.get("extracted_text", "").split())

#     # v1 summary specific
#     if m.get("type") == "summary":
#         m["summary_time_sec"]      = m.get("summary_time_sec", 0)
#         m["summary_length_words"]  = len(answer.split())

#     # v1 QA specific
#     if m.get("type") == "qa":
#         m["qa_time_sec"]       = m.get("qa_time_sec", 0)
#         m["confidence_score"]  = m.get("confidence_score", 0)

#     # ============================================================
#     # V2 METRICS (new - industry standard)
#     # ============================================================
#     m["ttft_sec"]          = round(total_time, 2)   # time to first token
#     m["tps"]               = tps                     # tokens per second
#     m["doc_type"]          = state.get("doc_type", "general")
#     m["query_type"]        = state.get("query_type", "")
#     m["chunks_created"]    = m.get("chunks_created", 0)
#     m["retry_count"]       = retry
#     m["model_used"]        = m.get("model_used", "roberta")

#     # v2 summary specific
#     if m.get("type") == "summary":
#         m["parallel_workers"]  = m.get("parallel_workers", 0)
#         m["map_time_sec"]      = m.get("map_time_sec", 0)
#         m["reduce_time_sec"]   = m.get("reduce_time_sec", 0)

#     # v2 QA specific
#     if m.get("type") == "qa":
#         m["chunks_retrieved"]  = m.get("chunks_retrieved", 0)

#     state["metrics"] = m
#     return state

# # ============================================================
# # LANGGRAPH ROUTER
# # ============================================================
# def route_query(state: DocState) -> str:
#     if state["query_type"] == "FULL_SUMMARY":
#         return "summarize"
#     return "qa"

# def route_validate(state: DocState) -> str:
#     if not state["answer"].strip() and \
#        state.get("retry_count", 0) < 2:
#         return "retry"
#     return "done"

# # ============================================================
# # BUILD LANGGRAPH WORKFLOW
# # ============================================================
# def build_workflow():
#     workflow = StateGraph(DocState)

#     # Add nodes
#     workflow.add_node("extract", node_extract)
#     workflow.add_node("chunk", node_chunk)
#     workflow.add_node("summarize", node_summarize)
#     workflow.add_node("qa", node_qa)
#     workflow.add_node("validate", node_validate)

#     # Entry point
#     workflow.set_entry_point("extract")

#     # Linear flow: extract → chunk → route
#     workflow.add_edge("extract", "chunk")

#     # Conditional routing after chunk
#     workflow.add_conditional_edges(
#         "chunk",
#         route_query,
#         {
#             "summarize": "summarize",
#             "qa": "qa"
#         }
#     )

#     # Both paths go to validate
#     workflow.add_edge("summarize", "validate")
#     workflow.add_edge("qa", "validate")

#     # Validate: done or retry
#     workflow.add_conditional_edges(
#         "validate",
#         route_validate,
#         {
#             "retry": "qa",
#             "done": END
#         }
#     )

#     return workflow.compile()

# # Build workflow once at startup
# print("Building LangGraph workflow...")
# workflow = build_workflow()
# print("Workflow ready!")

# # ============================================================
# # FLASK ROUTE
# # ============================================================
# @app.route("/process", methods=["POST"])
# def process():
#     data = request.json
#     pdf_path = data.get("pdf_path")
#     question = data.get("question")

#     if not pdf_path or not question:
#         return jsonify({"answer": "Missing PDF path or question."})
#     if not os.path.exists(pdf_path):
#         return jsonify({"answer": "File not found."})

#     # Initial state
#     initial_state: DocState = {
#         "pdf_path":       pdf_path,
#         "question":       question,
#         "extracted_text": "",
#         "chunks":         [],
#         "faiss_index":    None,
#         "answer":         "",
#         "metrics":        {},
#         "doc_type":       "general",
#         "query_type":     "",
#         "retry_count":    0,
#         "start_time":     time.time(),
#         "page_count":     0,
#         "char_count":     0
#     }

#     try:
#         # Run LangGraph workflow
#         result = workflow.invoke(initial_state)
#         m = result["metrics"]

#         # ---- V1 METRICS (exact same fields as old app.py benchmark) ----
#         v1 = {
#             "type":                  m.get("type", ""),
#             "response_time_sec":     round(m.get("response_time_sec", 0), 2),
#             "extraction_time_sec":   round(m.get("extraction_time_sec", 0), 2),
#             "pages_processed":       m.get("pages_processed", 0),
#             "characters_processed":  m.get("characters_processed", 0),
#             "words_processed":       m.get("words_processed", 0),
#         }
#         if m.get("type") == "summary":
#             v1["summary_time_sec"]     = round(m.get("summary_time_sec", 0), 2)
#             v1["summary_length_words"] = m.get("summary_length_words", 0)
#         if m.get("type") == "qa":
#             v1["qa_time_sec"]      = round(m.get("qa_time_sec", 0), 2)
#             # confidence_score: store as raw float (0.937)
#             # index.html multiplies by 100 to show as % (93.7%)
#             # same behavior as old app.py
#             v1["confidence_score"] = round(m.get("confidence_score", 0) / 100, 4) \
#                 if m.get("confidence_score", 0) > 1 \
#                 else round(m.get("confidence_score", 0), 4)

#         # ---- V2 METRICS (new fields) ----
#         v2 = {
#             "ttft_sec":        round(m.get("ttft_sec", 0), 2),
#             "tps":             m.get("tps", 0),
#             "doc_type":        m.get("doc_type", "general"),
#             "query_type":      m.get("query_type", ""),
#             "model_used":      m.get("model_used", ""),
#             "chunks_created":  m.get("chunks_created", 0),
#             "retry_count":     m.get("retry_count", 0),
#         }
#         if m.get("type") == "summary":
#             v2["parallel_workers"] = m.get("parallel_workers", 0)
#             v2["map_time_sec"]     = round(m.get("map_time_sec", 0), 2)
#             v2["reduce_time_sec"]  = round(m.get("reduce_time_sec", 0), 2)
#         if m.get("type") == "qa":
#             v2["chunks_retrieved"] = m.get("chunks_retrieved", 0)

#         return jsonify({
#             "answer":     result["answer"],
#             "metrics":    {**v1, **v2},   # merged for frontend
#             "metrics_v1": v1,             # old metrics isolated
#             "metrics_v2": v2,             # new metrics isolated
#         })

#     except Exception as e:
#         return jsonify({
#             "answer":     f"Error: {str(e)}",
#             "metrics":    {},
#             "metrics_v1": {},
#             "metrics_v2": {}
#         })

# # ============================================================
# # RUN
# # ============================================================
# if __name__ == "__main__":
#     print("Flask server running on http://127.0.0.1:5000")
#     app.run(port=5000, threaded=True)

#  











# donotknow if it works or not this is the updateed one for one to 3min  but metrics are new so i am going for another code above same code but only old metrocs there to compare 
# # ============================================================
# # app.py - Agentic Document Intelligence System v2
# # Stack: LangGraph + LLaMA (Ollama) + FAISS RAG + RoBERTa
# # ============================================================

# import warnings
# warnings.filterwarnings("ignore")

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import re
# import io
# import time
# import json
# import hashlib
# import numpy as np
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import TypedDict, Optional

# # PDF + OCR
# import fitz
# import pytesseract
# from PIL import Image

# # LangChain + LangGraph
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# from langgraph.graph import StateGraph, END

# # Ollama (LLaMA)
# import ollama

# # HuggingFace (RoBERTa)
# from transformers import pipeline, logging
# logging.set_verbosity_error()

# import torch

# # ============================================================
# # FLASK SETUP
# # ============================================================
# app = Flask(__name__)
# CORS(app)

# # ============================================================
# # CONFIG
# # ============================================================
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# OLLAMA_MODEL    = "llama3.2"
# CHUNK_SIZE      = 4000       # bigger chunks = fewer LLM calls
# CHUNK_OVERLAP   = 200
# MAX_WORKERS     = 6          # parallel workers
# EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
# FAISS_CACHE_DIR = "faiss_cache"
# os.makedirs(FAISS_CACHE_DIR, exist_ok=True)

# device = 0 if torch.cuda.is_available() else -1

# # ============================================================
# # LOAD ROBERTA (once at startup - small model)
# # ============================================================
# print("Loading RoBERTa QA model...")
# qa_pipeline = pipeline(
#     "question-answering",
#     model="deepset/roberta-base-squad2",
#     device=device
# )
# print("RoBERTa loaded!")

# # ============================================================
# # LOAD EMBEDDINGS (once at startup)
# # ============================================================
# print("Loading embedding model...")
# embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# print("Embeddings loaded!")

# # ============================================================
# # LANGGRAPH STATE
# # ============================================================
# class DocState(TypedDict):
#     pdf_path:         str
#     question:         str
#     extracted_text:   str
#     chunks:           list
#     faiss_index:      object
#     answer:           str
#     metrics:          dict
#     doc_type:         str
#     query_type:       str
#     retry_count:      int
#     start_time:       float
#     page_count:       int
#     char_count:       int

# # ============================================================
# # PROMPTS
# # ============================================================
# SUMMARY_PROMPTS = {
#     "academic": (
#         "You are an academic summarizer. Extract: key concepts, "
#         "methodology, findings, conclusions. Use bullet points. "
#         "Be concise.\n\nText:\n{text}\n\nBullet point summary:"
#     ),
#     "legal": (
#         "You are a legal document analyst. Extract: parties, "
#         "key clauses, obligations, dates. Be precise.\n\n"
#         "Text:\n{text}\n\nKey points:"
#     ),
#     "technical": (
#         "You are a technical writer. Extract: key concepts, "
#         "processes, specifications. Use bullet points.\n\n"
#         "Text:\n{text}\n\nTechnical summary:"
#     ),
#     "general": (
#         "Summarize the key points of this text in clear "
#         "bullet points. Be concise.\n\nText:\n{text}\n\nSummary:"
#     )
# }

# MULTIPART_PROMPT = (
#     "Context:\n{context}\n\n"
#     "Question: {question}\n\n"
#     "List ALL items mentioned in the context that answer "
#     "this question. Number each one. Do not miss any.\n\n"
#     "Complete answer:"
# )

# REASONING_PROMPT = (
#     "Context:\n{context}\n\n"
#     "Question: {question}\n\n"
#     "Think step by step using only the context above. "
#     "Provide a reasoned explanation.\n\n"
#     "Answer:"
# )

# MERGE_PROMPT = (
#     "Merge these summaries into one coherent final summary. "
#     "Remove repetition. Keep all key points. "
#     "Use clear bullet points.\n\n"
#     "Summaries:\n{summaries}\n\n"
#     "Final merged summary:"
# )

# # ============================================================
# # UTILITY: PDF HASH (for FAISS cache)
# # ============================================================
# def get_pdf_hash(pdf_path: str) -> str:
#     h = hashlib.md5()
#     with open(pdf_path, "rb") as f:
#         h.update(f.read())
#     return h.hexdigest()

# # ============================================================
# # UTILITY: FORMAT TIME
# # ============================================================
# def format_time(seconds: float) -> str:
#     if seconds < 60:
#         return f"{seconds:.2f} sec"
#     mins = int(seconds // 60)
#     secs = seconds % 60
#     return f"{mins} min {secs:.2f} sec"

# # ============================================================
# # UTILITY: DETECT QUERY TYPE
# # ============================================================
# def detect_query_type(question: str) -> str:
#     q = question.lower().strip()

#     # Summary triggers
#     summary_keywords = ["summary", "summarize", "overview",
#                         "brief", "outline", "gist", "tldr"]
#     if any(k in q for k in summary_keywords):
#         return "FULL_SUMMARY"

#     # Multi-part triggers
#     multipart_keywords = ["list", "all", "every", "what are",
#                           "how many", "enumerate", "types of",
#                           "examples of", "mention"]
#     if any(k in q for k in multipart_keywords):
#         return "MULTIPART_QA"

#     # Reasoning triggers
#     reasoning_keywords = ["why", "how does", "explain",
#                           "what would", "compare", "difference",
#                           "relationship", "impact", "effect"]
#     if any(k in q for k in reasoning_keywords):
#         return "REASONING_QA"

#     # Default: factual
#     return "FACTUAL_QA"

# # ============================================================
# # UTILITY: DETECT DOCUMENT TYPE
# # ============================================================
# def detect_doc_type(text: str) -> str:
#     sample = text[:1000].lower()
#     if any(w in sample for w in ["abstract", "methodology",
#            "conclusion", "research", "university", "lecture"]):
#         return "academic"
#     if any(w in sample for w in ["whereas", "clause", "agreement",
#            "party", "hereby", "legal", "contract"]):
#         return "legal"
#     if any(w in sample for w in ["api", "function", "install",
#            "configure", "technical", "specification"]):
#         return "technical"
#     return "general"

# # ============================================================
# # CORE: EXTRACT SINGLE PAGE
# # ============================================================
# def extract_page(args):
#     page, page_num = args
#     try:
#         text = page.get_text().strip()
#         if text:
#             return page_num, text
#         # OCR fallback for scanned pages
#         pix = page.get_pixmap(dpi=200)
#         img = Image.frombytes("RGB",
#               [pix.width, pix.height], pix.samples)
#         ocr_text = pytesseract.image_to_string(img).strip()
#         return page_num, ocr_text
#     except Exception:
#         return page_num, ""

# # ============================================================
# # CORE: PARALLEL PDF EXTRACTION
# # ============================================================
# def extract_pdf_parallel(pdf_path: str):
#     doc = fitz.open(pdf_path)
#     page_count = len(doc)
#     pages = [(doc[i], i) for i in range(page_count)]

#     results = {}
#     workers = min(MAX_WORKERS, page_count)

#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         futures = {ex.submit(extract_page, p): p[1] for p in pages}
#         for future in as_completed(futures):
#             page_num, text = future.result()
#             results[page_num] = text

#     doc.close()

#     # Join in page order
#     full_text = " ".join(
#         results[i] for i in sorted(results.keys())
#     )
#     full_text = re.sub(r'\s+', ' ', full_text).strip()
#     return full_text, page_count

# # ============================================================
# # CORE: SEMANTIC CHUNKING
# # ============================================================
# def semantic_chunk(text: str) -> list:
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""]
#     )
#     return splitter.split_text(text)

# # ============================================================
# # CORE: LLaMA CALL (via Ollama)
# # ============================================================
# def call_llama(prompt: str) -> str:
#     try:
#         response = ollama.chat(
#             model=OLLAMA_MODEL,
#             messages=[{"role": "user", "content": prompt}]
#         )
#         return response['message']['content'].strip()
#     except Exception as e:
#         return f"LLaMA error: {str(e)}"

# # ============================================================
# # CORE: SUMMARIZE SINGLE CHUNK (for parallel map)
# # ============================================================
# def summarize_chunk(args):
#     chunk, doc_type = args
#     prompt_template = SUMMARY_PROMPTS.get(doc_type,
#                       SUMMARY_PROMPTS["general"])
#     prompt = prompt_template.format(text=chunk[:3000])
#     return call_llama(prompt)

# # ============================================================
# # CORE: RAPTOR HIERARCHICAL SUMMARY
# # ============================================================
# def raptor_summarize(chunks: list, doc_type: str) -> str:

#     if not chunks:
#         return "No content to summarize."

#     # ---- MAP PHASE: parallel chunk summarization ----
#     map_start = time.time()
#     workers = min(MAX_WORKERS, len(chunks))
#     args = [(chunk, doc_type) for chunk in chunks]
#     mini_summaries = []

#     with ThreadPoolExecutor(max_workers=workers) as ex:
#         futures = list(ex.map(summarize_chunk, args))
#         mini_summaries = [f for f in futures if f and
#                          "error" not in f.lower()]

#     map_time = time.time() - map_start

#     if not mini_summaries:
#         return "Could not generate summary."

#     # ---- REDUCE PHASE: hierarchical merge ----
#     reduce_start = time.time()

#     # Group mini summaries into batches of 6
#     batch_size = 6
#     current_level = mini_summaries

#     while len(current_level) > 1:
#         next_level = []
#         for i in range(0, len(current_level), batch_size):
#             batch = current_level[i:i + batch_size]
#             if len(batch) == 1:
#                 next_level.append(batch[0])
#                 continue
#             combined = "\n\n---\n\n".join(batch)
#             prompt = MERGE_PROMPT.format(summaries=combined)
#             merged = call_llama(prompt)
#             next_level.append(merged)
#         current_level = next_level

#     reduce_time = time.time() - reduce_start
#     final_summary = current_level[0]

#     return final_summary

# # ============================================================
# # CORE: BUILD FAISS INDEX
# # ============================================================
# def build_faiss_index(chunks: list, pdf_hash: str):
#     cache_path = os.path.join(FAISS_CACHE_DIR, pdf_hash)

#     # Try loading from cache first
#     if os.path.exists(cache_path):
#         try:
#             return FAISS.load_local(
#                 cache_path,
#                 embedding_model,
#                 allow_dangerous_deserialization=True
#             )
#         except Exception:
#             pass

#     # Build new index
#     docs = [Document(page_content=c,
#             metadata={"chunk_id": i})
#             for i, c in enumerate(chunks)]

#     index = FAISS.from_documents(docs, embedding_model)

#     # Save to cache
#     try:
#         index.save_local(cache_path)
#     except Exception:
#         pass

#     return index

# # ============================================================
# # CORE: COMPUTE CONFIDENCE
# # ============================================================
# def compute_confidence(question: str, docs: list) -> float:
#     try:
#         q_embed = embedding_model.embed_query(question)
#         sims = []
#         for d in docs:
#             d_embed = embedding_model.embed_query(
#                 d.page_content[:300])
#             sim = np.dot(q_embed, d_embed) / (
#                 np.linalg.norm(q_embed) *
#                 np.linalg.norm(d_embed) + 1e-8
#             )
#             sims.append(sim)
#         return round(float(np.mean(sims)) * 100, 2)
#     except Exception:
#         return 0.0

# # ============================================================
# # CORE: ROBERTA QA
# # ============================================================
# def roberta_qa(question: str, chunks: list):
#     best_answer = ""
#     best_score = 0.0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(
#                 question=question,
#                 context=chunk[:2000]
#             )
#             if result['score'] > best_score:
#                 best_score = result['score']
#                 best_answer = result['answer']
#         except Exception:
#             continue

#     return best_answer, round(best_score, 4)

# # ============================================================
# # LANGGRAPH NODES
# # ============================================================

# # Node 1: Extract PDF
# def node_extract(state: DocState) -> DocState:
#     extract_start = time.time()
#     text, page_count = extract_pdf_parallel(state["pdf_path"])
#     extract_time = time.time() - extract_start

#     state["extracted_text"] = text
#     state["page_count"] = page_count
#     state["char_count"] = len(text)
#     state["metrics"]["extraction_time_sec"] = round(
#         extract_time, 2)
#     state["metrics"]["pages_processed"] = page_count
#     state["metrics"]["characters_processed"] = len(text)
#     state["metrics"]["words_processed"] = len(text.split())
#     return state

# # Node 2: Detect type + chunk
# def node_chunk(state: DocState) -> DocState:
#     text = state["extracted_text"]
#     state["doc_type"] = detect_doc_type(text)
#     state["query_type"] = detect_query_type(state["question"])
#     state["chunks"] = semantic_chunk(text)
#     state["metrics"]["chunks_created"] = len(state["chunks"])
#     state["metrics"]["doc_type"] = state["doc_type"]
#     state["metrics"]["query_type"] = state["query_type"]
#     return state

# # Node 3: Summary path
# def node_summarize(state: DocState) -> DocState:
#     summary_start = time.time()
#     chunks = state["chunks"]
#     doc_type = state["doc_type"]

#     summary = raptor_summarize(chunks, doc_type)
#     summary_time = time.time() - summary_start

#     state["answer"] = summary
#     state["metrics"]["summary_time_sec"] = round(summary_time, 2)
#     state["metrics"]["summary_length_words"] = len(
#         summary.split())
#     state["metrics"]["parallel_workers"] = min(
#         MAX_WORKERS, len(chunks))
#     state["metrics"]["type"] = "summary"
#     return state

# # Node 4: QA path
# def node_qa(state: DocState) -> DocState:
#     qa_start = time.time()
#     question = state["question"]
#     query_type = state["query_type"]
#     chunks = state["chunks"]

#     # Build or load FAISS index
#     pdf_hash = get_pdf_hash(state["pdf_path"])
#     faiss_index = build_faiss_index(chunks, pdf_hash)

#     # Retrieve relevant chunks
#     k = 8 if query_type in ["MULTIPART_QA",
#                              "REASONING_QA"] else 5
#     retrieved = faiss_index.similarity_search(question, k=k)
#     retrieved_texts = [d.page_content for d in retrieved]

#     # Confidence score
#     confidence = compute_confidence(question, retrieved)

#     answer = ""
#     model_used = ""

#     if query_type == "FACTUAL_QA":
#         # Try RoBERTa first
#         rob_answer, rob_score = roberta_qa(
#             question, retrieved_texts)

#         if rob_score > 0.3 and rob_answer.strip():
#             answer = rob_answer
#             model_used = "roberta"
#             confidence = round(rob_score * 100, 2)
#         else:
#             # Fallback to LLaMA
#             context = "\n\n".join(retrieved_texts)
#             prompt = (
#                 f"Context:\n{context}\n\n"
#                 f"Question: {question}\n\n"
#                 f"Give a short exact answer from "
#                 f"the context only:\n"
#             )
#             answer = call_llama(prompt)
#             model_used = "llama_fallback"

#     elif query_type == "MULTIPART_QA":
#         context = "\n\n".join(retrieved_texts)
#         prompt = MULTIPART_PROMPT.format(
#             context=context,
#             question=question
#         )
#         answer = call_llama(prompt)
#         model_used = "llama"

#     elif query_type == "REASONING_QA":
#         context = "\n\n".join(retrieved_texts)
#         prompt = REASONING_PROMPT.format(
#             context=context,
#             question=question
#         )
#         answer = call_llama(prompt)
#         model_used = "llama"

#     else:
#         # Default factual
#         rob_answer, rob_score = roberta_qa(
#             question, retrieved_texts)
#         answer = rob_answer if rob_answer else "Not found."
#         model_used = "roberta"

#     qa_time = time.time() - qa_start

#     if not answer.strip():
#         answer = "Could not find a relevant answer in the PDF."

#     state["answer"] = answer
#     state["metrics"]["qa_time_sec"] = round(qa_time, 2)
#     state["metrics"]["confidence_score"] = confidence
#     state["metrics"]["model_used"] = model_used
#     state["metrics"]["chunks_retrieved"] = k
#     state["metrics"]["type"] = "qa"
#     return state

# # Node 5: Validate output
# def node_validate(state: DocState) -> DocState:
#     answer = state["answer"]
#     retry = state.get("retry_count", 0)

#     # Basic quality checks
#     if len(answer.strip()) < 10 and retry < 2:
#         state["retry_count"] = retry + 1
#         state["answer"] = ""
#         return state

#     # Calculate final metrics
#     total_time = time.time() - state["start_time"]
#     state["metrics"]["response_time_sec"] = round(total_time, 2)
#     state["metrics"]["ttft_sec"] = round(total_time, 2)

#     # Tokens per second estimate
#     output_words = len(answer.split())
#     output_tokens = output_words * 1.3
#     if total_time > 0:
#         state["metrics"]["tps"] = round(
#             output_tokens / total_time, 2)

#     return state

# # ============================================================
# # LANGGRAPH ROUTER
# # ============================================================
# def route_query(state: DocState) -> str:
#     if state["query_type"] == "FULL_SUMMARY":
#         return "summarize"
#     return "qa"

# def route_validate(state: DocState) -> str:
#     if not state["answer"].strip() and \
#        state.get("retry_count", 0) < 2:
#         return "retry"
#     return "done"

# # ============================================================
# # BUILD LANGGRAPH WORKFLOW
# # ============================================================
# def build_workflow():
#     workflow = StateGraph(DocState)

#     # Add nodes
#     workflow.add_node("extract", node_extract)
#     workflow.add_node("chunk", node_chunk)
#     workflow.add_node("summarize", node_summarize)
#     workflow.add_node("qa", node_qa)
#     workflow.add_node("validate", node_validate)

#     # Entry point
#     workflow.set_entry_point("extract")

#     # Linear flow: extract → chunk → route
#     workflow.add_edge("extract", "chunk")

#     # Conditional routing after chunk
#     workflow.add_conditional_edges(
#         "chunk",
#         route_query,
#         {
#             "summarize": "summarize",
#             "qa": "qa"
#         }
#     )

#     # Both paths go to validate
#     workflow.add_edge("summarize", "validate")
#     workflow.add_edge("qa", "validate")

#     # Validate: done or retry
#     workflow.add_conditional_edges(
#         "validate",
#         route_validate,
#         {
#             "retry": "qa",
#             "done": END
#         }
#     )

#     return workflow.compile()

# # Build workflow once at startup
# print("Building LangGraph workflow...")
# workflow = build_workflow()
# print("Workflow ready!")

# # ============================================================
# # FLASK ROUTE
# # ============================================================
# @app.route("/process", methods=["POST"])
# def process():
#     data = request.json
#     pdf_path = data.get("pdf_path")
#     question = data.get("question")

#     if not pdf_path or not question:
#         return jsonify({"answer": "Missing PDF path or question."})
#     if not os.path.exists(pdf_path):
#         return jsonify({"answer": "File not found."})

#     # Initial state
#     initial_state: DocState = {
#         "pdf_path":       pdf_path,
#         "question":       question,
#         "extracted_text": "",
#         "chunks":         [],
#         "faiss_index":    None,
#         "answer":         "",
#         "metrics":        {},
#         "doc_type":       "general",
#         "query_type":     "",
#         "retry_count":    0,
#         "start_time":     time.time(),
#         "page_count":     0,
#         "char_count":     0
#     }

#     try:
#         # Run LangGraph workflow
#         result = workflow.invoke(initial_state)

#         # Format times in metrics
#         metrics = result["metrics"]
#         formatted = {}
#         for k, v in metrics.items():
#             if k.endswith("_sec") and isinstance(v, float):
#                 formatted[k] = format_time(v)
#             else:
#                 formatted[k] = v

#         return jsonify({
#             "answer":  result["answer"],
#             "metrics": formatted
#         })

#     except Exception as e:
#         return jsonify({
#             "answer": f"Error: {str(e)}",
#             "metrics": {}
#         })

# # ============================================================
# # RUN
# # ============================================================
# if __name__ == "__main__":
#     print("Flask server running on http://127.0.0.1:5000")
#     app.run(port=5000, threaded=True)










# without caching 

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import torch
# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import io
# import re
# import time
# from transformers import pipeline, logging

# logging.set_verbosity_error()  # suppress warnings

# app = Flask(__name__)
# CORS(app)

# # Set pytesseract path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Use GPU if available
# device = 0 if torch.cuda.is_available() else -1

# print("Loading models... ")

# # Summarization pipeline
# summarizer = pipeline(
#     "summarization",
#     model="sshleifer/distilbart-cnn-12-6",
#     device=device
# )

# # QA pipeline
# qa_pipeline = pipeline(
#     "question-answering",
#     model="deepset/roberta-base-squad2",
#     device=device
# )

# print("Models loaded!")


# # ---------------- PDF TEXT EXTRACTION ----------------
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""
#     page_count = len(doc)

#     for page in doc:
#         page_text = page.get_text()
#         if page_text.strip():
#             text += page_text
#         else:
#             # OCR fallback for scanned pages
#             pix = page.get_pixmap()
#             img = Image.open(io.BytesIO(pix.tobytes()))
#             ocr_text = pytesseract.image_to_string(img)
#             text += ocr_text

#     doc.close()
#     text = re.sub(r'\s+', ' ', text)  # Clean OCR noise
#     return text.strip(), page_count


# # ---------------- SUMMARIZATION ----------------
# def summarize_text(text):
#     if len(text.strip()) == 0:
#         return "PDF has no readable text."

#     chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
#     summarized = ""

#     for chunk in chunks:
#         summary = summarizer(
#             chunk,
#             max_length=150,
#             min_length=30,
#             do_sample=False
#         )[0]['summary_text']
#         summarized += summary + " "

#     return summarized.strip()


# # ---------------- QUESTION ANSWERING ----------------
# def answer_question(context, question):
#     chunks = [context[i:i + 2000] for i in range(0, len(context), 2000)]
#     best_answer = ""
#     best_score = 0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(question=question, context=chunk)
#             if result['score'] > best_score:
#                 best_score = result['score']
#                 best_answer = result['answer']
#         except:
#             continue

#     if best_answer.strip() == "":
#         return "Could not find a relevant answer in the PDF.", 0.0

#     return best_answer, round(best_score, 4)


# # ---------------- FLASK ROUTE ----------------
# @app.route("/process", methods=["POST"])
# def process():
#     data = request.json
#     pdf_path = data.get("pdf_path")
#     question = data.get("question")

#     if not pdf_path or not question:
#         return jsonify({"answer": "Missing PDF path or question."})
#     if not os.path.exists(pdf_path):
#         return jsonify({"answer": "File not found."})

#     # --- Start total timer ---
#     total_start = time.time()

#     # --- Extract text + page count ---
#     extract_start = time.time()
#     context, page_count = extract_text_from_pdf(pdf_path)
#     extract_time = round(time.time() - extract_start, 2)

#     char_count = len(context)
#     word_count = len(context.split())

#     # --- Summary request ---
#     if "summary" in question.lower():
#         summary_start = time.time()
#         summary = summarize_text(context)
#         summary_time = round(time.time() - summary_start, 2)
#         total_time = round(time.time() - total_start, 2)
#         summary_word_count = len(summary.split())

#         return jsonify({
#             "answer": summary,
#             "metrics": {
#                 "type": "summary",
#                 "response_time_sec": total_time,
#                 "extraction_time_sec": extract_time,
#                 "summary_time_sec": summary_time,
#                 "summary_length_words": summary_word_count,
#                 "pages_processed": page_count,
#                 "characters_processed": char_count,
#                 "words_processed": word_count
#             }
#         })

#     # --- Normal QA ---
#     qa_start = time.time()
#     answer, confidence = answer_question(context, question)
#     qa_time = round(time.time() - qa_start, 2)
#     total_time = round(time.time() - total_start, 2)

#     return jsonify({
#         "answer": answer,
#         "metrics": {
#             "type": "qa",
#             "response_time_sec": total_time,
#             "extraction_time_sec": extract_time,
#             "qa_time_sec": qa_time,
#             "confidence_score": confidence,
#             "pages_processed": page_count,
#             "characters_processed": char_count,
#             "words_processed": word_count
#         }
#     })


# # ---------------- RUN APP ----------------
# if __name__ == "__main__":
#     print("Flask server running on http://127.0.0.1:5000")
#     app.run(port=5000)







# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# import torch
# import fitz  # PyMuPDF
# import pytesseract
# from PIL import Image
# import io
# import re
# import hashlib
# import json
# from transformers import pipeline, logging

# logging.set_verbosity_error()  # suppress warnings

# app = Flask(__name__)
# CORS(app)

# # Set pytesseract path (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# # Use GPU if available
# device = 0 if torch.cuda.is_available() else -1

# print("Loading models... ⏳")
# # Summarization pipeline
# summarizer = pipeline(
#     "summarization",
#     model="sshleifer/distilbart-cnn-12-6",
#     device=device
# )

# # QA pipeline with better model for long passages
# qa_pipeline = pipeline(
#     "question-answering",
#     model="deepset/roberta-base-squad2",
#     device=device
# )
# print("Models loaded ✅")

# # ---------------- CACHE SETUP ----------------
# CACHE_DIR = "cache"
# os.makedirs(CACHE_DIR, exist_ok=True)

# def get_pdf_hash(pdf_path):
#     """Generate a SHA256 hash of the PDF content."""
#     h = hashlib.sha256()
#     with open(pdf_path, "rb") as f:
#         while chunk := f.read(8192):
#             h.update(chunk)
#     return h.hexdigest()

# def get_cached_summary(pdf_hash):
#     """Retrieve cached summary if exists."""
#     cache_file = os.path.join(CACHE_DIR, f"{pdf_hash}.json")
#     if os.path.exists(cache_file):
#         with open(cache_file, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         return data.get("summary")
#     return None

# def cache_summary(pdf_hash, summary_text):
#     """Save summary to cache."""
#     cache_file = os.path.join(CACHE_DIR, f"{pdf_hash}.json")
#     with open(cache_file, "w", encoding="utf-8") as f:
#         json.dump({"summary": summary_text}, f, ensure_ascii=False)

# # ---------------- PDF TEXT EXTRACTION ----------------
# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     text = ""

#     for page in doc:
#         page_text = page.get_text()
#         if page_text.strip():
#             text += page_text
#         else:
#             # OCR fallback for scanned pages
#             pix = page.get_pixmap()
#             img = Image.open(io.BytesIO(pix.tobytes()))
#             ocr_text = pytesseract.image_to_string(img)
#             text += ocr_text

#     doc.close()
#     text = re.sub(r'\s+', ' ', text)  # Clean OCR noise
#     return text.strip()

# # ---------------- SUMMARIZATION ----------------
# def summarize_text(text):
#     if len(text.strip()) == 0:
#         return "❌ PDF has no readable text."

#     chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
#     summarized = ""
#     for chunk in chunks:
#         summary = summarizer(
#             chunk,
#             max_length=150,
#             min_length=30,
#             do_sample=False
#         )[0]['summary_text']
#         summarized += summary + " "
#     return summarized.strip()

# # ---------------- QUESTION ANSWERING ----------------
# def answer_question(context, question):
#     chunks = [context[i:i + 2000] for i in range(0, len(context), 2000)]
#     best_answer = ""
#     best_score = 0

#     for chunk in chunks:
#         try:
#             result = qa_pipeline(question=question, context=chunk)
#             if result['score'] > best_score:
#                 best_score = result['score']
#                 best_answer = result['answer']
#         except:
#             continue

#     if best_answer.strip() == "":
#         return "❌ Could not find a relevant answer in the PDF."
#     return best_answer

# # ---------------- FLASK ROUTE ----------------
# @app.route("/process", methods=["POST"])
# def process():
#     data = request.json
#     pdf_path = data.get("pdf_path")
#     question = data.get("question")

#     if not pdf_path or not question:
#         return jsonify({"answer": "❌ Missing PDF path or question."})
#     if not os.path.exists(pdf_path):
#         return jsonify({"answer": "❌ File not found."})

#     pdf_hash = get_pdf_hash(pdf_path)

#     # If asking for summary, try cache first
#     if "summary" in question.lower():
#         cached = get_cached_summary(pdf_hash)
#         if cached:
#             return jsonify({"answer": cached})  # Return cached summary

        
#         context = extract_text_from_pdf(pdf_path)
#         summary = summarize_text(context)
#         cache_summary(pdf_hash, summary)
#         return jsonify({"answer": summary})

   
#     context = extract_text_from_pdf(pdf_path)
#     answer = answer_question(context, question)
#     return jsonify({"answer": answer})


# if __name__ == "__main__":
#     print("Flask server running on http://127.0.0.1:5000")
#     app.run(port=5000)


