from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import hashlib
import json
from transformers import pipeline, logging

logging.set_verbosity_error()  # suppress warnings

app = Flask(__name__)
CORS(app)

# Set pytesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

print("Loading models... ⏳")
# Summarization pipeline
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device
)

# QA pipeline with better model for long passages
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2",
    device=device
)
print("Models loaded ✅")

# ---------------- CACHE SETUP ----------------
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_pdf_hash(pdf_path):
    """Generate a SHA256 hash of the PDF content."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_cached_summary(pdf_hash):
    """Retrieve cached summary if exists."""
    cache_file = os.path.join(CACHE_DIR, f"{pdf_hash}.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("summary")
    return None

def cache_summary(pdf_hash, summary_text):
    """Save summary to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{pdf_hash}.json")
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_text}, f, ensure_ascii=False)

# ---------------- PDF TEXT EXTRACTION ----------------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            text += page_text
        else:
            # OCR fallback for scanned pages
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text

    doc.close()
    text = re.sub(r'\s+', ' ', text)  # Clean OCR noise
    return text.strip()

# ---------------- SUMMARIZATION ----------------
def summarize_text(text):
    if len(text.strip()) == 0:
        return "❌ PDF has no readable text."

    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
    summarized = ""
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=150,
            min_length=30,
            do_sample=False
        )[0]['summary_text']
        summarized += summary + " "
    return summarized.strip()

# ---------------- QUESTION ANSWERING ----------------
def answer_question(context, question):
    chunks = [context[i:i + 2000] for i in range(0, len(context), 2000)]
    best_answer = ""
    best_score = 0

    for chunk in chunks:
        try:
            result = qa_pipeline(question=question, context=chunk)
            if result['score'] > best_score:
                best_score = result['score']
                best_answer = result['answer']
        except:
            continue

    if best_answer.strip() == "":
        return "❌ Could not find a relevant answer in the PDF."
    return best_answer

# ---------------- FLASK ROUTE ----------------
@app.route("/process", methods=["POST"])
def process():
    data = request.json
    pdf_path = data.get("pdf_path")
    question = data.get("question")

    if not pdf_path or not question:
        return jsonify({"answer": "❌ Missing PDF path or question."})
    if not os.path.exists(pdf_path):
        return jsonify({"answer": "❌ File not found."})

    pdf_hash = get_pdf_hash(pdf_path)

    # If asking for summary, try cache first
    if "summary" in question.lower():
        cached = get_cached_summary(pdf_hash)
        if cached:
            return jsonify({"answer": cached})  # Return cached summary

        # Not cached → extract, summarize, cache
        context = extract_text_from_pdf(pdf_path)
        summary = summarize_text(context)
        cache_summary(pdf_hash, summary)
        return jsonify({"answer": summary})

    # For normal questions
    context = extract_text_from_pdf(pdf_path)
    answer = answer_question(context, question)
    return jsonify({"answer": answer})

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    print("Flask server running on http://127.0.0.1:5000")
    app.run(port=5000)


