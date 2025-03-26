from fastapi import FastAPI, Query
from typing import Optional
import json
import re
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Load and chunk markdown transcript file
import os

print("📂 Current working directory:", os.getcwd())
print("📄 Files in directory:", os.listdir())

try:
    with open("criminal_law_class_transcripts.md", "r", encoding="utf-8") as f:
        transcript_text = f.read()
    print("✅ Loaded transcript markdown file successfully.")
except FileNotFoundError:
    print("🚨 File not found: criminal_law_class_transcripts.md")
    transcript_text = "## Class 0 – Placeholder\nThis is a fallback transcript used when the file is missing."

transcript_chunks = transcript_text.split("\n# Class")  # Uses Markdown headers to split
transcript_chunks = [chunk.strip() for chunk in transcript_chunks if chunk.strip()]

model = SentenceTransformer('all-MiniLM-L6-v2')
transcript_embeddings = model.encode(transcript_chunks, convert_to_tensor=True)

@app.get("/")
def read_root():
    return {"message": "Transcript Search API is up and running."}

@app.get("/search_transcripts")
def search_transcripts(query: str = Query(...), top_k: int = 3):
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Parse classNumber + topic from header
    results = []
    for chunk in transcript_chunks:
        match = re.search(r"(?:Class\s*)?(\d+)\s*[–\-]\s*(.+)", chunk, re.IGNORECASE)
        class_num = int(match.group(1)) if match else None
        topic = match.group(2).strip() if match else "Unknown"

        results.append({
            "chunk": chunk,
            "classNumber": class_num,
            "topic": topic
        })

    texts = [r["chunk"] for r in results]
    embeddings = model.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embedding, embeddings)[0]
    top = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)[:top_k]

    return {
        "query": query,
        "results": [
            {
                "score": float(score),
                "classNumber": r["classNumber"],
                "topic": r["topic"],
                "excerpt": r["chunk"][:500].replace("\n", " ") + "..."
            }
            for score, r in top
        ]
    }
