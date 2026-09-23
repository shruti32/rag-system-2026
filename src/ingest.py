"""
ingest.py — Week 5 Hands-On: PDF → Chunks → Embeddings → Qdrant
----------------------------------------------------------------
Pipeline:
  1. Load PDFs from data/papers/
  2. Split into chunks with RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
  3. Embed each chunk with all-MiniLM-L6-v2 (free, runs locally, 384 dimensions)
  4. Upsert vectors + metadata into a Qdrant collection

Why all-MiniLM-L6-v2?
  - Free, no API key needed
  - Fast (runs on CPU)
  - Good quality for retrieval tasks (MTEB benchmark top performer for its size)

Why Qdrant local mode?
  - No server needed for development — QdrantClient(path=...) persists to disk
  - Same client API as the cloud/server version — easy to upgrade later

Usage:
    cd rag-system-2026
    poetry run python src/ingest.py

    # Ingest a specific folder:
    poetry run python src/ingest.py --papers-dir data/papers

Output:
    Qdrant collection persisted to data/qdrant_db/
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

COLLECTION_NAME  = "ml-papers"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # 384-dim, MIT license
EMBEDDING_DIM    = 384
CHUNK_SIZE       = 1000   # characters — larger chunks keep full explanations intact
CHUNK_OVERLAP    = 150     # ~15% overlap so context isn't lost at boundaries
MIN_CHUNK_CHARS  = 200     # drop tiny fragments (page headers, stray lines)

BASE_DIR    = pathlib.Path(__file__).parents[1]
PAPERS_DIR  = BASE_DIR / "data" / "papers"
QDRANT_PATH = BASE_DIR / "data" / "qdrant_db"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_pdf(path: pathlib.Path) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(str(path))
    pages  = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks.
    Returns list of dicts: {text, source, chunk_id, char_start}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.create_documents([text], metadatas=[{"source": source}])

    out = []
    idx = 0
    for c in chunks:
        text_clean = c.page_content.strip()

        # Skip tiny fragments (headers, stray lines)
        if len(text_clean) < MIN_CHUNK_CHARS:
            continue

        # Skip likely reference-list chunks: dense with "et al." / years / arXiv ids
        lowered = text_clean.lower()
        ref_signals = lowered.count("et al.") + lowered.count("arxiv:") + lowered.count("proceedings")
        if ref_signals >= 3:
            continue

        out.append({"text": text_clean, "source": source, "chunk_index": idx})
        idx += 1

    return out


def stable_id(source: str, chunk_index: int) -> str:
    """Generate a deterministic UUID from source + chunk_index."""
    key = f"{source}::{chunk_index}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


# ── Core pipeline ─────────────────────────────────────────────────────────────

def ingest(papers_dir: pathlib.Path = PAPERS_DIR) -> None:
    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {papers_dir}. Add some PDFs and retry.")
        print("Tip: download ML papers from https://arxiv.org/")
        return

    print(f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}")

    # 1. Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # 2. Connect to Qdrant (local disk mode)
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(QDRANT_PATH))

    # 3. Create collection if it doesn't exist
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Using existing collection: {COLLECTION_NAME}")

    # 4. Process each PDF
    total_chunks = 0
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        text   = load_pdf(pdf_path)
        chunks = chunk_text(text, source=pdf_path.name)
        print(f"  → {len(chunks)} chunks")

        # Embed in batches of 64
        batch_size = 64
        points = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c["text"] for c in batch]
            embeddings = model.encode(texts, show_progress_bar=False)

            for chunk, vector in zip(batch, embeddings):
                points.append(PointStruct(
                    id=stable_id(chunk["source"], chunk["chunk_index"]),
                    vector=vector.tolist(),
                    payload={
                        "text":        chunk["text"],
                        "source":      chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                    },
                ))

        # Upsert (insert or update if ID already exists)
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  → Upserted {len(points)} vectors into Qdrant")
        total_chunks += len(points)

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nDone. Total vectors in collection: {count}")
    print(f"Qdrant DB persisted to: {QDRANT_PATH}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into Qdrant")
    parser.add_argument("--papers-dir", type=pathlib.Path, default=PAPERS_DIR,
                        help="Directory containing PDF files")
    args = parser.parse_args()
    ingest(papers_dir=args.papers_dir)
