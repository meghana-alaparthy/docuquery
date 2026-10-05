"""HTTP API for docuquery.

Run with:
    uvicorn docuquery.api:app --reload
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from docuquery import config
from docuquery.ask import extractive_answer
from docuquery.embeddings import Embedder
from docuquery.ingest import build_chunks
from docuquery.store import DocStore

app = FastAPI(title="docuquery", description="Local retrieval-augmented Q&A over your documents.")

_embedder = None
_store = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = Embedder(config.MODEL_NAME)
    return _embedder


def get_store():
    global _store
    if _store is None:
        emb = get_embedder()
        if config.INDEX_PATH.exists():
            _store = DocStore.load(emb.dim, config.INDEX_PATH, config.META_PATH)
        else:
            _store = DocStore(emb.dim, config.INDEX_PATH, config.META_PATH)
    return _store


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    """Upload .txt/.md/.pdf files; their chunks are appended to the index."""
    global _store
    tmp = Path(tempfile.mkdtemp())
    try:
        saved = []
        for f in files:
            dest = tmp / Path(f.filename).name
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved.append(dest.name)

        chunks = build_chunks(tmp)
        if not chunks:
            return {"indexed": 0, "detail": "no readable .txt/.md/.pdf content found"}

        emb = get_embedder()
        store = get_store()
        if store.dim != emb.dim:  # embedding model changed since the index was built
            store = DocStore(emb.dim, config.INDEX_PATH, config.META_PATH)
            _store = store
        store.add(chunks, emb.encode([c["text"] for c in chunks]))
        store.save()
        return {"indexed": len(chunks), "files": saved}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.post("/ask")
def ask(req: AskRequest):
    emb = get_embedder()
    passages = get_store().search(emb.encode_query(req.question), req.top_k)
    return {
        "question": req.question,
        "answer": extractive_answer(req.question, passages),
        "passages": passages,
    }
