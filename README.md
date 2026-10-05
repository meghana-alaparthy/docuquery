# docuquery

Ask questions over your own documents and get answers with sources. Everything runs on your machine — no API keys, no cloud bills, no data leaving your laptop.

I built this because I kept losing answers inside long PDFs and wiki pages. Ctrl+F only finds exact words; I wanted to ask a question in plain English and land on the right paragraph. So: drop your docs in a folder, build an index once, then ask away.

## How it works

Documents are split into overlapping chunks, each chunk is turned into a vector with a sentence-transformer model, and the vectors go into a FAISS index. When you ask a question, it gets embedded the same way, the index returns the closest chunks by cosine similarity, and the best-matching sentences are stitched into a short extractive answer. No LLM involved — the answer is always quoted from your own docs, so it can't hallucinate.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# build the index from the sample docs
# (first run downloads the embedding model, ~90 MB)
PYTHONPATH=src python -m docuquery.ingest sample_docs

# ask something
PYTHONPATH=src python -m docuquery.ask "how many remote days per week are allowed?"

# or run the API instead
PYTHONPATH=src uvicorn docuquery.api:app --reload
# interactive docs at http://127.0.0.1:8000/docs
```

## API

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET    | `/health` | liveness check |
| POST   | `/ingest` | upload `.txt` / `.md` / `.pdf` files; their chunks get added to the index |
| POST   | `/ask`    | `{"question": "...", "top_k": 5}` → short answer plus ranked passages with source file names and scores |

## Project structure

```
docuquery/
├── sample_docs/            # example docs so the quickstart works out of the box
├── src/docuquery/
│   ├── config.py           # chunk size, model name, paths — env-overridable
│   ├── chunking.py         # sentence-aware overlapping chunker
│   ├── embeddings.py       # lazy-loaded sentence-transformer wrapper
│   ├── store.py            # FAISS index + JSON metadata sidecar
│   ├── ingest.py           # CLI: build the index from a folder of docs
│   ├── ask.py              # CLI: retrieve passages, print ranked sources + extractive answer
│   └── api.py              # FastAPI: /health, /ingest, /ask
├── requirements.txt
└── README.md
```

## Tech stack

Python 3.10+, sentence-transformers (`all-MiniLM-L6-v2`), FAISS (`IndexFlatIP` on normalized vectors = cosine similarity), FastAPI + uvicorn, pypdf.

## Configuration

All settings have sane defaults and can be overridden with environment variables — see `.env.example`.

## License

MIT — see [LICENSE](LICENSE).
