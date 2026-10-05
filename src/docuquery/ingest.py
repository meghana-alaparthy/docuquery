"""Build the search index from a folder of documents.

Usage:
    python -m docuquery.ingest ./docs
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docuquery import config
from docuquery.chunking import split_text
from docuquery.embeddings import Embedder
from docuquery.store import DocStore

TEXT_EXTS = {".txt", ".md"}
PDF_EXTS = {".pdf"}


def read_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def iter_documents(folder):
    """Yield (path, text) for every readable .txt/.md/.pdf under folder."""
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        try:
            if ext in TEXT_EXTS:
                yield str(path), path.read_text(encoding="utf-8", errors="replace")
            elif ext in PDF_EXTS:
                yield str(path), read_pdf(path)
        except Exception as e:  # corrupt file, weird encoding — skip, don't die
            print(f"skipping {path}: {e}")


def build_chunks(folder):
    chunks = []
    for path, text in iter_documents(folder):
        for piece in split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            chunks.append({"text": piece, "source": Path(path).name})
    return chunks


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ingest documents into the docuquery index.")
    ap.add_argument("docs", help="folder containing .txt / .md / .pdf files")
    ap.add_argument("--data-dir", default=None, help="where to store the index")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR
    chunks = build_chunks(args.docs)
    if not chunks:
        print("no readable documents found.")
        return 1

    embedder = Embedder(config.MODEL_NAME)
    print(f"embedding {len(chunks)} chunks with {config.MODEL_NAME} ...")
    vectors = embedder.encode([c["text"] for c in chunks])

    store = DocStore(
        embedder.dim,
        data_dir / "index" / "docuquery.faiss",
        data_dir / "index" / "metadata.json",
    )
    store.add(chunks, vectors)
    store.save()
    print(f"indexed {len(chunks)} chunks from {args.docs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
