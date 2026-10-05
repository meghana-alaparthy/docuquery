"""Ask a question against the indexed documents.

Usage:
    python -m docuquery.ask "how many remote days per week are allowed?"
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from docuquery import config
from docuquery.chunking import split_sentences
from docuquery.embeddings import Embedder
from docuquery.store import DocStore

_STOPWORDS = set(
    "a an the and or of to in on for with is are was were be as at by from "
    "this that it its we you they he she i me my our your their his her will "
    "would can could should shall may might do does did not no yes if then "
    "than so such just about into over after before between what when how many".split()
)


def _tokens(s):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOPWORDS]


def extractive_answer(question, passages, max_sentences=3):
    """Stitch the best-matching sentences from the retrieved passages into an answer."""
    qterms = set(_tokens(question))
    scored = []
    for p in passages:
        for sent in split_sentences(p["text"]):
            overlap = len(qterms & set(_tokens(sent)))
            if overlap:
                scored.append((overlap, sent))
    scored.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for _, sent in scored:
        if sent.lower() not in seen:
            seen.add(sent.lower())
            out.append(sent)
        if len(out) >= max_sentences:
            break
    return " ".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ask a question over your indexed documents.")
    ap.add_argument("question")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--data-dir", default=None, help="where the index was stored")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR
    index_path = data_dir / "index" / "docuquery.faiss"
    meta_path = data_dir / "index" / "metadata.json"
    if not index_path.exists():
        print("no index found — run: python -m docuquery.ingest <docs folder>")
        return 1

    k = args.top_k or config.TOP_K
    embedder = Embedder(config.MODEL_NAME)
    store = DocStore.load(embedder.dim, index_path, meta_path)
    passages = store.search(embedder.encode_query(args.question), k)
    if not passages:
        print("nothing found.")
        return 0

    print(f"\nQuestion: {args.question}\n")
    answer = extractive_answer(args.question, passages)
    if answer:
        print("Answer:", answer, "\n")
    print("Sources:")
    for i, p in enumerate(passages, 1):
        snippet = p["text"][:400] + ("..." if len(p["text"]) > 400 else "")
        print(f"\n[{i}] {p['source']}  (score {p['score']:.3f})\n{snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
