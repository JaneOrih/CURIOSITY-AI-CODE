"""
Script to build a FAISS vector store from a directory of text files.

This script reads all `.txt` files in the specified data directory, splits
them into sentences, embeds them using a sentence transformer model and
writes a FAISS index to disk.  A corresponding `.meta.json` file is also
written containing the raw sentences used to build the index.

Usage:

```
python scripts/build_vectorstore.py --data-dir data --index-path config/index.faiss --model sentence-transformers/all-MiniLM-L6-v2
```

The default model is `sentence-transformers/all-MiniLM-L6-v2` which offers
a good balance between speed and quality.  To build an index for a larger
corpus or a specialised domain you may choose a different model.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import faiss
os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORTS", "1")
from sentence_transformers import SentenceTransformer


def collect_texts(data_dir: str) -> List[str]:
    """Collect and clean sentences from all text files in `data_dir`."""
    texts: List[str] = []
    for filepath in glob.glob(os.path.join(data_dir, "*.txt")):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                # Only include lines of reasonable length to avoid noise
                if len(s) > 20:
                    texts.append(s)
    return texts


def build_index(texts: List[str], model_name: str) -> faiss.Index:
    """Embed a list of texts and construct a FAISS index."""
    if not texts:
        raise ValueError("No texts provided for indexing.")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    # Convert embeddings to float32 as required by FAISS
    emb = embeddings.astype("float32")
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index


def save_index(index: faiss.Index, index_path: str, texts: List[str]) -> None:
    """Save the FAISS index and associated metadata to disk."""
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    meta = {"texts": texts}
    with open(index_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS vector store from text files.")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing .txt files to index.")
    parser.add_argument("--index-path", type=str, required=True, help="Path to output FAISS index file.")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence transformer model to use for embeddings.")
    args = parser.parse_args()

    texts = collect_texts(args.data_dir)
    if not texts:
        print(f"No suitable lines found in {args.data_dir}.")
        return
    print(f"Collected {len(texts)} sentences. Embedding...")
    index = build_index(texts, args.model)
    save_index(index, args.index_path, texts)
    print(f"Index written to {args.index_path}")


if __name__ == "__main__":
    main()