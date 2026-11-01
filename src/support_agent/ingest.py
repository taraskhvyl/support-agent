"""Embed chunks and store them in ChromaDB."""

import shutil

import chromadb
from openai import OpenAI

from . import config
from .chunking import build_chunks


def main():
    rows = build_chunks(config.DOCS_DIR)
    print(f"{len(rows)} chunks")
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    col = chromadb.PersistentClient(str(config.CHROMA_DIR)).create_collection(
        config.COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    client = OpenAI()
    for i in range(0, len(rows), 100):
        batch = rows[i : i + 100]
        emb = client.embeddings.create(model=config.EMBED_MODEL, input=[r[1] for r in batch])
        col.add(
            ids=[r[0] for r in batch],
            documents=[r[1] for r in batch],
            metadatas=[r[2] for r in batch],
            embeddings=[e.embedding for e in emb.data],
        )
    print("done")


if __name__ == "__main__":
    main()
