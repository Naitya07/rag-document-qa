"""ChromaDB vector store and retrieval with cross-encoder reranking."""

import chromadb
from sentence_transformers import CrossEncoder

from .embedder import embed_texts

_cross_encoder = None
_client = None
_chunk_counter = 0


def get_cross_encoder() -> CrossEncoder:
    """Load cross-encoder model for reranking (cached)."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def get_client():
    """Get a persistent ChromaDB client (singleton)."""
    global _client
    if _client is None:
        _client = chromadb.Client()
    return _client


def get_collection(collection_name: str = "documents"):
    """Get or create a ChromaDB collection."""
    client = get_client()
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(collection_name: str = "documents"):
    """Delete and recreate the collection (used when starting fresh)."""
    global _chunk_counter
    client = get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    _chunk_counter = 0
    return get_collection(collection_name)


def index_chunks(chunks: list[dict], filename: str = "", collection_name: str = "documents"):
    """Embed and store chunks in ChromaDB with source filename."""
    global _chunk_counter
    collection = get_collection(collection_name)

    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    ids = [f"chunk_{_chunk_counter + i}" for i in range(len(chunks))]
    metadatas = [
        {"page": c["page"], "chunk_index": c["chunk_index"], "source": filename}
        for c in chunks
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    _chunk_counter += len(chunks)
    return len(chunks)


def retrieve(query: str, top_k: int = 10, rerank_top: int = 5, collection_name: str = "documents") -> list[dict]:
    """Retrieve relevant chunks: embed query -> vector search -> cross-encoder rerank."""
    collection = get_collection(collection_name)
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    if not results["documents"][0]:
        return []

    # Cross-encoder reranking
    ce = get_cross_encoder()
    pairs = [(query, doc) for doc in results["documents"][0]]
    scores = ce.predict(pairs)

    ranked = []
    for i, score in enumerate(scores):
        ranked.append({
            "text": results["documents"][0][i],
            "page": results["metadatas"][0][i]["page"],
            "source": results["metadatas"][0][i].get("source", ""),
            "score": float(score),
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:rerank_top]
