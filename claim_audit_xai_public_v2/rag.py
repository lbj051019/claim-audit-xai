from __future__ import annotations
from typing import List

import faiss
import numpy as np
from openai import OpenAI

from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)


def embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype="float32")

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    vectors = [item.embedding for item in response.data]
    return np.asarray(vectors, dtype="float32")


class RAG:
    def __init__(self):
        self.index = None
        self.texts: List[str] = []

    def build(self, chunks: List[str]) -> None:
        if not chunks:
            raise ValueError("No chunks provided for RAG index building.")

        self.texts = chunks
        vectors = embed(chunks)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

    def search(self, query: str, k: int = 5) -> List[str]:
        if self.index is None or not self.texts:
            return []

        q_vec = embed([query])
        _, indices = self.index.search(q_vec, k)
        results = [self.texts[i] for i in indices[0] if i >= 0]

        keywords = [token for token in query.lower().split() if len(token) > 2]
        filtered: List[str] = []

        for item in results:
            lower = item.lower()

            if "data |" in lower:
                continue
            if "navigation" in lower:
                continue
            if len(item) < 80:
                continue
            if keywords and not any(k in lower for k in keywords[:3]):
                continue

            filtered.append(item)

        return filtered if filtered else results[:1]
