"""BM25, dense embeddings, and small local search-backend adapters."""
import math
import re
import uuid
from collections import Counter


def words(text):
    return re.findall(r"\w+", text.lower())


class Search:
    def __init__(self, docs, retriever="bm25", embedding_model="BAAI/bge-small-en-v1.5",
                 backend="numpy", exact=True, device="cpu", revision="main"):
        self.docs, self.retriever, self.backend, self.exact = docs, retriever, backend, exact
        self.embedding_model = embedding_model
        self.truncated = 0
        self.mode = "bm25" if retriever == "bm25" else "exact"
        if retriever == "bm25":
            self.tokens = [Counter(words(d["text"])) for d in docs]
            self.lengths = [sum(c.values()) for c in self.tokens]
            self.avg = sum(self.lengths)/max(1, len(docs))
            df = Counter(t for c in self.tokens for t in c)
            self.idf = {t: math.log(1+(len(docs)-n+.5)/(n+.5)) for t, n in df.items()}
            return
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(embedding_model, device=device, revision=revision)
        self.truncated = sum(len(self.encoder.tokenizer(d["text"], truncation=False)["input_ids"]) > self.encoder.max_seq_length for d in docs)
        self.vectors = self.encoder.encode([d["text"] for d in docs], normalize_embeddings=True,
                                          show_progress_bar=True).astype("float32")
        if backend == "qdrant":
            from qdrant_client import QdrantClient, models
            self.db = QdrantClient(location=":memory:")
            self.db.create_collection("c2t", vectors_config=models.VectorParams(size=self.vectors.shape[1], distance=models.Distance.COSINE))
            for start in range(0, len(docs), 256):
                self.db.upsert("c2t", points=[models.PointStruct(id=i, vector=self.vectors[i].tolist()) for i in range(start, min(start+256,len(docs)))])
        elif backend == "chroma":
            import chromadb
            self.db = chromadb.EphemeralClient().create_collection("c2t-"+uuid.uuid4().hex, metadata={"hnsw:space":"cosine"})
            for start in range(0, len(docs), 256):
                stop = min(start+256,len(docs))
                self.db.add(ids=[str(i) for i in range(start,stop)], embeddings=self.vectors[start:stop].tolist(),
                            metadatas=[{"entry":i} for i in range(start,stop)])
            # Chroma native search is ANN. Exact mode explicitly reads eligible
            # vectors from Chroma and ranks in NumPy; not native exact search.
            self.mode = "exact_adapter" if exact else "ann"
        elif backend == "faiss":
            import faiss
            self.faiss = faiss
        elif backend != "numpy":
            raise ValueError("Unknown backend")
        if not exact and backend != "chroma":
            raise ValueError("This small runner supports ANN only for Chroma; use exact mode elsewhere")

    def query(self, text, eligible=None, k=10):
        eligible = list(range(len(self.docs))) if eligible is None else sorted(eligible)
        if not eligible:
            return []
        k = min(k, len(eligible))
        if self.retriever == "bm25":
            scores = {}
            for i in eligible:
                c, length = self.tokens[i], self.lengths[i]
                scores[i] = sum(self.idf.get(t,0)*c.get(t,0)*2.5 / (c.get(t,0)+1.5*(.25+.75*length/max(1,self.avg))) for t in set(words(text)))
            return sorted(eligible, key=lambda i:(-scores[i], i))[:k]
        import numpy as np
        query_text = "Represent this sentence for searching relevant passages: " + text if "bge-" in self.embedding_model.lower() else text
        q = self.encoder.encode([query_text], normalize_embeddings=True).astype("float32")
        if self.backend == "qdrant":
            from qdrant_client import models
            result = self.db.query_points("c2t", query=q[0].tolist(),
                query_filter=models.Filter(must=[models.HasIdCondition(has_id=eligible)]),
                search_params=models.SearchParams(exact=True), limit=k).points
            return [int(p.id) for p in result]
        if self.backend == "chroma":
            where = {"entry":{"$in":eligible}}
            if not self.exact:
                return [int(i) for i in self.db.query(query_embeddings=q.tolist(), where=where, n_results=k)["ids"][0]]
            result = self.db.get(where=where, include=["embeddings"])
            ids = [int(i) for i in result["ids"]]
            scores = np.asarray(result["embeddings"]) @ q[0]
            return [ids[j] for j in sorted(range(len(ids)),key=lambda j:(-float(scores[j]),ids[j]))[:k]]
        if self.backend == "faiss":
            # Rebuild an eligible subset index for exact filtering; latency includes it.
            index = self.faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors[eligible])
            _, ids = index.search(q,k)
            return [eligible[i] for i in ids[0] if i >= 0]
        scores = self.vectors[eligible] @ q[0]
        return [eligible[j] for j in sorted(range(len(eligible)),key=lambda j:(-float(scores[j]),eligible[j]))[:k]]
