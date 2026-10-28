from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=384):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def clear_collection(self):
        # Deletes the entire collection if it exists
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(collection_name=self.collection)
        # Recreate a fresh, empty collection
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def search(self, query_vector, top_k: int = 5):
        # Updated to query_points for modern qdrant-client versions
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )
        contexts = []
        scores = []  # cosine similarity per chunk, same order as contexts
        sources = set()

        for r in response.points:
            payload = getattr(r, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                scores.append(round(r.score, 4))
                sources.add(source)

        return {"contexts": contexts, "scores": scores, "sources": list(sources)}