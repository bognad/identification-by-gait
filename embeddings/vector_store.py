# embeddings/vector_store.py
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from typing import List, Dict, Any

from .video_embedder import build_video_embedding


class VectorStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "gait_vectors",
        dim: int = 128,
    ):
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.dim = dim

        # создаём коллекцию, если её нет
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE,
                ),
            )

    def add(self, ids: List[str], vectors: np.ndarray, metadata: List[Dict[str, Any]] | None = None):
        payload = metadata or [{} for _ in ids]
        points = [
            models.PointStruct(
                id=ids[i],
                vector=vectors[i].astype(float).tolist(),
                payload=payload[i],
            )
            for i in range(len(ids))
        ]

        self.client.upsert(
            collection_name=self.collection,
            wait=True,
            points=points,
        )

    def index_video(store: VectorStore, video_path: str, person_id: str, model, visualize: bool = False):
        emb = build_video_embedding(model, video_path, visualize=visualize)
        if emb is None:
            print(f"Не удалось построить эмбеддинг для {video_path}")
            return

        emb = emb.astype(np.float32)
        store.add(
            ids=[video_path],  # можно использовать путь как id
            vectors=np.stack([emb]),
            metadata=[{"person_id": person_id, "video_path": video_path}],
        )
        print(f"Индексировано: {video_path} (person_id={person_id})")

    def search(self, query_vector: np.ndarray, top_k: int = 3, query_filter: models.Filter | None = None):
        res = self.client.query_points(
            collection_name=self.collection,
            query=query_vector.astype(float),  # numpy вектор
            query_filter=query_filter,
            limit=top_k,
        )
        return res

    def identify_video(store: VectorStore, video_path: str, model, top_k: int = 3, visualize: bool = False):
        emb = build_video_embedding(model, video_path, visualize=visualize)
        if emb is None:
            print(f"Не удалось построить эмбеддинг для {video_path}")
            return

        emb = emb.astype(np.float32)
        res = store.search(emb, top_k=top_k)

        for p in res.points:
            print(
                f"ID={p.id}, score={p.score:.4f}, "
                f"person_id={p.payload.get('person_id')}, "
                f"video_path={p.payload.get('video_path')}"
            )