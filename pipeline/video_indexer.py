from ..embeddings.vector_store import VectorStore

class VideoIndexer:
    def __init__(self, store: VectorStore, embedder):
        self.store = store
        self.embedder = embedder  # функция или объект

    def index_clip(self, clip_path: str, clip_id: str, meta: dict | None = None):
        vec = self.embedder(clip_path)
        self.store.add([clip_id], vec[None, :], metadata=[meta or {}])