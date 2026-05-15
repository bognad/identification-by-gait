import numpy as np
from sympy.strategies.core import switch
from ultralytics import YOLO
from embeddings.vector_store import VectorStore
from embeddings.video_embedder import build_video_embedding_lstm

# Модель позы
model = YOLO("../yolov8n-pose.pt")

store = VectorStore(dim=128)
VIDEOS = [
    f"Dataset/Originals/S0{("0" + str(i)) if i < 10 else i}.avi" for i in range(1, 45)
    # f"Dataset/m1.mp4",
    # f"Dataset/s1.mp4",
    # f"Dataset/sh1.mp4",
]
i = 0
for path in VIDEOS:
    i += 1
    emb = build_video_embedding_lstm(model, path)
    print(f"idPoint={i},Name={path} embedding_shape={emb.shape}")

    store.add(
        ids=[i],
        vectors=np.stack([emb]),
        metadata=[{
            "path": path,
            "name": path,
        }]
    )