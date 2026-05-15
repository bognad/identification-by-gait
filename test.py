import numpy as np
from ultralytics import YOLO

from embeddings.video_embedder import build_video_embedding_lstm

same_person = [
    "Dataset/Originals/S001.avi",
    "Dataset/Sx_Track 1/Sx_Track 1/S001_T1.avi",
]

other_person = [
    "Dataset/Originals/S002.avi",
    "Dataset/Originals/S003.avi",
]
model = YOLO("../yolov8n-pose.pt")
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

embs_same = []
for p in same_person:
    e = build_video_embedding_lstm(model, p)
    if e is not None:
        embs_same.append((p, e))

embs_other = []
for p in other_person:
    e = build_video_embedding_lstm(model, p)
    if e is not None:
        embs_other.append((p, e))

print("=== SAME PERSON ===")
for i in range(len(embs_same)):
    for j in range(i + 1, len(embs_same)):
        p_i, e_i = embs_same[i]
        p_j, e_j = embs_same[j]
        s = cos_sim(e_i, e_j)
        print(f"{p_i}  <->  {p_j}  cos={s:.3f}")

print("\n=== SAME vs OTHER ===")
for p_i, e_i in embs_same:
    for p_j, e_j in embs_other:
        s = cos_sim(e_i, e_j)
        print(f"{p_i}  <->  {p_j}  cos={s:.3f}")