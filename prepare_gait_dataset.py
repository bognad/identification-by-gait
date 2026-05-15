import os
import numpy as np
from ultralytics import YOLO  # как ты загружаешь модель
from embeddings.video_embedder import build_sequence_embedding  # см. выше

# Список (video_path, person_id)
VIDEOS = [
    (f"Dataset/Originals/S0{("0" + str(i)) if i < 10 else i}.avi", f"S_{i}") for i in range(1, 45)
]
print(VIDEOS)
def main():
    model = YOLO("../yolov8n-pose.pt")
    seqs = []
    labels = []

    for video_path, person_id in VIDEOS:
        print("Видео:", video_path)
        seq = build_sequence_embedding(model, video_path, visualize=False)
        if seq is None:
            print("  пропуск (нет кадров)")
            continue
        if seq.ndim != 2 or seq.shape[1] != 41:
            print("  странная форма:", seq.shape)
            continue

        seqs.append(seq)  # (T_i, 41)
        labels.append(person_id)

    if not seqs:
        print("Ни одной валидной последовательности – нечего сохранять")
        return

    np.savez(
        "gait_sequences_41.npz",
        sequences=np.array(seqs, dtype=object),
        labels=np.array(labels),
    )
    print("Сохранён gait_sequences_41.npz")

if __name__ == "__main__":
    main()