# python .\DetecorMovePeople\soloWork\search.py .\DetecorMovePeople\Dataset\Originals\Originals\S003.avi --top-k 3
import argparse
import numpy as np
import cv2
from ultralytics import YOLO

from embeddings.vector_store import VectorStore
from embeddings.video_embedder import build_video_embedding_lstm

def main():
    parser = argparse.ArgumentParser(description="Gait search (только поиск)")
    parser.add_argument("video_path", type=str, help="Путь к видео-запросу")
    parser.add_argument("--top-k", type=int, default=5, help="Сколько кандидатов вернуть")
    parser.add_argument("--model-path", type=str, default="../yolov8n-pose.pt")
    args = parser.parse_args()

    print(f"[SEARCH] Видео-запрос: {args.video_path}")

    model = YOLO(args.model_path)
    store = VectorStore(dim=128)

    query_emb = build_video_embedding_lstm(model, args.video_path)
    results = store.search(query_emb, top_k=args.top_k)
    print("[SEARCH] Кандидаты:")
    for hit in results.points:
        print(f"id : {hit.id} | Это - {hit.payload.get("name")} | Шанс =  {hit.score}")

        # if hasattr(hit, "id"):
        #     _id = hit.id
        #     _score = getattr(hit, "score", None)
        #     _payload = getattr(hit, "payload", None)
        # # если это dict
        # elif isinstance(hit, dict):
        #     _id = hit.get("id")
        #     _score = hit.get("score")
        #     _payload = hit.get("payload")
        # # если это кортеж (id, score, payload) или что‑то похожее
        # elif isinstance(hit, tuple) and len(hit) >= 3:
        #     _id, _score, _payload = hit[0], hit[1], hit[2]
        # else:
        #     print("  raw hit:", hit)
        #     continue
        #
        # if _score is not None:
        #     print(f"  id={_id}, score={float(_score):.4f}, payload={_payload}")
        # else:
        #     print(f"  id={_id}, payload={_payload}")


if __name__ == "__main__":
    main()