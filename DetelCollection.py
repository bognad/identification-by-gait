from qdrant_client import QdrantClient

client = QdrantClient("http://localhost:6333")

client.delete_collection("gait_vectors")
print("deleted")

print("Collections after delete:", client.get_collections())