import weaviate

INDEX_NAME = "AuroraRagDocuments"


def inspect(limit: int = 21):
    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get(INDEX_NAME)
        print(
            f"Object count: {collection.aggregate.over_all(total_count=True).total_count}")

        for i, item in enumerate(collection.iterator(include_vector=True)):
            if i >= limit:
                break
            vector = item.vector.get("default", item.vector)
            print(f"\n--- Object {item.uuid} ---")
            print("Properties:", item.properties)
            print(f"Vector (dim={len(vector)}): {vector[:8]}...")
    finally:
        client.close()


if __name__ == "__main__":
    inspect()
