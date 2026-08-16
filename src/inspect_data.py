import weaviate

from config import ALL_COLLECTIONS


def inspect(limit: int = 21):
    client = weaviate.connect_to_local()
    try:
        for index_name in ALL_COLLECTIONS:
            if not client.collections.exists(index_name):
                print(f"\n=== {index_name} (does not exist) ===")
                continue
            collection = client.collections.get(index_name)
            print(f"\n=== {index_name} ===")
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
