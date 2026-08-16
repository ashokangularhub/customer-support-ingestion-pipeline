import weaviate
from weaviate.classes.query import Filter

from config import ALL_COLLECTIONS


def dedupe():
    client = weaviate.connect_to_local()
    try:
        for index_name in ALL_COLLECTIONS:
            if not client.collections.exists(index_name):
                continue
            collection = client.collections.get(index_name)
            seen = set()
            duplicate_uuids = []

            for obj in collection.iterator():
                key = obj.properties.get("content_hash") or (
                    obj.properties.get("source"),
                    obj.properties.get("page"),
                    obj.properties.get("text"),
                )
                if key in seen:
                    duplicate_uuids.append(obj.uuid)
                else:
                    seen.add(key)

            if not duplicate_uuids:
                print(f"{index_name}: no duplicates found.")
                continue

            print(
                f"{index_name}: deleting {len(duplicate_uuids)} duplicate objects...")
            collection.data.delete_many(
                where=Filter.by_id().contains_any(duplicate_uuids))
    finally:
        client.close()


if __name__ == "__main__":
    dedupe()
