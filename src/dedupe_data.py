import weaviate
from weaviate.classes.query import Filter

INDEX_NAME = "AuroraRagDocuments"


def dedupe():
    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get(INDEX_NAME)
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
            print("No duplicates found.")
            return

        print(f"Deleting {len(duplicate_uuids)} duplicate objects...")
        collection.data.delete_many(
            where=Filter.by_id().contains_any(duplicate_uuids))
        print("Done.")
    finally:
        client.close()


if __name__ == "__main__":
    dedupe()
