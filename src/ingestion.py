import argparse
import os
import time

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import weaviate

from chunking import build_documents
from config import BATCH_SIZE, EMBEDDING_MODEL, INCOMING_FOLDER, INDEX_NAME
from weaviate_schema import build_schema, ensure_collection, reset_collection

load_dotenv()

SUPPORTED_EXTENSIONS = {".pdf"}


def _get_vectorstore(client) -> WeaviateVectorStore:
    ensure_collection(client, INDEX_NAME)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return WeaviateVectorStore(
        client=client,
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text",
        schema=build_schema(INDEX_NAME),
    )


def ingest_file(file_path: str, client=None) -> None:
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"Skipping unsupported file type: {file_path}")
        return

    print(f"Ingesting {file_path} ...")
    docs = build_documents(file_path)
    if not docs:
        print(f"  No content extracted from {file_path}")
        return

    owns_client = client is None
    client = client or weaviate.connect_to_local()
    try:
        vectorstore = _get_vectorstore(client)

        total = len(docs)
        table_count = sum(
            1 for d in docs if d.metadata["content_type"] == "table")
        print(f"  {total} chunks ({table_count} table, "
              f"{total - table_count} text)")

        for start in range(0, total, BATCH_SIZE):
            batch = docs[start:start + BATCH_SIZE]
            # chunk_id is a deterministic uuid5 of the content hash, so
            # re-ingesting the same file upserts instead of duplicating.
            batch_uuids = [doc.metadata["chunk_id"] for doc in batch]
            vectorstore.add_documents(batch, uuids=batch_uuids)
            print(f"  Batch {start // BATCH_SIZE + 1}: indexed "
                  f"{min(start + BATCH_SIZE, total)}/{total} chunks")

        print(f"Successfully indexed {total} chunks from "
              f"{os.path.basename(file_path)}")
    finally:
        if owns_client:
            client.close()


def ingest_existing_files(reset: bool = False) -> None:
    os.makedirs(INCOMING_FOLDER, exist_ok=True)
    client = weaviate.connect_to_local()
    try:
        if reset:
            print(f"Resetting collection: {INDEX_NAME}")
            reset_collection(client)
        ensure_collection(client, INDEX_NAME)

        for name in sorted(os.listdir(INCOMING_FOLDER)):
            path = os.path.join(INCOMING_FOLDER, name)
            if os.path.isfile(path):
                ingest_file(path, client=client)
    finally:
        client.close()


class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        time.sleep(1)  # let the OS finish writing the file before reading it
        ingest_file(event.src_path)


def start_watcher() -> None:
    os.makedirs(INCOMING_FOLDER, exist_ok=True)
    observer = Observer()
    observer.schedule(NewFileHandler(), INCOMING_FOLDER, recursive=False)
    observer.start()
    print(f"Watching folder: {INCOMING_FOLDER}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aurora RAG ingestion pipeline")
    parser.add_argument("--reset", action="store_true",
                        help="Delete and recreate the Weaviate collection before ingesting")
    parser.add_argument("--watch", action="store_true",
                        help="Keep watching the folder for new files after the initial ingest")
    args = parser.parse_args()

    ingest_existing_files(reset=args.reset)
    if args.watch:
        start_watcher()
