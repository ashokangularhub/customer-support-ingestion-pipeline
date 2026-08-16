"""Central configuration for the Aurora RAG ingestion pipeline."""
import re

INCOMING_FOLDER = r"D:\AI-Sessions\Langchain-RAG-Project\rag-files"
INDEX_NAME = "AuroraRagDocuments"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100

# Narrative text chunking (RecursiveCharacterTextSplitter)
TEXT_CHUNK_SIZE = 800
TEXT_CHUNK_OVERLAP = 150

# Tables are kept intact per chunk unless they exceed this size, in which case
# they are split into row-groups that each repeat the header row for context.
TABLE_MAX_CHARS = 1800

# --- Document type detection -------------------------------------------------
# Maps a filename substring (case-insensitive) to a canonical doc_type + title.
DOC_TYPE_RULES = [
    ("product_information_catalog", "product_catalog", "Product Information Catalog"),
    ("returns_refunds_policy", "returns_refunds_policy", "Returns & Refunds Policy"),
    ("technical_support_guide", "technical_support_guide",
     "Technical Support & Troubleshooting Guide"),
]
DEFAULT_DOC_TYPE = "general"


def detect_doc_type(file_name: str) -> tuple[str, str]:
    """Returns (doc_type, doc_title) inferred from the file name."""
    lowered = file_name.lower()
    for needle, doc_type, title in DOC_TYPE_RULES:
        if needle in lowered:
            return doc_type, title
    return DEFAULT_DOC_TYPE, file_name


# --- Known product catalog (used to tag chunks with product_name/product_id) -
# Small, closed catalog for this dataset; keeps metadata enrichment simple and
# deterministic instead of relying on NER/LLM extraction.
KNOWN_PRODUCTS = [
    {"product_id": "AUR-EB-PRO2", "product_name": "AuroraBuds Pro 2"},
    {"product_id": "AUR-WT-FIT3", "product_name": "AuroraWatch Fit 3"},
    {"product_id": "AUR-SPK-GO", "product_name": "AuroraSound Go"},
    {"product_id": "AUR-ACC-DESKRISE", "product_name": "AuroraDesk Rise"},
    {"product_id": "AUR-KB-K5", "product_name": "AuroraType K5"},
]

PRODUCT_NAME_PATTERN = re.compile(
    "|".join(re.escape(p["product_name"]) for p in KNOWN_PRODUCTS)
)

# Regex to pull "Document version X.Y | Last (policy) review: Month YYYY" style
# headers that appear on page 1 of every Aurora document.
DOC_VERSION_PATTERN = re.compile(
    r"Document version\s+(?P<version>[\d.]+)\s*\|\s*Last[^:]*:\s*(?P<review_date>[A-Za-z]+ \d{4})",
    re.IGNORECASE,
)
