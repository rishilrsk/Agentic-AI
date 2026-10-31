"""
Semantic RAG Knowledge Base.
Indexes data dictionaries, column definitions, and business glossaries using ChromaDB.
Provides structured semantic retrieval with explainability.
"""

import sys
from pathlib import Path
from typing import Any, Optional

# Linux Cloud Deployment fix for ChromaDB sqlite3 version check
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import settings
from src.utils.logger import logger


# Common domain dictionary patterns to auto-enrich any dataset
COMMON_DOMAIN_GLOSSARY = {
    "revenue": "Total monetary volume generated from completed sales transactions.",
    "sales": "Gross aggregate monetary amount transacted.",
    "units_sold": "Number of individual physical units purchased in an order.",
    "quantity": "Total count of items or units transacted.",
    "unit_price": "Base unit price per single item in USD before discounts.",
    "price": "Standard catalog pricing per unit.",
    "discount": "Promotional deduction applied against the gross order value (0.00 = 0%, 0.10 = 10%).",
    "customer_rating": "Customer satisfaction score recorded on a 1.0 to 5.0 scale.",
    "rating": "Evaluative customer satisfaction score.",
    "category": "High-level merchandise hierarchy classification (e.g., Electronics, Furniture).",
    "product_name": "Specific commercial item or SKU title.",
    "product": "Product or service name.",
    "city": "Metropolitan delivery market where the purchase originated.",
    "order_date": "Calendar timestamp recording when the customer placed the transaction.",
    "date": "Temporal transaction occurrence date.",
    "order_id": "Unique alphanumeric or numeric transactional primary identifier.",
    "profit": "Net financial gain calculated as Revenue minus Total Costs.",
    "margin": "Profit margin ratio expressing profitability percentage.",
    "churn": "Customer retention indicator (1 indicates cancelled/lost, 0 indicates retained).",
    "salary": "Annual compensation figure in base currency.",
    "age": "Demographic age of subject in completed years.",
}


class RAGKnowledgeBase:
    """Manages semantic search over data dictionaries and business documentation."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or settings.get_chroma_path()
        self.in_memory_docs: list[dict[str, str]] = []
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_or_create_collection(name="data_dictionary")
            logger.info(f"Initialized ChromaDB vector store at {self.persist_dir}")
        except Exception as e:
            logger.warning(f"ChromaDB initialization failed ({str(e)}). Using in-memory fallback store.")
            self.client = None

    def index_definitions(self, definitions: list[dict[str, str]]):
        """
        Index field definitions or business glossary entries into vector storage.
        Each item is expected to have 'term' and 'definition' (and optional 'category').
        """
        if not definitions:
            return

        # Keep in-memory cache for fast UI inspection
        for d in definitions:
            # Check if term already exists in in_memory_docs
            existing = [x for x in self.in_memory_docs if x.get("term", "").lower() == d.get("term", "").lower()]
            if not existing:
                self.in_memory_docs.append(d)

        if self.client:
            try:
                documents = []
                metadatas = []
                ids = []

                for idx, item in enumerate(definitions):
                    term = item.get("term", "")
                    definition = item.get("definition", "")
                    doc_text = f"Term: {term}. Definition: {definition}"
                    doc_id = f"def_{term.lower().replace(' ', '_')}"

                    documents.append(doc_text)
                    metadatas.append({"term": term, "definition": definition})
                    ids.append(doc_id)

                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.info(f"Successfully indexed {len(definitions)} definitions in ChromaDB.")
            except Exception as e:
                logger.warning(f"ChromaDB indexing error ({str(e)}). Preserving in-memory list.")

    def get_all_definitions(self) -> list[dict[str, str]]:
        """Return all indexed glossary definitions for display in UI table."""
        return list(self.in_memory_docs)

    def retrieve_structured(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Retrieve matching glossary entries with semantic relevance annotations.
        """
        lower_query = query.lower()
        matched: list[dict[str, Any]] = []
        seen_terms = set()

        # Step 1: ChromaDB semantic search if available
        if self.client:
            try:
                count = self.collection.count()
                if count > 0:
                    results = self.collection.query(
                        query_texts=[query],
                        n_results=min(top_k, count),
                    )
                    docs = results.get("documents", [[]])[0]
                    metas = results.get("metadatas", [[]])[0]
                    distances = results.get("distances", [[]])[0] if "distances" in results else []

                    for i, meta in enumerate(metas):
                        term = meta.get("term", "")
                        if term and term.lower() not in seen_terms:
                            seen_terms.add(term.lower())
                            dist = distances[i] if i < len(distances) else 0.5
                            rel = "High Relevance" if dist < 0.8 else "Context Match"
                            matched.append({
                                "term": term,
                                "definition": meta.get("definition", ""),
                                "relevance": rel,
                                "source": "ChromaDB Semantic Search",
                            })
            except Exception as e:
                logger.warning(f"ChromaDB query fallback: {str(e)}")

        # Step 2: In-memory keyword and domain match
        for item in self.in_memory_docs:
            term = item.get("term", "").lower()
            if term not in seen_terms:
                if term in lower_query or any(w in term for w in lower_query.split() if len(w) > 3):
                    seen_terms.add(term)
                    matched.append({
                        "term": item.get("term", ""),
                        "definition": item.get("definition", ""),
                        "relevance": "Direct Query Match",
                        "source": "Domain Glossary",
                    })

        return matched[:top_k]

    def retrieve_relevant_definitions(self, query: str, top_k: int = 3) -> str:
        """
        Retrieve relevant glossary entries as a formatted prompt/markdown text block.
        """
        structured = self.retrieve_structured(query, top_k=top_k)
        if not structured:
            return "No specific data dictionary terms matched this inquiry."

        lines = []
        for item in structured:
            lines.append(f"- **`{item['term']}`** [{item['relevance']}]: {item['definition']}")
        return "\n".join(lines)


def auto_enrich_dataset_glossary(columns: list[Any]) -> list[dict[str, str]]:
    """
    Automatically produces rich, human-understandable business definitions
    for any uploaded dataset columns by combining domain knowledge and profiling metrics.
    """
    enriched = []
    for col in columns:
        col_name = getattr(col, "clean_name", str(col)).lower()
        orig_name = getattr(col, "name", str(col))
        inferred = getattr(col, "inferred_type", "data")
        uniques = getattr(col, "unique_count", "")

        # Check if known domain pattern matches
        matched_desc = None
        for key, desc in COMMON_DOMAIN_GLOSSARY.items():
            if key in col_name or col_name in key:
                matched_desc = desc
                break

        if not matched_desc:
            matched_desc = (
                f"Dataset attribute '{orig_name}' with {inferred} format. "
                f"Contains {uniques} unique distinct values."
            )

        enriched.append({
            "term": col_name,
            "definition": matched_desc,
        })

    return enriched
