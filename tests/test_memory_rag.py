"""
Unit tests for Phase 8: SQLite Memory and ChromaDB RAG.
"""

from pathlib import Path
from src.memory import SQLiteMemory, RAGKnowledgeBase


def test_sqlite_memory_lifecycle(tmp_path: Path):
    db_file = tmp_path / "test_history.db"
    mem = SQLiteMemory(db_path=db_file)

    # Save an analysis session
    row_id = mem.save_analysis(
        dataset_name="sales.csv",
        user_query="Find top revenue products",
        sql_query="SELECT product, revenue FROM dataset",
        execution_success=True,
        row_count=5,
        execution_time_ms=12.5,
        insights="Top product is Headphones.",
    )
    assert row_id is not None

    # Retrieve history
    history = mem.get_recent_history(limit=5)
    assert len(history) == 1
    assert history[0]["dataset_name"] == "sales.csv"
    assert history[0]["execution_success"] == 1
    assert history[0]["row_count"] == 5

    # Clear history
    mem.clear_history()
    assert len(mem.get_recent_history()) == 0


def test_rag_knowledge_base_indexing():
    rag = RAGKnowledgeBase()
    sample_defs = [
        {"term": "revenue", "definition": "Total monetary amount collected before discounts."},
        {"term": "units_sold", "definition": "Quantity of physical items sold in each transaction."},
    ]
    rag.index_definitions(sample_defs)

    result = rag.retrieve_relevant_definitions("What is the total revenue?")
    assert "revenue" in result.lower()

    structured = rag.retrieve_structured("revenue and units", top_k=2)
    assert len(structured) >= 1
    assert "relevance" in structured[0]
    assert "term" in structured[0]


def test_auto_enrich_dataset_glossary():
    from src.profiler.models import ColumnProfile
    from src.memory import auto_enrich_dataset_glossary

    col = ColumnProfile(
        name="Revenue",
        clean_name="revenue",
        original_dtype="float64",
        inferred_type="numeric",
        total_count=10,
        null_count=0,
        null_percentage=0.0,
        unique_count=10,
    )
    enriched = auto_enrich_dataset_glossary([col])
    assert len(enriched) == 1
    assert enriched[0]["term"] == "revenue"
    assert "sales transactions" in enriched[0]["definition"]

