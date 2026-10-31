"""
Unit tests for Phase 3: Dataset Ingestion and Profiler.
"""

import io
import pytest
import pandas as pd
from pathlib import Path

from config import settings
from src.profiler import (
    load_dataset,
    sanitize_column_name,
    IngestionError,
    DatasetProfiler,
    DatasetProfile,
)


def test_sanitize_column_name():
    """Verify that column names are converted to safe, lowercase SQL identifiers."""
    assert sanitize_column_name("Order ID") == "order_id"
    assert sanitize_column_name("Product-Name (USD)") == "product_name_usd"
    assert sanitize_column_name("  Total Sales %  ") == "total_sales"
    assert sanitize_column_name("123Number") == "col_123number"
    assert sanitize_column_name("") == "col"


def test_invalid_file_extension():
    """Verify that unsupported file extensions are rejected."""
    buf = io.BytesIO(b"dummy,content\n1,2")
    with pytest.raises(IngestionError, match="Unsupported file format"):
        load_dataset(buf, "test_file.txt")


def test_empty_file_rejection():
    """Verify that zero-byte files raise IngestionError."""
    buf = io.BytesIO(b"")
    with pytest.raises(IngestionError, match="empty"):
        load_dataset(buf, "empty.csv")


def test_sample_sales_dataset_profiling():
    """Test full ingestion and profiling on the sample benchmark sales_data.csv."""
    sample_path = settings.PROJECT_ROOT / "data" / "samples" / "sales_data.csv"
    assert sample_path.exists(), "Benchmark dataset sales_data.csv must exist."

    df, mapping = load_dataset(sample_path, "sales_data.csv")
    assert len(df) == 20
    assert len(df.columns) == 10
    assert "order_id" in df.columns
    assert "revenue" in df.columns

    profiler = DatasetProfiler()
    profile = profiler.profile_dataframe(df, file_name="sales_data.csv", column_mapping=mapping)

    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == 20
    assert profile.column_count == 10
    assert profile.total_cells == 200

    # Verify column profile lookups
    col_dict = {col.clean_name: col for col in profile.columns}

    # order_date should be inferred as datetime
    assert col_dict["order_date"].inferred_type == "datetime"

    # category should be inferred as categorical
    assert col_dict["category"].inferred_type == "categorical"
    assert "Electronics" in col_dict["category"].top_categories

    # revenue should be numeric
    assert col_dict["revenue"].inferred_type == "numeric"
    assert col_dict["revenue"].min_val is not None
    assert col_dict["revenue"].max_val is not None
    assert col_dict["revenue"].mean_val is not None

    # Missing value detection: customer_rating has missing values in our sample
    assert col_dict["customer_rating"].null_count > 0

    # LLM Schema formatted text
    llm_schema = profiler.format_schema_for_llm(profile)
    assert "### Table Schema: `dataset`" in llm_schema
    assert "revenue" in llm_schema
    assert "order_date" in llm_schema


def test_in_memory_dataframe_profiling():
    """Test profiler with synthetic data including duplicates and edge cases."""
    data = {
        "User ID": [1, 2, 3, 4, 4],
        "Country": ["USA", "Canada", "USA", "UK", "UK"],
        "Score": [95.5, None, 88.0, 72.5, 72.5],
        "Is Active": [1, 0, 1, 1, 1],
    }
    df = pd.DataFrame(data)

    profiler = DatasetProfiler()
    profile = profiler.profile_dataframe(df, file_name="synthetic.csv", table_name="users")

    assert profile.row_count == 5
    assert profile.column_count == 4
    assert profile.duplicate_row_count == 1  # Row 4 is duplicated
    assert profile.total_missing_cells == 1  # 1 missing score

    col_dict = {c.name: c for c in profile.columns}
    assert col_dict["Is Active"].inferred_type == "boolean"
    assert col_dict["Score"].null_count == 1
    assert col_dict["Country"].inferred_type == "categorical"
