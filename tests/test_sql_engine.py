"""
Unit tests for Phase 4: SQL Analysis Engine (DuckDB Integration & Safety).
"""

import pytest
import pandas as pd

from src.sql_engine import (
    DuckDBEngine,
    ExecutionResult,
    validate_read_only_query,
    enforce_query_limits,
    SQLSecurityError,
    SQLValidationError,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104, 105],
        "category": ["Electronics", "Furniture", "Electronics", "Furniture", "Electronics"],
        "revenue": [100.0, 250.0, 150.0, 50.0, 300.0],
    })


def test_sanitize_and_validate_select():
    """Verify that legitimate SELECT queries pass validation."""
    query = "```sql\nSELECT category, SUM(revenue) AS total FROM dataset GROUP BY category;\n```"
    valid = validate_read_only_query(query)
    assert valid.startswith("SELECT category")
    assert not valid.endswith(";")


@pytest.mark.parametrize("dangerous_query", [
    "DROP TABLE dataset",
    "DELETE FROM dataset WHERE order_id = 101",
    "UPDATE dataset SET revenue = 0",
    "INSERT INTO dataset VALUES (999, 'Test', 0)",
    "ALTER TABLE dataset DROP COLUMN revenue",
    "ATTACH 'evil.db' AS evil",
    "PRAGMA show_tables",
    "COPY dataset TO 'stolen.csv'",
])
def test_block_destructive_queries(dangerous_query):
    """Ensure any modifying or system queries are rejected with SQLSecurityError."""
    with pytest.raises(SQLSecurityError):
        validate_read_only_query(dangerous_query)


def test_block_multi_statement():
    """Ensure multiple statements cannot be smuggled."""
    query = "SELECT * FROM dataset; DROP TABLE dataset"
    with pytest.raises(SQLSecurityError, match="Multiple SQL statements"):
        validate_read_only_query(query)


def test_enforce_query_limits():
    """Ensure queries without LIMIT get LIMIT appended, while queries with LIMIT are kept."""
    q1 = "SELECT * FROM dataset"
    limited_1 = enforce_query_limits(q1, max_rows=100)
    assert "LIMIT 100" in limited_1

    q2 = "SELECT * FROM dataset LIMIT 10"
    limited_2 = enforce_query_limits(q2, max_rows=100)
    assert limited_2.strip() == "SELECT * FROM dataset LIMIT 10"


def test_duckdb_execution_success(sample_df):
    """Test DuckDB query execution on registered DataFrame."""
    engine = DuckDBEngine()
    engine.register_dataset(sample_df, table_name="dataset")

    query = "SELECT category, SUM(revenue) AS total_rev FROM dataset GROUP BY category ORDER BY total_rev DESC"
    result: ExecutionResult = engine.execute_query(query)

    assert result.success is True
    assert result.error_message is None
    assert result.row_count == 2
    assert "category" in result.columns
    assert "total_rev" in result.columns
    assert result.execution_time_ms >= 0

    # Electronics total: 100 + 150 + 300 = 550
    electronics_rev = result.data[result.data["category"] == "Electronics"]["total_rev"].iloc[0]
    assert electronics_rev == 550.0


def test_duckdb_error_isolation(sample_df):
    """Verify DuckDB catches invalid columns gracefully without crashing."""
    engine = DuckDBEngine()
    engine.register_dataset(sample_df, table_name="dataset")

    query = "SELECT non_existent_column FROM dataset"
    result: ExecutionResult = engine.execute_query(query)

    assert result.success is False
    assert result.data is None
    assert "non_existent_column" in result.error_message


def test_duckdb_security_rejection_in_engine(sample_df):
    """Verify engine returns structured failure if dangerous query is submitted."""
    engine = DuckDBEngine()
    engine.register_dataset(sample_df, table_name="dataset")

    result = engine.execute_query("DROP TABLE dataset")
    assert result.success is False
    assert ("Unauthorized statement type" in result.error_message) or ("Forbidden" in result.error_message)
