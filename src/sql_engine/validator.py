"""
SQL Validator module.
Ensures queries are strictly read-only, safe, syntactically sound, and constrained.
"""

import re
import sqlparse
from sqlparse.sql import Statement
from sqlparse.tokens import DDL, DML, Keyword

from config import settings


class SQLSecurityError(Exception):
    """Raised when a SQL query violates security or read-only constraints."""
    pass


class SQLValidationError(Exception):
    """Raised when a SQL query is malformed or invalid."""
    pass


# Forbidden SQL commands and keywords that modify state or access system resources
FORBIDDEN_KEYWORDS = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
    "ATTACH",
    "DETACH",
    "COPY",
    "EXPORT",
    "INSTALL",
    "LOAD",
    "PRAGMA",
    "CALL",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "REVOKE",
    "VACUUM",
    "CHECKPOINT",
}


def sanitize_sql_query(raw_query: str) -> str:
    """
    Clean and normalize SQL query string:
    - Strips markdown code blocks (```sql ... ```)
    - Strips leading/trailing semicolons and whitespace
    """
    cleaned = raw_query.strip()
    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:sql)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    cleaned = cleaned.rstrip(";").strip()
    return cleaned


def validate_read_only_query(query: str) -> str:
    """
    Validate that the SQL query is strictly a single, read-only SELECT statement.
    Returns sanitized query string if valid, raises SQLSecurityError or SQLValidationError otherwise.
    """
    cleaned = sanitize_sql_query(query)
    if not cleaned:
        raise SQLValidationError("SQL query is empty.")

    # Parse with sqlparse
    statements = sqlparse.parse(cleaned)
    if not statements:
        raise SQLValidationError("Could not parse SQL query.")

    if len(statements) > 1:
        raise SQLSecurityError(
            "Multiple SQL statements detected. Only a single SELECT query is permitted."
        )

    stmt: Statement = statements[0]
    stmt_type = stmt.get_type()

    if stmt_type != "SELECT":
        raise SQLSecurityError(
            f"Unauthorized statement type '{stmt_type}'. Only 'SELECT' statements are permitted."
        )

    # Token-level inspection for forbidden keywords
    tokens = [t for t in stmt.flatten() if not t.is_whitespace]
    for token in tokens:
        val = token.value.upper()
        if val in FORBIDDEN_KEYWORDS:
            raise SQLSecurityError(
                f"Forbidden SQL keyword '{val}' detected. Database modification or administrative operations are blocked."
            )

    return cleaned


def enforce_query_limits(query: str, max_rows: int = None) -> str:
    """
    Ensure the query has an explicit LIMIT clause to prevent out-of-memory errors.
    If no LIMIT is present, appends LIMIT {max_rows}.
    """
    if max_rows is None:
        max_rows = settings.MAX_RESULT_ROWS

    cleaned = sanitize_sql_query(query)
    # Check if query already has a LIMIT clause
    if not re.search(r"\bLIMIT\s+\d+\b", cleaned, flags=re.IGNORECASE):
        cleaned = f"{cleaned} LIMIT {max_rows}"

    return cleaned
