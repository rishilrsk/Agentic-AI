"""
DuckDB Analytical SQL Execution Engine.
Provides in-memory relational execution with schema registration, security checks, and timing.
"""

import time
from typing import Optional
import duckdb
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from config import settings
from src.sql_engine.validator import (
    validate_read_only_query,
    enforce_query_limits,
    SQLSecurityError,
    SQLValidationError,
)
from src.utils.logger import logger


class ExecutionResult(BaseModel):
    """Encapsulates the structured output of an executed SQL query."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = Field(..., description="Whether the query executed without errors")
    query: str = Field(..., description="Sanitized SQL query executed")
    data: Optional[pd.DataFrame] = Field(default=None, description="Result set as Pandas DataFrame")
    columns: list[str] = Field(default_factory=list, description="Output column names")
    row_count: int = Field(default=0, description="Number of returned rows")
    execution_time_ms: float = Field(default=0.0, description="Execution duration in milliseconds")
    error_message: Optional[str] = Field(default=None, description="Error explanation if execution failed")


class DuckDBEngine:
    """Manages an in-memory DuckDB analytical engine session."""

    def __init__(self):
        # Create an in-memory database connection
        self.conn = duckdb.connect(database=":memory:")
        self.registered_tables: set[str] = set()
        logger.info("Initialized in-memory DuckDB execution engine.")

    def register_dataset(self, df: pd.DataFrame, table_name: str = "dataset") -> None:
        """
        Register a pandas DataFrame as a queryable relational table in DuckDB.
        """
        try:
            self.conn.register(table_name, df)
            self.registered_tables.add(table_name)
            logger.info(f"Registered table '{table_name}' in DuckDB ({len(df)} rows).")
        except Exception as e:
            logger.error(f"Failed to register table '{table_name}': {str(e)}")
            raise RuntimeError(f"Failed to register dataset in DuckDB: {str(e)}")

    def execute_query(
        self,
        raw_query: str,
        table_name: str = "dataset",
        validate_security: bool = True,
    ) -> ExecutionResult:
        """
        Validate and execute a read-only SQL query against registered tables.
        """
        start_time = time.perf_counter()
        sanitized_query = raw_query.strip()

        # Step 1: Security and syntax validation
        if validate_security:
            try:
                sanitized_query = validate_read_only_query(raw_query)
                sanitized_query = enforce_query_limits(
                    sanitized_query, max_rows=settings.MAX_RESULT_ROWS
                )
            except (SQLSecurityError, SQLValidationError) as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.warning(f"SQL Validation Rejected: {str(e)} | Query: {raw_query}")
                return ExecutionResult(
                    success=False,
                    query=raw_query,
                    execution_time_ms=round(elapsed_ms, 2),
                    error_message=str(e),
                )

        # Step 2: Execute query in DuckDB
        try:
            result_df: pd.DataFrame = self.conn.execute(sanitized_query).df()
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"SQL Execution Succeeded: {len(result_df)} rows in {elapsed_ms:.2f}ms. Query: {sanitized_query}"
            )
            return ExecutionResult(
                success=True,
                query=sanitized_query,
                data=result_df,
                columns=list(result_df.columns),
                row_count=len(result_df),
                execution_time_ms=round(elapsed_ms, 2),
            )
        except duckdb.CatalogException as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            err_msg = f"Catalog/Column Error: {str(e)}"
            logger.error(f"DuckDB Catalog Error: {err_msg}")
            return ExecutionResult(
                success=False,
                query=sanitized_query,
                execution_time_ms=round(elapsed_ms, 2),
                error_message=err_msg,
            )
        except duckdb.ParserException as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            err_msg = f"SQL Syntax Error: {str(e)}"
            logger.error(f"DuckDB Syntax Error: {err_msg}")
            return ExecutionResult(
                success=False,
                query=sanitized_query,
                execution_time_ms=round(elapsed_ms, 2),
                error_message=err_msg,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            err_msg = f"Execution Error: {str(e)}"
            logger.error(f"DuckDB Execution Error: {err_msg}")
            return ExecutionResult(
                success=False,
                query=sanitized_query,
                execution_time_ms=round(elapsed_ms, 2),
                error_message=err_msg,
            )

    def close(self):
        """Close connection and free resources."""
        self.conn.close()
