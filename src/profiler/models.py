"""
Data models for dataset profiling and schema inspection.
"""

from typing import Any, Optional
from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str = Field(..., description="Original column name")
    clean_name: str = Field(..., description="Sanitized SQL-compatible identifier")
    original_dtype: str = Field(..., description="Pandas underlying data type")
    inferred_type: str = Field(..., description="Inferred semantic type: numeric, categorical, datetime, boolean, text")
    total_count: int = Field(..., description="Total row count")
    null_count: int = Field(..., description="Number of missing / null values")
    null_percentage: float = Field(..., description="Percentage of missing values (0-100)")
    unique_count: int = Field(..., description="Number of unique non-null values")
    sample_values: list[Any] = Field(default_factory=list, description="Up to 5 representative unique sample values")
    min_val: Optional[Any] = Field(default=None, description="Minimum value if numeric/date")
    max_val: Optional[Any] = Field(default=None, description="Maximum value if numeric/date")
    mean_val: Optional[float] = Field(default=None, description="Mean average if numeric")
    top_categories: Optional[dict[str, int]] = Field(default=None, description="Most frequent categories with frequencies")


class DatasetProfile(BaseModel):
    file_name: str = Field(..., description="Name of the uploaded or sampled file")
    table_name: str = Field(default="dataset", description="DuckDB registered table name")
    row_count: int = Field(..., description="Total number of rows")
    column_count: int = Field(..., description="Total number of columns")
    total_cells: int = Field(..., description="Total cells (rows * columns)")
    total_missing_cells: int = Field(..., description="Total missing cells across the dataset")
    missing_cell_percentage: float = Field(..., description="Dataset-wide missing percentage")
    duplicate_row_count: int = Field(..., description="Count of duplicate rows")
    memory_usage_mb: float = Field(..., description="Approximate in-memory footprint in MB")
    columns: list[ColumnProfile] = Field(default_factory=list, description="Per-column profiles")
    summary_text: str = Field(default="", description="Human and LLM-readable summary description")
