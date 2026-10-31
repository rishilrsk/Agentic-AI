"""
Dataset ingestion module for CSV and Excel files.
Handles safe validation, format parsing, column sanitization, and persistence.
"""

import io
import re
from pathlib import Path
from typing import Union, BinaryIO
import pandas as pd

from config import settings
from src.utils.logger import logger


class IngestionError(Exception):
    """Custom exception raised when file validation or ingestion fails."""
    pass


def sanitize_column_name(col_name: str) -> str:
    """
    Sanitize a column name into a clean, safe SQL identifier:
    - Lowercase
    - Replace spaces and hyphens with underscores
    - Strip invalid characters (keeping only letters, numbers, underscores)
    - Ensure it does not start with a number
    """
    cleaned = str(col_name).strip().lower()
    cleaned = re.sub(r"[\s\-\.]+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")

    if not cleaned:
        cleaned = "col"
    if cleaned[0].isdigit():
        cleaned = f"col_{cleaned}"
    return cleaned


def deduplicate_column_names(columns: list[str]) -> list[str]:
    """Ensure all column names are strictly unique by appending numeric suffixes."""
    seen: dict[str, int] = {}
    unique_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            unique_cols.append(col)
        else:
            unique_cols.append(f"{col}_{seen[col]}")
            seen[col] += 1
    return unique_cols


def validate_file_metadata(file_name: str, file_size_bytes: int) -> None:
    """Validate file extension and size against configured boundaries."""
    ext = Path(file_name).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file format '{ext}'. Allowed extensions: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )

    if file_size_bytes <= 0:
        raise IngestionError("The uploaded file is empty (0 bytes).")

    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise IngestionError(
            f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE_MB}MB "
            f"(actual: {file_size_bytes / (1024 * 1024):.2f}MB)."
        )


def load_dataset(
    file_source: Union[str, Path, BinaryIO, io.BytesIO],
    file_name: str,
    sanitize_cols: bool = True
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load a CSV or Excel dataset into a pandas DataFrame.

    Returns:
        tuple[pd.DataFrame, dict[str, str]]:
            - Loaded DataFrame
            - Mapping dictionary {original_column_name: clean_column_name}
    """
    logger.info(f"Ingesting dataset: {file_name}")

    # Resolve byte buffer or file path
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.exists():
            raise IngestionError(f"File not found: {path}")
        validate_file_metadata(file_name, path.stat().st_size)
        file_input = path
    else:
        # Buffer / stream
        file_source.seek(0, io.SEEK_END)
        size = file_source.tell()
        file_source.seek(0)
        validate_file_metadata(file_name, size)
        file_input = file_source

    ext = Path(file_name).suffix.lower()

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(file_input, encoding="utf-8")
            except UnicodeDecodeError:
                if hasattr(file_input, "seek"):
                    file_input.seek(0)
                logger.warning(f"UTF-8 decoding failed for {file_name}. Retrying with ISO-8859-1.")
                df = pd.read_csv(file_input, encoding="ISO-8859-1")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_input, engine="openpyxl")
        else:
            raise IngestionError(f"Unsupported file format '{ext}'.")
    except Exception as e:
        logger.error(f"Error parsing file {file_name}: {str(e)}")
        raise IngestionError(f"Failed to parse file '{file_name}': {str(e)}")

    if df.empty:
        raise IngestionError(f"The dataset '{file_name}' contains no rows.")

    if len(df.columns) == 0:
        raise IngestionError(f"The dataset '{file_name}' contains no columns.")

    # Column mapping & sanitization
    original_cols = [str(c).strip() for c in df.columns]
    if sanitize_cols:
        cleaned_base = [sanitize_column_name(c) for c in original_cols]
        unique_cleaned = deduplicate_column_names(cleaned_base)
        col_mapping = dict(zip(original_cols, unique_cleaned))
        df.columns = unique_cleaned
    else:
        col_mapping = {c: c for c in original_cols}
        df.columns = original_cols

    logger.info(
        f"Successfully loaded '{file_name}' ({len(df)} rows, {len(df.columns)} columns)."
    )
    return df, col_mapping


def save_uploaded_file(file_bytes: bytes, file_name: str) -> Path:
    """Save an uploaded file to data/uploads directory."""
    uploads_dir = settings.get_uploads_path()
    safe_name = re.sub(r"[^\w\.-]", "_", file_name)
    destination = uploads_dir / safe_name
    with open(destination, "wb") as f:
        f.write(file_bytes)
    logger.info(f"Saved uploaded file to {destination}")
    return destination
