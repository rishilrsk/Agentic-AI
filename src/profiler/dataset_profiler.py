"""
Dataset Profiling module.
Performs comprehensive schema inspection, semantic type inference, missing value analysis,
and LLM-ready context generation.
"""

from typing import Any
import pandas as pd
import numpy as np

from src.profiler.models import ColumnProfile, DatasetProfile
from src.utils.logger import logger


class DatasetProfiler:
    """Profiles pandas DataFrames to extract rich structural and statistical metadata."""

    @staticmethod
    def infer_column_type(series: pd.Series) -> str:
        """
        Infer high-level semantic type:
        - 'numeric'
        - 'datetime'
        - 'boolean'
        - 'categorical'
        - 'text'
        """
        # If boolean dtype
        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        # If numeric dtype
        if pd.api.types.is_numeric_dtype(series):
            # Check if it behaves as a binary indicator
            non_null = series.dropna()
            if len(non_null) > 0 and set(non_null.unique()).issubset({0, 1}):
                return "boolean"
            return "numeric"

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Try parsing strings as datetime if feasible
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            sample = series.dropna().astype(str).head(20)
            if len(sample) > 0:
                try:
                    # Only attempt if looks like a date format (contains / or -)
                    if all("-" in s or "/" in s for s in sample):
                        pd.to_datetime(sample, errors="raise")
                        return "datetime"
                except Exception:
                    pass

        # String / Object categorical check
        non_null_count = series.count()
        if non_null_count == 0:
            return "text"

        unique_count = series.nunique()
        # If unique ratio is small or <= 25 distinct values
        if unique_count <= 25 or (unique_count / non_null_count) < 0.15:
            return "categorical"

        return "text"

    def profile_column(
        self, series: pd.Series, col_name: str, clean_name: str
    ) -> ColumnProfile:
        """Generate profile metrics for a single column."""
        total_count = len(series)
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_count * 100), 2) if total_count > 0 else 0.0
        unique_count = int(series.nunique())

        inferred_type = self.infer_column_type(series)
        non_null = series.dropna()

        # Collect up to 5 representative samples
        sample_values = []
        if len(non_null) > 0:
            sample_series = non_null.drop_duplicates().head(5)
            for v in sample_series:
                if isinstance(v, (np.integer, int)):
                    sample_values.append(int(v))
                elif isinstance(v, (np.floating, float)):
                    sample_values.append(round(float(v), 2))
                else:
                    sample_values.append(str(v))

        min_val = None
        max_val = None
        mean_val = None
        top_cats = None

        if inferred_type == "numeric" and len(non_null) > 0:
            try:
                min_val = round(float(non_null.min()), 2)
                max_val = round(float(non_null.max()), 2)
                mean_val = round(float(non_null.mean()), 2)
            except Exception:
                pass
        elif inferred_type == "categorical" and len(non_null) > 0:
            try:
                top_counts = non_null.value_counts().head(5).to_dict()
                top_cats = {str(k): int(v) for k, v in top_counts.items()}
            except Exception:
                pass
        elif inferred_type == "datetime" and len(non_null) > 0:
            try:
                dt_series = pd.to_datetime(non_null, errors="coerce")
                min_val = str(dt_series.min())
                max_val = str(dt_series.max())
            except Exception:
                pass

        return ColumnProfile(
            name=col_name,
            clean_name=clean_name,
            original_dtype=str(series.dtype),
            inferred_type=inferred_type,
            total_count=total_count,
            null_count=null_count,
            null_percentage=null_pct,
            unique_count=unique_count,
            sample_values=sample_values,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            top_categories=top_cats,
        )

    def profile_dataframe(
        self,
        df: pd.DataFrame,
        file_name: str = "dataset.csv",
        table_name: str = "dataset",
        column_mapping: dict[str, str] = None,
    ) -> DatasetProfile:
        """
        Generate full structural and statistical profile of a dataset.
        """
        logger.info(f"Generating profile for table '{table_name}' ({len(df)} rows, {len(df.columns)} cols)")

        if column_mapping is None:
            column_mapping = {c: c for c in df.columns}

        # Reverse lookup for original names
        rev_map = {v: k for k, v in column_mapping.items()}

        col_profiles: list[ColumnProfile] = []
        for col in df.columns:
            orig_name = rev_map.get(col, col)
            profile = self.profile_column(df[col], orig_name, col)
            col_profiles.append(profile)

        row_count = len(df)
        col_count = len(df.columns)
        total_cells = row_count * col_count
        total_missing = int(df.isna().sum().sum())
        missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0
        dup_count = int(df.duplicated().sum())
        mem_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 3)

        summary_text = (
            f"Table '{table_name}' contains {row_count:,} rows and {col_count} columns "
            f"({mem_mb:.2f} MB in-memory). Missing cells: {missing_pct}% ({total_missing:,} cells). "
            f"Duplicate rows: {dup_count:,}."
        )

        return DatasetProfile(
            file_name=file_name,
            table_name=table_name,
            row_count=row_count,
            column_count=col_count,
            total_cells=total_cells,
            total_missing_cells=total_missing,
            missing_cell_percentage=missing_pct,
            duplicate_row_count=dup_count,
            memory_usage_mb=mem_mb,
            columns=col_profiles,
            summary_text=summary_text,
        )

    @staticmethod
    def format_schema_for_llm(profile: DatasetProfile) -> str:
        """
        Create a concise, structured markdown schema description
        tailored for LLM prompt context during planning and SQL generation.
        """
        lines = [
            f"### Table Schema: `{profile.table_name}`",
            f"- **Source File:** {profile.file_name}",
            f"- **Dimensions:** {profile.row_count:,} rows × {profile.column_count} columns",
            f"- **Data Quality:** {profile.missing_cell_percentage}% missing cells, {profile.duplicate_row_count} duplicate rows",
            "",
            "| Column Name | Semantic Type | Pandas Dtype | Missing % | Uniques | Value Range / Categories / Samples |",
            "|---|---|---|---|---|---|",
        ]

        for col in profile.columns:
            details = []
            if col.inferred_type == "numeric":
                if col.min_val is not None and col.max_val is not None:
                    details.append(f"Range: [{col.min_val} to {col.max_val}]")
                if col.mean_val is not None:
                    details.append(f"Avg: {col.mean_val}")
            elif col.inferred_type == "categorical" and col.top_categories:
                cats_str = ", ".join([f"'{k}' ({v})" for k, v in list(col.top_categories.items())[:3]])
                details.append(f"Top: {cats_str}")
            elif col.sample_values:
                samples_str = ", ".join([f"'{s}'" for s in col.sample_values[:3]])
                details.append(f"Samples: [{samples_str}]")

            details_str = " | ".join(details) if details else "None"
            lines.append(
                f"| `{col.clean_name}` | {col.inferred_type} | {col.original_dtype} | "
                f"{col.null_percentage}% | {col.unique_count} | {details_str} |"
            )

        return "\n".join(lines)
