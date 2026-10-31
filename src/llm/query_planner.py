"""
Intelligent Schema-Aware Query Engine and Dynamic Insight Generator.
Parses natural language analytical queries against real dataset schemas, builds accurate
DuckDB SQL queries, and generates grounded mathematical insights from query results.
"""

import re
from typing import Any, Optional
import pandas as pd


def parse_schema_columns(schema_context: str) -> dict[str, dict[str, str]]:
    """
    Extract columns, types, and descriptions from markdown schema context string.
    Returns: {col_name: {"type": semantic_type, "dtype": pandas_dtype, "info": summary_info}}
    """
    columns: dict[str, dict[str, str]] = {}
    lines = schema_context.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("|") and not line.startswith("| Column Name") and not line.startswith("|---"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 3:
                raw_name = parts[0].replace("`", "").strip()
                semantic_type = parts[1].strip().lower()
                dtype = parts[2].strip()
                extra_info = parts[5].strip() if len(parts) >= 6 else ""
                columns[raw_name] = {
                    "type": semantic_type,
                    "dtype": dtype,
                    "info": extra_info,
                }
    return columns


def generate_dynamic_sql(
    user_query: str,
    schema_context: str,
    table_name: str = "dataset"
) -> str:
    """
    Intelligently generates DuckDB SQL based on actual columns in the schema and the user's inquiry.
    Supports top-N ranking, multi-column aggregation, filtering, date-based grouping, and counting.
    """
    columns = parse_schema_columns(schema_context)
    query_lower = user_query.lower()

    # If schema parsing is empty, provide sensible default
    if not columns:
        return f"SELECT * FROM {table_name} LIMIT 10"

    # Categorize columns
    numeric_cols = [c for c, meta in columns.items() if meta["type"] == "numeric"]
    categorical_cols = [c for c, meta in columns.items() if meta["type"] in ("categorical", "text")]
    datetime_cols = [c for c, meta in columns.items() if meta["type"] == "datetime"]

    # 1. Detect limit count (e.g., "top 5", "top 3", "first 10")
    limit_match = re.search(r"\b(?:top|first|limit)\s+(\d+)\b", query_lower)
    limit_num = int(limit_match.group(1)) if limit_match else 10

    # 2. Detect desired aggregation
    agg_func = "SUM"
    if "average" in query_lower or "avg" in query_lower or "mean" in query_lower:
        agg_func = "AVG"
    elif "minimum" in query_lower or "min" in query_lower or "lowest" in query_lower:
        agg_func = "MIN"
    elif "maximum" in query_lower or "max" in query_lower or "highest" in query_lower:
        agg_func = "MAX"
    elif "count" in query_lower or "how many" in query_lower:
        agg_func = "COUNT"

    # 3. Find target metric column (numeric)
    target_metric = None
    for col in numeric_cols:
        col_clean = col.replace("_", " ")
        if col in query_lower or col_clean in query_lower:
            target_metric = col
            break

    # Keyword synonyms for common metrics if direct match not found
    if not target_metric:
        if "revenue" in query_lower or "sales" in query_lower:
            for c in numeric_cols:
                if any(k in c for k in ["revenue", "sales", "amount", "total"]):
                    target_metric = c
                    break
        elif "unit" in query_lower or "quantity" in query_lower or "volume" in query_lower:
            for c in numeric_cols:
                if any(k in c for k in ["unit", "quantity", "volume", "sold", "count"]):
                    target_metric = c
                    break
        elif "rating" in query_lower or "score" in query_lower:
            for c in numeric_cols:
                if any(k in c for k in ["rating", "score", "satisfaction"]):
                    target_metric = c
                    break
        elif "price" in query_lower or "cost" in query_lower:
            for c in numeric_cols:
                if any(k in c for k in ["price", "cost"]):
                    target_metric = c
                    break

    # If still not found, pick first numeric column
    if not target_metric and numeric_cols:
        target_metric = numeric_cols[0]

    # 4. Find target dimension column (categorical or date)
    target_dim = None

    # Check date keywords first
    if any(k in query_lower for k in ["month", "date", "trend", "daily", "year", "time"]) and datetime_cols:
        target_dim = datetime_cols[0]
        # Format date for monthly trend if requested
        if "month" in query_lower:
            return (
                f"SELECT strftime(TRY_CAST({target_dim} AS DATE), '%Y-%m') AS order_month, "
                f"ROUND({agg_func}({target_metric}), 2) AS total_{target_metric} "
                f"FROM {table_name} "
                f"WHERE {target_dim} IS NOT NULL "
                f"GROUP BY order_month "
                f"ORDER BY order_month ASC"
            )

    # Check categorical columns
    for col in categorical_cols:
        col_clean = col.replace("_", " ")
        if col in query_lower or col_clean in query_lower:
            target_dim = col
            break

    # Keyword synonyms for dimensions
    if not target_dim:
        if "product" in query_lower or "item" in query_lower:
            for c in categorical_cols:
                if any(k in c for k in ["product", "item", "name", "title"]):
                    target_dim = c
                    break
        elif "city" in query_lower or "location" in query_lower or "region" in query_lower or "market" in query_lower:
            for c in categorical_cols:
                if any(k in c for k in ["city", "state", "region", "country", "location"]):
                    target_dim = c
                    break
        elif "category" in query_lower or "department" in query_lower or "genre" in query_lower:
            for c in categorical_cols:
                if any(k in c for k in ["category", "dept", "department", "type"]):
                    target_dim = c
                    break

    # 5. Check for filter predicates (e.g. "rating > 4.5", "discount > 0")
    filter_clause = ""
    pred_match = re.search(r"\b([a-zA-Z_]+)\s*(>|<|>=|<=|=)\s*(\d+(?:\.\d+)?)\b", user_query)
    if pred_match:
        pred_col = pred_match.group(1).lower().replace(" ", "_")
        pred_op = pred_match.group(2)
        pred_val = pred_match.group(3)
        # Check if pred_col matches any actual column
        for c in columns:
            if pred_col in c or c in pred_col:
                filter_clause = f"WHERE {c} {pred_op} {pred_val}"
                break

    # 6. Compose Final SQL Query based on detected structure

    # Case A: Filter query without grouping
    if filter_clause and not target_dim:
        return f"SELECT * FROM {table_name} {filter_clause} LIMIT {limit_num}"

    # Case B: Both dimension and metric identified (Aggregation & Group By)
    if target_dim and target_metric:
        alias = f"{agg_func.lower()}_{target_metric}"
        order_dir = "DESC"
        if "lowest" in query_lower or "least" in query_lower or "bottom" in query_lower or "worst" in query_lower:
            order_dir = "ASC"

        where_part = f"{filter_clause} AND " if filter_clause else "WHERE "
        where_part += f"{target_dim} IS NOT NULL AND {target_metric} IS NOT NULL"

        return (
            f"SELECT {target_dim}, ROUND({agg_func}({target_metric}), 2) AS {alias} "
            f"FROM {table_name} "
            f"{where_part} "
            f"GROUP BY {target_dim} "
            f"ORDER BY {alias} {order_dir} "
            f"LIMIT {limit_num}"
        )

    # Case C: Only dimension identified -> Count frequencies
    if target_dim and not target_metric:
        return (
            f"SELECT {target_dim}, COUNT(*) AS total_count "
            f"FROM {table_name} "
            f"WHERE {target_dim} IS NOT NULL "
            f"GROUP BY {target_dim} "
            f"ORDER BY total_count DESC "
            f"LIMIT {limit_num}"
        )

    # Case D: Only metric identified -> Overall summary statistic
    if target_metric and not target_dim:
        where_part = f"{filter_clause}" if filter_clause else f"WHERE {target_metric} IS NOT NULL"
        return (
            f"SELECT ROUND({agg_func}({target_metric}), 2) AS overall_{agg_func.lower()}_{target_metric} "
            f"FROM {table_name} {where_part}"
        )

    # Fallback: Top rows of dataset
    return f"SELECT * FROM {table_name} LIMIT {limit_num}"


def generate_grounded_insights(
    user_query: str,
    sql_query: str,
    df: Optional[pd.DataFrame]
) -> str:
    """
    Dynamically computes actual mathematical facts from the returned query results
    to generate an evidence-based narrative without numerical hallucination.
    """
    if df is None or df.empty:
        return (
            "**Key Finding**: The executed query returned no matching records.\n"
            "- **Observation**: The specified filter criteria did not match any active rows in the dataset.\n"
            "- **Recommendation**: Relax filter constraints or check for null values."
        )

    cols = list(df.columns)
    row_count = len(df)

    # Case 1: Single row aggregated scalar (e.g., SELECT AVG(...) ...)
    if row_count == 1 and len(cols) == 1:
        val = df.iloc[0, 0]
        col_name = cols[0].replace("_", " ").title()
        return (
            f"### 💡 Key Analytical Finding\n"
            f"- **Computed Metric**: The overall **{col_name}** is **{val:,.2f}**.\n"
            f"- **Data Grounding**: Deterministically calculated across the filtered records in DuckDB.\n"
            f"- **Next Step**: You can segment this value by categories, cities, or dates to inspect variations."
        )

    # Case 2: Dimension + Metric aggregation table
    dim_col = cols[0]
    metric_col = None
    for c in cols[1:]:
        if pd.api.types.is_numeric_dtype(df[c]):
            metric_col = c
            break

    if metric_col is not None and row_count >= 1:
        # Sort values
        sorted_df = df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)
        top_row = sorted_df.iloc[0]
        top_dim = str(top_row[dim_col])
        top_val = float(top_row[metric_col])

        total_sum = float(sorted_df[metric_col].sum())
        mean_val = float(sorted_df[metric_col].mean())
        pct_share = round((top_val / total_sum * 100), 1) if total_sum > 0 else 0.0

        bottom_row = sorted_df.iloc[-1]
        bottom_dim = str(bottom_row[dim_col])
        bottom_val = float(bottom_row[metric_col])

        dim_title = dim_col.replace("_", " ").title()
        metric_title = metric_col.replace("_", " ").title()

        insights_text = (
            f"### 💡 Grounded Analytical Findings\n\n"
            f"- **Leading Performer**: **{top_dim}** ranks #1 in **{metric_title}** with **{top_val:,.2f}**, "
            f"representing **{pct_share}%** of the total volume analyzed.\n"
            f"- **Performance Range & Average**: Values range from **{top_val:,.2f}** ({top_dim}) down to "
            f"**{bottom_val:,.2f}** ({bottom_dim}), with a group average of **{mean_val:,.2f}** across {row_count} segments.\n"
            f"- **Actionable Recommendation**: Prioritize operational support and inventory for **{top_dim}**, "
            f"while evaluating targeted growth strategies for lower-performing segments like **{bottom_dim}**."
        )
        return insights_text

    # Default tabular summary
    return (
        f"### 💡 Result Summary\n"
        f"- **Rows Returned**: {row_count} records matching the criteria.\n"
        f"- **Columns Analyzed**: {', '.join(cols)}.\n"
        f"- **Observation**: Review the detailed records in the data table below."
    )
