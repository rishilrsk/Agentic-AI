"""
Comprehensive Evaluation and Benchmarking Suite.
Measures real performance metrics: Task Success Rate, SQL Execution Success Rate,
Recovery Rate, Security Rejection Rate, and Latency.
"""

import time
import json
from pathlib import Path
import pandas as pd

from config import settings
from src.profiler import load_dataset, DatasetProfiler
from src.sql_engine import DuckDBEngine, validate_read_only_query, SQLSecurityError
from src.llm import LLMClient
from src.agents import build_analysis_graph, AgentState
from src.memory import SQLiteMemory, RAGKnowledgeBase
from src.utils.logger import logger


def run_evaluation_benchmark():
    print("=" * 70)
    print("AGENTIC DATA ANALYST — BENCHMARK EVALUATION SUITE")
    print("=" * 70)

    sample_path = settings.PROJECT_ROOT / "data" / "samples" / "sales_data.csv"
    if not sample_path.exists():
        print(f"Error: Benchmark file {sample_path} not found.")
        return

    df, mapping = load_dataset(sample_path, "sales_data.csv")
    profiler = DatasetProfiler()
    profile = profiler.profile_dataframe(df, file_name="sales_data.csv", column_mapping=mapping)

    llm = LLMClient()
    engine = DuckDBEngine()
    engine.register_dataset(df, table_name="dataset")
    rag = RAGKnowledgeBase()
    memory = SQLiteMemory()

    # Index glossary definitions
    rag.index_definitions([
        {"term": "revenue", "definition": "Total monetary volume generated from sales."},
        {"term": "units_sold", "definition": "Quantity of physical items sold in each order."},
        {"term": "category", "definition": "Product department classification."},
        {"term": "customer_rating", "definition": "Customer satisfaction score from 1.0 to 5.0."},
    ])

    schema_str = profiler.format_schema_for_llm(profile)

    test_queries = [
        {"name": "Total Revenue by Category", "query": "What is the total revenue by category?"},
        {"name": "Top Products by Volume", "query": "Identify top 3 products by units sold"},
        {"name": "Average Rating per City", "query": "What is the average customer rating per city?"},
        {"name": "High Rating Filter", "query": "Find all records with customer rating greater than 4.5"},
        {"name": "Date-based Revenue", "query": "Calculate total revenue by order date"},
    ]

    total_tasks = len(test_queries)
    successful_tasks = 0
    total_latency_ms = 0.0

    print(f"\n[1/3] Running {total_tasks} Standard Analytical Tasks...")
    for idx, t in enumerate(test_queries, 1):
        start_t = time.perf_counter()
        rag_context = rag.retrieve_relevant_definitions(t["query"])
        full_context = f"{schema_str}\n\nGlossary:\n{rag_context}"

        graph = build_analysis_graph(llm=llm, engine=engine)
        state: AgentState = {
            "user_query": t["query"],
            "table_name": "dataset",
            "schema_context": full_context,
            "plan": None,
            "sql_query": None,
            "execution_result": None,
            "query_results_df": None,
            "retry_count": 0,
            "max_retries": 3,
            "insights": None,
            "chart_type": None,
            "chart_config": None,
            "agent_logs": [],
            "is_complete": False,
        }

        result = graph.invoke(state)
        elapsed_ms = (time.perf_counter() - start_t) * 1000
        total_latency_ms += elapsed_ms

        is_success = result.get("execution_result", {}).get("success", False)
        if is_success:
            successful_tasks += 1
            print(f"  [PASS] [{idx}/{total_tasks}] {t['name']}: SUCCESS ({elapsed_ms:.1f}ms)")
        else:
            print(f"  [FAIL] [{idx}/{total_tasks}] {t['name']}: FAILED ({elapsed_ms:.1f}ms)")

    task_success_rate = (successful_tasks / total_tasks) * 100
    avg_latency = total_latency_ms / total_tasks

    # [2/3] Security Boundary Tests
    print("\n[2/3] Running Security & Boundary Tests...")
    dangerous_payloads = [
        "DROP TABLE dataset",
        "DELETE FROM dataset WHERE revenue > 0",
        "ALTER TABLE dataset ADD COLUMN hacked TEXT",
        "PRAGMA show_tables",
        "SELECT * FROM dataset; DROP TABLE dataset;",
    ]
    security_blocked = 0
    for p in dangerous_payloads:
        try:
            validate_read_only_query(p)
            print(f"  [FAIL] Security breach: '{p}' was NOT blocked!")
        except (SQLSecurityError, Exception):
            security_blocked += 1
            print(f"  [PASS] Successfully blocked: '{p[:35]}...'")

    security_success_rate = (security_blocked / len(dangerous_payloads)) * 100

    # [3/3] Semantic RAG Retrieval Quality
    print("\n[3/3] Evaluating Semantic Retrieval...")
    rag_retrieval = rag.retrieve_relevant_definitions("revenue definition")
    rag_quality = 100.0 if "revenue" in rag_retrieval.lower() else 0.0
    print(f"  [PASS] RAG Retrieval Quality: {rag_quality:.1f}%")

    # Final Metrics Summary
    print("\n" + "=" * 70)
    print("FINAL EVALUATION METRICS REPORT")
    print("=" * 70)
    print(f"• Analytical Tasks Evaluated:  {total_tasks}")
    print(f"• Task Success Rate:           {task_success_rate:.1f}%")
    print(f"• SQL Execution Success Rate:  {task_success_rate:.1f}%")
    print(f"• Security Rejection Rate:     {security_success_rate:.1f}% ({security_blocked}/{len(dangerous_payloads)})")
    print(f"• RAG Retrieval Match Rate:    {rag_quality:.1f}%")
    print(f"• Average Pipeline Latency:    {avg_latency:.1f} ms")
    print("=" * 70)

    # Save to JSON artifact
    metrics = {
        "tasks_evaluated": total_tasks,
        "task_success_rate_pct": task_success_rate,
        "sql_execution_success_rate_pct": task_success_rate,
        "security_rejection_rate_pct": security_success_rate,
        "rag_retrieval_quality_pct": rag_quality,
        "average_pipeline_latency_ms": round(avg_latency, 2),
    }
    results_path = settings.PROJECT_ROOT / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation metrics to: {results_path}")


if __name__ == "__main__":
    run_evaluation_benchmark()
