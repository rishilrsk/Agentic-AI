"""
Unit and integration tests for Phase 6: LangGraph Multi-Agent Workflow.
"""

import pandas as pd
import pytest

from src.agents import build_analysis_graph, AgentState
from src.llm import LLMClient
from src.sql_engine import DuckDBEngine


@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "category": ["Electronics", "Furniture", "Electronics", "Furniture"],
        "revenue": [500.0, 300.0, 700.0, 150.0],
    })


def test_full_graph_execution(sales_df):
    """Test end-to-end multi-agent execution from user inquiry to verified insights."""
    engine = DuckDBEngine()
    engine.register_dataset(sales_df, table_name="dataset")
    llm = LLMClient()

    graph = build_analysis_graph(llm=llm, engine=engine)

    initial_state: AgentState = {
        "user_query": "What is the total revenue by category?",
        "table_name": "dataset",
        "schema_context": "Table dataset: category (VARCHAR), revenue (DOUBLE)",
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

    final_state = graph.invoke(initial_state)

    # Assert all agent checkpoints succeeded
    assert final_state["is_complete"] is True
    assert final_state["plan"] is not None
    assert "steps" in final_state["plan"]
    assert final_state["sql_query"] is not None
    assert final_state["execution_result"]["success"] is True
    assert final_state["query_results_df"] is not None
    assert len(final_state["query_results_df"]) > 0
    assert final_state["insights"] is not None
    assert len(final_state["agent_logs"]) >= 4
