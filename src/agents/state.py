"""
AgentState definition for LangGraph workflow.
Provides strongly-typed shared memory passed between cooperating agent nodes.
"""

from typing import Any, Optional, TypedDict


class AgentState(TypedDict):
    """Shared state dictionary representing the analytical pipeline lifecycle."""

    # Initial inputs
    user_query: str
    table_name: str
    schema_context: str

    # Planner outputs
    plan: Optional[dict[str, Any]]

    # SQL generation & execution
    sql_query: Optional[str]
    execution_result: Optional[dict[str, Any]]
    query_results_df: Optional[Any]  # Pandas DataFrame or serializable dict
    retry_count: int
    max_retries: int

    # Insight & visualization
    insights: Optional[str]
    chart_type: Optional[str]
    chart_config: Optional[dict[str, Any]]

    # Execution traceability & logs
    agent_logs: list[str]
    is_complete: bool
