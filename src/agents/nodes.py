"""
Agent nodes for LangGraph workflow:
- Supervisor/Planner node
- SQL Analyst node
- SQL Execution node
- Recovery/Fixer node
- Insight & Visualization node
"""

import pandas as pd
from typing import Any

from src.agents.state import AgentState
from src.llm import (
    LLMClient,
    PLANNER_SYSTEM_PROMPT,
    SQL_GENERATOR_SYSTEM_PROMPT,
    SQL_CORRECTION_SYSTEM_PROMPT,
    INSIGHT_SYSTEM_PROMPT,
)
from src.sql_engine import DuckDBEngine, ExecutionResult
from src.utils.logger import logger


def supervisor_node(state: AgentState, llm: LLMClient) -> dict[str, Any]:
    """Supervisor/Planner: Understands goal, forms execution plan and identifies chart type."""
    logger.info("[Supervisor] Formulating analytical plan...")
    sys_prompt = PLANNER_SYSTEM_PROMPT.format(schema_context=state["schema_context"])
    plan = llm.generate_json(prompt=state["user_query"], system_prompt=sys_prompt)

    chart_type = plan.get("chart_type", "bar")
    logs = list(state.get("agent_logs", []))
    logs.append(f"🎯 **Supervisor:** Created {len(plan.get('steps', []))}-step plan (Intent: {plan.get('intent', 'General')})")

    return {
        "plan": plan,
        "chart_type": chart_type,
        "agent_logs": logs,
    }


def sql_generator_node(state: AgentState, llm: LLMClient) -> dict[str, Any]:
    """SQL Analyst: Converts user query into single safe DuckDB SELECT query."""
    logger.info("[SQL Analyst] Generating SQL query...")
    sys_prompt = SQL_GENERATOR_SYSTEM_PROMPT.format(
        table_name=state["table_name"],
        schema_context=state["schema_context"],
    )
    raw_sql = llm.generate(prompt=state["user_query"], system_prompt=sys_prompt)

    logs = list(state.get("agent_logs", []))
    logs.append(f"💻 **SQL Analyst:** Generated SQL query: `{raw_sql.strip()}`")

    return {
        "sql_query": raw_sql.strip(),
        "agent_logs": logs,
    }


def sql_executor_node(state: AgentState, engine: DuckDBEngine) -> dict[str, Any]:
    """SQL Execution & Validation: Executes query with safety constraints and timing."""
    query = state.get("sql_query", "")
    logger.info(f"[Executor] Executing SQL: {query}")

    result: ExecutionResult = engine.execute_query(
        raw_query=query,
        table_name=state["table_name"],
        validate_security=True,
    )

    logs = list(state.get("agent_logs", []))
    retry_count = state.get("retry_count", 0)

    if result.success:
        logs.append(
            f"✅ **SQL Engine:** Query executed successfully in {result.execution_time_ms:.1f}ms "
            f"({result.row_count} rows returned)."
        )
        return {
            "execution_result": {
                "success": True,
                "row_count": result.row_count,
                "columns": result.columns,
                "execution_time_ms": result.execution_time_ms,
                "error_message": None,
            },
            "query_results_df": result.data,
            "agent_logs": logs,
        }
    else:
        new_retry = retry_count + 1
        logs.append(
            f"⚠️ **SQL Engine Failure (Attempt {new_retry}/{state.get('max_retries', 3)}):** "
            f"{result.error_message}"
        )
        return {
            "execution_result": {
                "success": False,
                "row_count": 0,
                "columns": [],
                "execution_time_ms": result.execution_time_ms,
                "error_message": result.error_message,
            },
            "query_results_df": None,
            "retry_count": new_retry,
            "agent_logs": logs,
        }


def sql_recovery_node(state: AgentState, llm: LLMClient) -> dict[str, Any]:
    """Recovery Agent: Analyzes execution failure and produces repaired SQL."""
    logger.info(f"[Recovery Agent] Attempting SQL repair for attempt {state.get('retry_count', 1)}...")
    failed_query = state.get("sql_query", "")
    err_msg = state.get("execution_result", {}).get("error_message", "Unknown error")

    sys_prompt = SQL_CORRECTION_SYSTEM_PROMPT.format(
        table_name=state["table_name"],
        schema_context=state["schema_context"],
        failed_query=failed_query,
        error_message=err_msg,
    )

    repaired_sql = llm.generate(
        prompt=f"Fix this query based on the error: {err_msg}",
        system_prompt=sys_prompt,
    )

    logs = list(state.get("agent_logs", []))
    logs.append(f"🔧 **Recovery Agent:** Corrected SQL: `{repaired_sql.strip()}`")

    return {
        "sql_query": repaired_sql.strip(),
        "agent_logs": logs,
    }


def insight_node(state: AgentState, llm: LLMClient) -> dict[str, Any]:
    """Insight & Storytelling Agent: Synthesizes grounded insights and visual config."""
    logger.info("[Insight Agent] Generating final insights and visual config...")
    results_df: pd.DataFrame = state.get("query_results_df")
    exec_result = state.get("execution_result", {})
    logs = list(state.get("agent_logs", []))

    if results_df is None or len(results_df) == 0:
        if exec_result.get("success"):
            insights = "Query executed successfully, but returned 0 rows matching your criteria."
        else:
            insights = f"Unable to retrieve results due to SQL error: {exec_result.get('error_message')}"
        logs.append("📊 **Insight Agent:** Concluded with empty or failed result set.")
        return {
            "insights": insights,
            "chart_config": None,
            "is_complete": True,
            "agent_logs": logs,
        }

    from src.llm.query_planner import generate_grounded_insights
    if llm.is_available():
        preview_text = results_df.head(10).to_string(index=False)
        sys_prompt = INSIGHT_SYSTEM_PROMPT.format(
            user_query=state["user_query"],
            executed_sql=state.get("sql_query", ""),
            query_results_text=preview_text,
        )
        try:
            insights = llm.generate(
                prompt="Synthesize the primary business insights from these results.",
                system_prompt=sys_prompt,
            )
        except Exception:
            insights = generate_grounded_insights(state["user_query"], state.get("sql_query", ""), results_df)
    else:
        insights = generate_grounded_insights(state["user_query"], state.get("sql_query", ""), results_df)

    # Determine chart configuration
    chart_config = None
    cols = list(results_df.columns)
    if len(cols) >= 2:
        # Pick x and y candidate columns
        x_col = cols[0]
        y_col = cols[1]
        # If second column is numeric, keep it; else check others
        for col in cols[1:]:
            if pd.api.types.is_numeric_dtype(results_df[col]):
                y_col = col
                break

        chart_config = {
            "chart_type": state.get("chart_type", "bar"),
            "x_column": x_col,
            "y_column": y_col,
            "title": f"{state.get('user_query', 'Analysis Results')}",
        }

    logs.append("📊 **Insight Agent:** Synthesized grounded insights and configured charts.")
    return {
        "insights": insights,
        "chart_config": chart_config,
        "is_complete": True,
        "agent_logs": logs,
    }
