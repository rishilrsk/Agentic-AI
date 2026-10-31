"""
LangGraph Multi-Agent Orchestration Workflow.
Assembles Supervisor, SQL Analyst, DuckDB Executor, Recovery, and Insight nodes into a reactive state graph.
"""

from functools import partial
from langgraph.graph import StateGraph, START, END

from src.agents.state import AgentState
from src.agents.nodes import (
    supervisor_node,
    sql_generator_node,
    sql_executor_node,
    sql_recovery_node,
    insight_node,
)
from src.llm import LLMClient
from src.sql_engine import DuckDBEngine
from src.utils.logger import logger


def route_after_execution(state: AgentState) -> str:
    """Conditional router determining whether to proceed to insights or attempt self-correction."""
    exec_result = state.get("execution_result", {})
    if exec_result.get("success", False):
        return "visualizer"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if retry_count < max_retries:
        logger.info(f"Routing to recovery loop (Attempt {retry_count}/{max_retries})")
        return "sql_recovery"

    logger.warning("Max recovery retries reached. Routing to final insight node to report failure.")
    return "visualizer"


def build_analysis_graph(llm: LLMClient, engine: DuckDBEngine):
    """
    Constructs and compiles the multi-agent LangGraph workflow.
    """
    graph = StateGraph(AgentState)

    # Add agent nodes with injected dependencies
    graph.add_node("supervisor", partial(supervisor_node, llm=llm))
    graph.add_node("sql_analyst", partial(sql_generator_node, llm=llm))
    graph.add_node("sql_executor", partial(sql_executor_node, engine=engine))
    graph.add_node("sql_recovery", partial(sql_recovery_node, llm=llm))
    graph.add_node("visualizer", partial(insight_node, llm=llm))

    # Connect edges
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "sql_analyst")
    graph.add_edge("sql_analyst", "sql_executor")

    # Conditional routing from executor
    graph.add_conditional_edges(
        "sql_executor",
        route_after_execution,
        {
            "visualizer": "visualizer",
            "sql_recovery": "sql_recovery",
        },
    )

    # Recovery loop points back to execution
    graph.add_edge("sql_recovery", "sql_executor")
    graph.add_edge("visualizer", END)

    compiled_graph = graph.compile()
    logger.info("Successfully compiled LangGraph agentic workflow.")
    return compiled_graph
