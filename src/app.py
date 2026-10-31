"""
Agentic Data Analyst — Complete Multi-Agent Interactive Web Dashboard
Integrates Profiling, LangGraph Multi-Agent Orchestration, DuckDB Engine,
Plotly Visualizations, SQLite History, and ChromaDB Semantic RAG.
"""

import sys
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from config import settings
from src.profiler import (
    load_dataset,
    save_uploaded_file,
    DatasetProfiler,
    IngestionError,
)
from src.sql_engine import DuckDBEngine
from src.llm import LLMClient
from src.agents import build_analysis_graph, AgentState
from src.visualizer import build_plotly_chart
from src.memory import SQLiteMemory, RAGKnowledgeBase, auto_enrich_dataset_glossary
from src.utils.logger import logger


@st.cache_resource
def get_shared_resources():
    """Initialize singleton engine, llm client, memory, and rag store."""
    llm = LLMClient()
    memory = SQLiteMemory()
    rag = RAGKnowledgeBase()
    return llm, memory, rag


def init_page():
    st.set_page_config(
        page_title="Agentic Data Analyst",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header():
    st.title("📊 Agentic Data Analyst")
    st.caption("Autonomous Multi-Step Data Analysis, Safe SQL Generation & Grounded Visual Insights")
    st.markdown("---")


def render_sidebar(llm: LLMClient, memory: SQLiteMemory, rag: RAGKnowledgeBase):
    with st.sidebar:
        st.header("📁 Dataset Ingestion")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help=f"Max file size: {settings.MAX_FILE_SIZE_MB}MB",
        )

        if uploaded_file is not None:
            if st.session_state.get("last_uploaded_name") != uploaded_file.name:
                try:
                    with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                        file_bytes = uploaded_file.getvalue()
                        save_uploaded_file(file_bytes, uploaded_file.name)
                        df, mapping = load_dataset(uploaded_file, uploaded_file.name)
                        profiler = DatasetProfiler()
                        profile = profiler.profile_dataframe(
                            df,
                            file_name=uploaded_file.name,
                            column_mapping=mapping,
                        )

                        # Auto-enrich and index rich business definitions into RAG store
                        enriched_defs = auto_enrich_dataset_glossary(profile.columns)
                        rag.index_definitions(enriched_defs)

                        st.session_state["current_df"] = df
                        st.session_state["current_profile"] = profile
                        st.session_state["column_mapping"] = mapping
                        st.session_state["last_uploaded_name"] = uploaded_file.name
                        st.success(f"Loaded '{uploaded_file.name}' with RAG indexing!")
                        st.rerun()
                except IngestionError as e:
                    st.error(f"Ingestion Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        sample_path = settings.PROJECT_ROOT / "data" / "samples" / "sales_data.csv"
        if sample_path.exists():
            st.markdown("— or —")
            if st.button("📊 Load Benchmark Dataset (Sales)", use_container_width=True):
                try:
                    with st.spinner("Loading benchmark dataset and indexing RAG knowledge base..."):
                        df, mapping = load_dataset(sample_path, "sales_data.csv")
                        profiler = DatasetProfiler()
                        profile = profiler.profile_dataframe(
                            df,
                            file_name="sales_data.csv",
                            column_mapping=mapping,
                        )
                        # Index rich domain glossary definitions
                        sample_defs = [
                            {"term": "revenue", "definition": "Total sales revenue in USD (Units Sold × Unit Price - Discount)."},
                            {"term": "units_sold", "definition": "Quantity of physical merchandise units purchased in an order."},
                            {"term": "unit_price", "definition": "Standard base price per single unit in USD."},
                            {"term": "discount", "definition": "Promotional deduction rate (0.00 = 0% discount, 0.10 = 10% discount)."},
                            {"term": "category", "definition": "High-level merchandise classification (Electronics, Furniture)."},
                            {"term": "city", "definition": "Metropolitan delivery market where order was fulfilled."},
                            {"term": "customer_rating", "definition": "Customer satisfaction score recorded on a 1.0 to 5.0 scale."},
                            {"term": "product_name", "definition": "Specific commercial product SKU description."},
                            {"term": "order_date", "definition": "Calendar timestamp recording when customer placed the transaction."},
                            {"term": "order_id", "definition": "Unique transactional primary key identifier."},
                        ]
                        rag.index_definitions(sample_defs)

                        st.session_state["current_df"] = df
                        st.session_state["current_profile"] = profile
                        st.session_state["column_mapping"] = mapping
                        st.session_state["last_uploaded_name"] = "sales_data.csv"
                        st.success("Loaded benchmark dataset successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading benchmark: {str(e)}")

        st.markdown("---")
        st.header("⚙️ Agent & Engine Config")
        llm_status = llm.get_status_label()
        st.markdown(f"**LLM Status:** {llm_status}")
        st.markdown(f"**Model:** `{settings.OLLAMA_MODEL}`")
        st.markdown(f"**Engine:** `DuckDB (In-Memory)`")
        st.markdown(f"**Orchestrator:** `LangGraph`")
        st.markdown(f"**Vector Store:** `ChromaDB` ({len(rag.get_all_definitions())} terms indexed)")

        # History Expander
        st.markdown("---")
        st.header("📜 Recent Queries")
        recent_history = memory.get_recent_history(limit=5)
        if recent_history:
            for item in recent_history:
                with st.expander(f"Q: {item['user_query'][:28]}..."):
                    st.caption(f"📅 {item['timestamp']} | Rows: {item['row_count']}")
                    st.code(item["sql_query"], language="sql")
        else:
            st.caption("No queries recorded yet.")


def generate_dynamic_suggestions(profile: Any) -> list[str]:
    """Generates 10 diverse, context-aware question chips based on actual columns in the dataset."""
    numeric_cols = [c.clean_name for c in profile.columns if c.inferred_type == "numeric"]
    categorical_cols = [c.clean_name for c in profile.columns if c.inferred_type in ("categorical", "text")]
    date_cols = [c.clean_name for c in profile.columns if c.inferred_type == "datetime"]

    # Helper for readable names
    def r(col: str) -> str:
        return col.replace("_", " ")

    suggestions = []

    # 1. Total metric by first category
    if numeric_cols and categorical_cols:
        suggestions.append(
            f"What is the total {r(numeric_cols[0])} by {r(categorical_cols[0])}?"
        )

    # 2. Top 5 category by first metric
    if categorical_cols and numeric_cols:
        cat = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
        num = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        suggestions.append(f"Show the top 5 {r(cat)} ranked by {r(num)}.")

    # 3. Average metric per category
    if numeric_cols and categorical_cols:
        num = numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[-1]
        cat = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[-1]
        suggestions.append(f"What is the average {r(num)} per {r(cat)}?")

    # 4. Monthly trend over time
    if date_cols and numeric_cols:
        suggestions.append(
            f"How does {r(numeric_cols[0])} trend month-over-month?"
        )

    # 5. Distribution / count per category
    if categorical_cols:
        suggestions.append(
            f"How many records are there for each {r(categorical_cols[0])}?"
        )

    # 6. Maximum metric per category
    if numeric_cols and categorical_cols:
        num = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        cat = categorical_cols[0]
        suggestions.append(f"Which {r(cat)} has the highest {r(num)}?")

    # 7. Minimum metric per category
    if numeric_cols and categorical_cols:
        num = numeric_cols[0]
        cat = categorical_cols[-1] if len(categorical_cols) > 1 else categorical_cols[0]
        suggestions.append(f"Which {r(cat)} has the lowest {r(num)}?")

    # 8. Compare two numeric metrics
    if len(numeric_cols) >= 2 and categorical_cols:
        suggestions.append(
            f"Compare {r(numeric_cols[0])} vs {r(numeric_cols[1])} across each {r(categorical_cols[0])}."
        )

    # 9. Bottom N performers
    if categorical_cols and numeric_cols:
        suggestions.append(
            f"List the bottom 5 {r(categorical_cols[0])} by {r(numeric_cols[0])}."
        )

    # 10. Overall summary statistics
    if numeric_cols:
        col_list = ", ".join([r(c) for c in numeric_cols[:3]])
        suggestions.append(
            f"Show summary statistics (min, max, average) for {col_list}."
        )

    # Fallback if dataset has no typed columns at all
    if not suggestions:
        suggestions = [
            "Show summary statistics for this dataset.",
            "Count the total number of records.",
            "List the first 10 rows of the dataset.",
        ]

    # Return up to 10 unique suggestions
    return suggestions[:10]


def run_agentic_pipeline(
    user_query: str,
    df: pd.DataFrame,
    profile: Any,
    llm: LLMClient,
    memory: SQLiteMemory,
    rag: RAGKnowledgeBase,
):
    """Executes the complete multi-agent workflow and returns state."""
    engine = DuckDBEngine()
    engine.register_dataset(df, table_name="dataset")

    # 1. Retrieve structured RAG glossary definitions
    rag_matches = rag.retrieve_structured(user_query, top_k=3)
    rag_context = rag.retrieve_relevant_definitions(user_query, top_k=3)

    # 2. Format complete schema context
    profiler = DatasetProfiler()
    base_schema = profiler.format_schema_for_llm(profile)
    full_context = f"{base_schema}\n\n### Semantic Knowledge / Dictionary:\n{rag_context}"

    # 3. Build and compile LangGraph
    graph = build_analysis_graph(llm=llm, engine=engine)

    initial_state: AgentState = {
        "user_query": user_query,
        "table_name": "dataset",
        "schema_context": full_context,
        "plan": None,
        "sql_query": None,
        "execution_result": None,
        "query_results_df": None,
        "retry_count": 0,
        "max_retries": settings.MAX_RETRY_COUNT,
        "insights": None,
        "chart_type": None,
        "chart_config": None,
        "agent_logs": [],
        "is_complete": False,
    }

    final_state = graph.invoke(initial_state)

    # Attach RAG retrieval information for transparent UI display
    final_state["rag_matches"] = rag_matches

    # 4. Persist to SQLite Memory
    exec_res = final_state.get("execution_result", {})
    memory.save_analysis(
        dataset_name=profile.file_name,
        user_query=user_query,
        sql_query=final_state.get("sql_query", ""),
        execution_success=bool(exec_res.get("success", False)),
        row_count=int(exec_res.get("row_count", 0)),
        execution_time_ms=float(exec_res.get("execution_time_ms", 0.0)),
        insights=final_state.get("insights", ""),
    )

    return final_state


def render_dashboard(llm: LLMClient, memory: SQLiteMemory, rag: RAGKnowledgeBase):
    df = st.session_state.get("current_df")
    profile = st.session_state.get("current_profile")

    if df is None or profile is None:
        st.info("👋 Welcome! Upload a dataset in the sidebar or click **'Load Benchmark Dataset'** to start.")
        st.markdown(
            """
            $$\\text{User Question} \\longrightarrow \\text{Supervisor Planner} \\longrightarrow \\text{RAG Context} \\longrightarrow \\text{SQL Analyst} \\longrightarrow \\text{DuckDB Safety Engine} \\longrightarrow \\text{Plotly Insights}$$
            """
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Python", "3.12")
        col2.metric("Engine", "DuckDB")
        col3.metric("Orchestration", "LangGraph")
        col4.metric("Visuals", "Plotly")
        return

    # Top KPI Metrics
    st.subheader(f"📊 Dataset: `{profile.file_name}`")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Rows", f"{profile.row_count:,}")
    kpi2.metric("Total Columns", profile.column_count)
    kpi3.metric("Missing Cells", f"{profile.missing_cell_percentage}%", delta=f"{profile.total_missing_cells} cells", delta_color="inverse")
    kpi4.metric("Duplicate Rows", profile.duplicate_row_count)
    kpi5.metric("Memory", f"{profile.memory_usage_mb:.2f} MB")

    # Main Tabs
    tab_chat, tab_preview, tab_schema, tab_rag, tab_history = st.tabs([
        "💬 Agentic Analysis",
        "📋 Dataset Preview",
        "🔍 Schema & Column Stats",
        "📚 RAG Data Dictionary",
        "📜 Audit History",
    ])

    with tab_chat:
        st.markdown("### Ask an Analytical Question")

        # Dynamic question suggestions generated specifically from this dataset's columns!
        example_queries = generate_dynamic_suggestions(profile)

        preset_col1, preset_col2 = st.columns([3, 1])
        with preset_col1:
            selected_preset = st.selectbox(
                "Quick Suggestions (Tailored to this dataset):",
                ["Custom Question..."] + example_queries,
            )

        default_text = "" if selected_preset == "Custom Question..." else selected_preset
        user_query = st.text_input("Enter your analytical question in plain English:", value=default_text)

        if st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True):
            if not user_query.strip():
                st.warning("Please enter an analytical question.")
            else:
                with st.spinner("Executing Multi-Agent Workflow (Supervisor ➔ RAG ➔ SQL Analyst ➔ DuckDB ➔ Plotly)..."):
                    result_state = run_agentic_pipeline(
                        user_query=user_query,
                        df=df,
                        profile=profile,
                        llm=llm,
                        memory=memory,
                        rag=rag,
                    )
                    st.session_state["latest_analysis"] = result_state

        # Display latest analysis output
        latest = st.session_state.get("latest_analysis")
        if latest and latest.get("user_query") == user_query:
            st.markdown("---")

            # 1. RAG Transparent Retrieval Card
            rag_matches = latest.get("rag_matches", [])
            with st.container():
                st.markdown("#### 📚 RAG Knowledge Retrieval & Grounding")
                if rag_matches:
                    st.caption("The vector store semantically retrieved these domain definitions to guide the SQL generation:")
                    rag_cols = st.columns(len(rag_matches))
                    for idx, match in enumerate(rag_matches):
                        with rag_cols[idx]:
                            st.markdown(
                                f"""
                                <div style="background-color: #f0f4f8; border-left: 4px solid #3366ff; padding: 10px; border-radius: 6px; margin-bottom: 10px;">
                                    <strong>🔍 {match['term']}</strong> <span style="font-size: 11px; background-color: #dbeafe; color: #1e40af; padding: 2px 6px; border-radius: 4px;">{match['relevance']}</span><br/>
                                    <small style="color: #4b5563;">{match['definition']}</small>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No specific custom business glossary terms required for this general query. Standard schema was used.")

            # 2. Agent Trace Accordion
            with st.expander("🔍 Live Multi-Agent Workflow Trace (LangGraph)", expanded=False):
                for log in latest.get("agent_logs", []):
                    st.markdown(log)

            exec_res = latest.get("execution_result", {})

            # 3. Grounded Insights Block
            if latest.get("insights"):
                st.markdown("### 💡 Grounded Analytical Insights")
                st.info(latest["insights"])

            # 4. Interactive Visualization
            res_df = latest.get("query_results_df")
            chart_cfg = latest.get("chart_config")
            if res_df is not None and not res_df.empty and chart_cfg:
                st.markdown("### 📈 Visual Analytics")
                fig = build_plotly_chart(
                    df=res_df,
                    chart_type=chart_cfg.get("chart_type", "bar"),
                    x_col=chart_cfg.get("x_column"),
                    y_col=chart_cfg.get("y_column"),
                    title=chart_cfg.get("title", "Analysis Results"),
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

            # 5. Query Results Table
            if res_df is not None and not res_df.empty:
                st.markdown(f"### 📋 Query Results ({len(res_df)} rows)")
                st.dataframe(res_df, use_container_width=True)

            # 6. Executed SQL Inspector
            if latest.get("sql_query"):
                with st.expander("💻 Safe SQL Inspector (DuckDB Columnar)", expanded=True):
                    col_s1, col_s2 = st.columns(2)
                    col_s1.metric("Execution Time", f"{exec_res.get('execution_time_ms', 0):.2f} ms")
                    col_s2.metric("Result Rows", exec_res.get("row_count", 0))
                    st.code(latest["sql_query"], language="sql")

    with tab_preview:
        st.markdown(f"Showing raw data preview ({min(len(df), 50)} rows):")
        st.dataframe(df.head(50), use_container_width=True)

    with tab_schema:
        schema_rows = []
        for col in profile.columns:
            stats_str = ""
            if col.inferred_type == "numeric":
                stats_str = f"Range: [{col.min_val} to {col.max_val}] | Avg: {col.mean_val}"
            elif col.inferred_type == "categorical" and col.top_categories:
                stats_str = "Top: " + ", ".join([f"{k} ({v})" for k, v in list(col.top_categories.items())[:3]])
            elif col.sample_values:
                stats_str = "Samples: " + ", ".join([str(s) for s in col.sample_values[:3]])

            schema_rows.append({
                "Column Name": col.clean_name,
                "Original Name": col.name,
                "Semantic Type": col.inferred_type,
                "Pandas Dtype": col.original_dtype,
                "Missing": f"{col.null_count} ({col.null_percentage}%)",
                "Uniques": col.unique_count,
                "Summary / Samples": stats_str,
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

    with tab_rag:
        st.markdown("### 📚 Semantic RAG Knowledge Base")
        st.markdown(
            "The **RAG (Retrieval-Augmented Generation)** subsystem indexes dataset dictionaries, business metrics, "
            "and column formulas in **ChromaDB**. When you ask a question, RAG retrieves matching definitions to "
            "prevent the SQL Agent from hallucinating nonexistent columns or misunderstanding business terminology."
        )

        col_r1, col_r2, col_r3 = st.columns(3)
        all_defs = rag.get_all_definitions()
        col_r1.metric("Indexed Knowledge Entries", len(all_defs))
        col_r2.metric("Vector Database", "ChromaDB")
        col_r3.metric("Retrieval Strategy", "Hybrid Semantic + Keyword")

        # Interactive Semantic Search Tester
        st.markdown("#### 🔍 Test Real-Time Semantic Retrieval")
        search_q = st.text_input("Enter a query to test what RAG retrieves:", value="What is revenue and rating?")
        if search_q:
            results = rag.retrieve_structured(search_q, top_k=4)
            if results:
                st.write("**Top Relevant Knowledge Entries Retrieved:**")
                for r in results:
                    st.markdown(
                        f"- **`{r['term']}`** *({r['relevance']} via {r['source']})*: {r['definition']}"
                    )
            else:
                st.info("No matching entries found.")

        st.markdown("---")
        st.markdown("#### 📖 Complete Data Dictionary & Glossary")
        if all_defs:
            st.dataframe(pd.DataFrame(all_defs), use_container_width=True)
        else:
            st.caption("No dictionary terms indexed yet.")

        # Allow user to add custom definitions
        with st.expander("➕ Register New Custom Business Definition"):
            with st.form("add_term_form"):
                new_term = st.text_input("Business Term / Column Name (e.g. Net Margin)")
                new_def = st.text_area("Definition / Business Formula (e.g. Calculated as Revenue minus Operating Costs)")
                submitted = st.form_submit_button("Index into Vector Store")
                if submitted:
                    if new_term.strip() and new_def.strip():
                        rag.index_definitions([{"term": new_term.strip(), "definition": new_def.strip()}])
                        st.success(f"Successfully indexed '{new_term}' into ChromaDB!")
                        st.rerun()
                    else:
                        st.warning("Please provide both a term and a definition.")

    with tab_history:
        st.markdown("### 📜 Audit History & Memory Logs")
        hist_rows = memory.get_recent_history(limit=50)
        if hist_rows:
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)
            if st.button("🗑️ Clear History"):
                memory.clear_history()
                st.success("History cleared!")
                st.rerun()
        else:
            st.info("No query history recorded yet.")


def main():
    logger.info("Initializing Agentic Data Analyst application...")
    init_page()
    llm, memory, rag = get_shared_resources()
    render_sidebar(llm, memory, rag)
    render_header()
    render_dashboard(llm, memory, rag)


if __name__ == "__main__":
    main()
