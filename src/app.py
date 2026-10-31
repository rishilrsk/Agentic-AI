"""
Agentic Data Analyst — Professional Analytics Workspace
Crafted with clean UI/UX design principles: disciplined typography,
restrained color palette, structured dropdown query selection, and zero visual clutter.
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


def inject_custom_css():
    """Injects professional, modern enterprise dashboard styling (Stripe / Datadog style)."""
    st.markdown(
        """
        <style>
        /* Modern enterprise typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"], .stMarkdown, p, div {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            color: #1E293B;
        }

        /* Clean canvas */
        .stApp {
            background-color: #F8FAFC;
        }

        /* Top Navigation Bar */
        .app-navbar {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 14px 20px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
        }
        .app-title {
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            color: #0F172A;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .app-version-badge {
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
            background: #F1F5F9;
            color: #475569;
            border: 1px solid #E2E8F0;
        }
        .app-subtitle {
            font-size: 0.82rem;
            color: #64748B;
            margin-top: 2px;
        }
        .status-badge-group {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .status-badge {
            font-size: 0.75rem;
            font-weight: 500;
            padding: 3px 10px;
            border-radius: 4px;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            color: #475569;
        }
        .status-badge.active {
            background: #F0FDF4;
            color: #15803D;
            border-color: #DCFCE7;
        }

        /* Dataset Summary Strip */
        .dataset-strip {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 18px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        }
        .dataset-name-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .dataset-name {
            font-weight: 600;
            color: #0F172A;
            font-size: 0.92rem;
        }
        .dataset-meta-items {
            display: flex;
            align-items: center;
            gap: 16px;
            font-size: 0.82rem;
            color: #64748B;
        }
        .dataset-meta-item strong {
            color: #1E293B;
            font-weight: 600;
        }

        /* Analytical Card */
        .analytics-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        }
        .card-header-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #0F172A;
            margin-bottom: 14px;
        }

        /* Findings Callout */
        .findings-callout {
            background: #F8FAFC;
            border-left: 3px solid #2563EB;
            border: 1px solid #E2E8F0;
            border-left-width: 4px;
            border-left-color: #2563EB;
            border-radius: 6px;
            padding: 16px 18px;
            margin: 16px 0;
            font-size: 0.9rem;
            line-height: 1.6;
            color: #334155;
        }

        /* Clean Primary Button */
        div.stButton > button[kind="primary"],
        div.stButton > button {
            border-radius: 6px !important;
            font-weight: 500 !important;
            font-size: 0.875rem !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.15s ease !important;
        }
        div.stButton > button[kind="primary"] {
            background: #2563EB !important;
            border: 1px solid #1D4ED8 !important;
            color: #FFFFFF !important;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #1D4ED8 !important;
            border-color: #1E40AF !important;
        }

        /* Inputs & Selectbox */
        div[data-testid="stTextInput"] input,
        div[data-baseweb="select"] {
            border-radius: 6px !important;
            font-size: 0.875rem !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            border-bottom: 1px solid #E2E8F0 !important;
            gap: 16px !important;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            color: #64748B !important;
            padding: 10px 4px !important;
        }
        .stTabs [aria-selected="true"] {
            color: #2563EB !important;
            border-bottom: 2px solid #2563EB !important;
            background: transparent !important;
        }

        /* Code block */
        pre, code {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.82rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_benchmark_dataset(rag: RAGKnowledgeBase):
    """Loads benchmark sales dataset and indexes domain glossary definitions."""
    sample_path = settings.PROJECT_ROOT / "data" / "samples" / "sales_data.csv"
    if sample_path.exists():
        df, mapping = load_dataset(sample_path, "sales_data.csv")
        profiler = DatasetProfiler()
        profile = profiler.profile_dataframe(
            df,
            file_name="sales_data.csv",
            column_mapping=mapping,
        )
        sample_defs = [
            {"term": "revenue", "definition": "Total monetary volume in USD from completed sales transactions."},
            {"term": "units_sold", "definition": "Quantity of physical merchandise units purchased in an order."},
            {"term": "unit_price", "definition": "Base unit catalog price in USD before discounts."},
            {"term": "discount", "definition": "Promotional deduction rate (e.g. 0.10 represents 10% discount)."},
            {"term": "category", "definition": "High-level merchandise department classification (Electronics, Furniture)."},
            {"term": "city", "definition": "Delivery metropolitan market location."},
            {"term": "customer_rating", "definition": "Customer satisfaction score recorded on a 1.0 to 5.0 scale."},
            {"term": "product_name", "definition": "Specific commercial product SKU description."},
            {"term": "order_date", "definition": "Calendar timestamp recording when customer placed the transaction."},
            {"term": "order_id", "definition": "Unique transactional identifier."},
        ]
        rag.index_definitions(sample_defs)
        st.session_state["current_df"] = df
        st.session_state["current_profile"] = profile
        st.session_state["column_mapping"] = mapping
        st.session_state["last_uploaded_name"] = "sales_data.csv"


def render_navbar(llm: LLMClient, rag: RAGKnowledgeBase):
    st.markdown(
        f"""
        <div class="app-navbar">
            <div>
                <div class="app-title">
                    Agentic Data Analyst
                    <span class="app-version-badge">v1.0</span>
                </div>
                <div class="app-subtitle">Autonomous Multi-Agent Analytical Intelligence & Safe SQL Execution</div>
            </div>
            <div class="status-badge-group">
                <span class="status-badge active">● {llm.get_status_label()}</span>
                <span class="status-badge">DuckDB Columnar</span>
                <span class="status-badge">ChromaDB ({len(rag.get_all_definitions())} Terms)</span>
                <span class="status-badge">AST Safety Guard</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(llm: LLMClient, memory: SQLiteMemory, rag: RAGKnowledgeBase):
    with st.sidebar:
        st.subheader("Dataset Ingestion")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help=f"File size limit: {settings.MAX_FILE_SIZE_MB}MB",
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
                        enriched_defs = auto_enrich_dataset_glossary(profile.columns)
                        rag.index_definitions(enriched_defs)

                        st.session_state["current_df"] = df
                        st.session_state["current_profile"] = profile
                        st.session_state["column_mapping"] = mapping
                        st.session_state["last_uploaded_name"] = uploaded_file.name
                        st.success(f"Ingested '{uploaded_file.name}' successfully.")
                        st.rerun()
                except IngestionError as e:
                    st.error(f"Ingestion error: {str(e)}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        sample_path = settings.PROJECT_ROOT / "data" / "samples" / "sales_data.csv"
        if sample_path.exists():
            st.markdown("<div style='text-align: center; color: #94A3B8; font-size: 0.75rem; margin: 8px 0;'>OR</div>", unsafe_allow_html=True)
            if st.button("Load Benchmark Sales Dataset", use_container_width=True):
                try:
                    with st.spinner("Loading benchmark dataset..."):
                        load_benchmark_dataset(rag)
                        st.success("Benchmark dataset loaded.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error loading benchmark: {str(e)}")

        st.markdown("---")
        st.subheader("System Information")
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #475569; line-height: 1.8;">
                <div><strong>Engine:</strong> DuckDB (In-Memory Columnar)</div>
                <div><strong>Orchestration:</strong> LangGraph Multi-Agent</div>
                <div><strong>Knowledge Store:</strong> ChromaDB Semantic RAG</div>
                <div><strong>Audit Persistence:</strong> SQLite Database</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("Recent Queries")
        recent_history = memory.get_recent_history(limit=5)
        if recent_history:
            for item in recent_history:
                with st.expander(f"{item['user_query'][:24]}..."):
                    st.caption(f"{item['timestamp']} • {item['row_count']} rows")
                    st.code(item["sql_query"], language="sql")
        else:
            st.caption("No recent queries recorded.")


def generate_preset_questions(profile: Any) -> list[str]:
    """Generates clean, domain-appropriate questions tailored to dataset columns."""
    numeric_cols = [c.clean_name for c in profile.columns if c.inferred_type == "numeric"]
    categorical_cols = [c.clean_name for c in profile.columns if c.inferred_type in ("categorical", "text")]
    date_cols = [c.clean_name for c in profile.columns if c.inferred_type == "datetime"]

    def r(col: str) -> str:
        return col.replace("_", " ")

    options = []

    if numeric_cols and categorical_cols:
        options.append(f"What is the total {r(numeric_cols[0])} by {r(categorical_cols[0])}?")

    if categorical_cols and numeric_cols:
        cat = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
        num = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        options.append(f"Show the top 5 {r(cat)} ranked by {r(num)}.")

    if numeric_cols and categorical_cols:
        num = numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[-1]
        cat = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[-1]
        options.append(f"What is the average {r(num)} per {r(cat)}?")

    if date_cols and numeric_cols:
        options.append(f"How does {r(numeric_cols[0])} trend month-over-month?")

    if categorical_cols:
        options.append(f"How many records are there for each {r(categorical_cols[0])}?")

    if not options:
        options = [
            "Show summary statistics for this dataset.",
            "Count total records.",
            "List the first 10 rows.",
        ]

    return options


def run_agentic_pipeline(
    user_query: str,
    df: pd.DataFrame,
    profile: Any,
    llm: LLMClient,
    memory: SQLiteMemory,
    rag: RAGKnowledgeBase,
):
    """Executes the complete multi-agent workflow."""
    engine = DuckDBEngine()
    engine.register_dataset(df, table_name="dataset")

    # 1. Semantic RAG retrieval
    rag_matches = rag.retrieve_structured(user_query, top_k=3)
    rag_context = rag.retrieve_relevant_definitions(user_query, top_k=3)

    # 2. Schema formatting
    profiler = DatasetProfiler()
    base_schema = profiler.format_schema_for_llm(profile)
    full_context = f"{base_schema}\n\n### Semantic Knowledge / Dictionary:\n{rag_context}"

    # 3. LangGraph compilation and execution
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
    final_state["rag_matches"] = rag_matches

    # 4. Persistence
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

    # Empty State when no dataset is loaded
    if df is None or profile is None:
        st.markdown(
            """
            <div class="analytics-card" style="text-align: center; padding: 48px 24px; margin-top: 12px;">
                <h3 style="font-size: 1.25rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">
                    No Active Dataset Loaded
                </h3>
                <p style="color: #64748B; font-size: 0.9rem; max-width: 500px; margin: 0 auto 24px auto;">
                    Upload a CSV or Excel dataset from the sidebar to inspect schemas, execute safe analytical SQL,
                    and generate verified visualizations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_empty_1, col_empty_2, col_empty_3 = st.columns([1, 2, 1])
        with col_empty_2:
            if st.button("Load Benchmark Sales Dataset", type="primary", use_container_width=True):
                with st.spinner("Loading benchmark dataset..."):
                    load_benchmark_dataset(rag)
                    st.rerun()
        return

    # Dataset Summary Context Bar
    st.markdown(
        f"""
        <div class="dataset-strip">
            <div class="dataset-name-group">
                <span style="color: #64748B; font-size: 0.82rem;">Active Dataset:</span>
                <span class="dataset-name">{profile.file_name}</span>
            </div>
            <div class="dataset-meta-items">
                <div class="dataset-meta-item">Rows: <strong>{profile.row_count:,}</strong></div>
                <div class="dataset-meta-item">Columns: <strong>{profile.column_count}</strong></div>
                <div class="dataset-meta-item">Missing: <strong>{profile.missing_cell_percentage}%</strong></div>
                <div class="dataset-meta-item">Memory: <strong>{profile.memory_usage_mb:.2f} MB</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Tabs
    tab_analysis, tab_preview, tab_schema, tab_rag, tab_history = st.tabs([
        "Analysis",
        "Data Preview",
        "Schema & Profiling",
        "Data Dictionary (RAG)",
        "Audit Log",
    ])

    with tab_analysis:
        st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

        # Structured Question Selection (Dropdown restored per user request)
        preset_questions = generate_preset_questions(profile)
        dropdown_options = ["Select a preset analytical question..."] + preset_questions + ["Custom Question..."]

        selected_dropdown = st.selectbox(
            "Analytical Questions (Tailored to this dataset):",
            options=dropdown_options,
            index=0,
            help="Choose a pre-formulated analytical question or enter your own below.",
        )

        # Sync selection into text input
        initial_input = ""
        if selected_dropdown not in ("Select a preset analytical question...", "Custom Question..."):
            initial_input = selected_dropdown

        user_query = st.text_input(
            "Query Prompt:",
            value=initial_input,
            placeholder="Type your analytical question in plain English...",
        )

        col_run1, col_run2 = st.columns([5, 1])
        with col_run1:
            run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)
        with col_run2:
            if st.button("Clear", use_container_width=True):
                st.session_state["latest_analysis"] = None
                st.rerun()

        if run_clicked:
            if not user_query.strip():
                st.warning("Please select a preset question or enter a question prompt.")
            else:
                with st.spinner("Executing analytical workflow (Planning ➔ Retrieval ➔ DuckDB Execution)..."):
                    result_state = run_agentic_pipeline(
                        user_query=user_query,
                        df=df,
                        profile=profile,
                        llm=llm,
                        memory=memory,
                        rag=rag,
                    )
                    st.session_state["latest_analysis"] = result_state

        # Display Latest Analysis Result
        latest = st.session_state.get("latest_analysis")
        if latest and latest.get("user_query") == user_query:
            st.markdown("---")

            # 1. RAG Definitions Card (if relevant definitions were retrieved)
            rag_matches = latest.get("rag_matches", [])
            if rag_matches:
                with st.container():
                    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 8px;'>Retrieved Knowledge Definitions:</div>", unsafe_allow_html=True)
                    rag_cols = st.columns(len(rag_matches))
                    for idx, match in enumerate(rag_matches):
                        with rag_cols[idx]:
                            st.markdown(
                                f"""
                                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 14px; font-size: 0.82rem;">
                                    <div style="font-weight: 600; color: #0F172A;">{match['term']}</div>
                                    <div style="color: #64748B; font-size: 0.75rem; margin-top: 2px;">{match['definition']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

            exec_res = latest.get("execution_result", {})

            # 2. Findings Callout
            if latest.get("insights"):
                st.markdown(
                    f"""
                    <div class="findings-callout">
                        <div style="font-weight: 600; color: #0F172A; margin-bottom: 6px;">Key Analytical Findings</div>
                        <div>{latest['insights']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 3. Visualization
            res_df = latest.get("query_results_df")
            chart_cfg = latest.get("chart_config")
            if res_df is not None and not res_df.empty and chart_cfg:
                st.markdown("<div class='card-header-title'>Visual Analytics</div>", unsafe_allow_html=True)
                fig = build_plotly_chart(
                    df=res_df,
                    chart_type=chart_cfg.get("chart_type", "bar"),
                    x_col=chart_cfg.get("x_column"),
                    y_col=chart_cfg.get("y_column"),
                    title=chart_cfg.get("title", "Analysis Results"),
                )
                if fig:
                    fig.update_layout(
                        template="plotly_white",
                        font_family="Inter, sans-serif",
                        title_font_size=14,
                        margin=dict(l=20, r=20, t=35, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # 4. Result Records Table
            if res_df is not None and not res_df.empty:
                st.markdown(f"<div class='card-header-title'>Query Results ({len(res_df)} records)</div>", unsafe_allow_html=True)
                st.dataframe(res_df, use_container_width=True)

            # 5. Safe SQL Inspector
            if latest.get("sql_query"):
                with st.expander(f"Executed DuckDB SQL (Completed in {exec_res.get('execution_time_ms', 0):.1f} ms)"):
                    st.code(latest["sql_query"], language="sql")

            # 6. Workflow Trace
            with st.expander("Agent Execution Trace"):
                for log in latest.get("agent_logs", []):
                    st.markdown(log)

    with tab_preview:
        st.markdown(f"**Dataset Preview:** Showing top {min(len(df), 50)} rows of `{profile.file_name}`:")
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
                "Inferred Type": col.inferred_type,
                "Data Type": col.original_dtype,
                "Null Values": f"{col.null_count} ({col.null_percentage}%)",
                "Distinct Values": col.unique_count,
                "Summary Stats / Samples": stats_str,
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True)

    with tab_rag:
        st.markdown("### Semantic Data Dictionary")
        st.markdown(
            "The **RAG (Retrieval-Augmented Generation)** subsystem indexes business glossaries, field definitions, "
            "and metric calculation rules in **ChromaDB**. When queries are executed, relevant definitions are retrieved "
            "to prevent column hallucinations and align calculations with domain business logic."
        )

        col_r1, col_r2, col_r3 = st.columns(3)
        all_defs = rag.get_all_definitions()
        col_r1.metric("Indexed Definitions", len(all_defs))
        col_r2.metric("Vector Store", "ChromaDB")
        col_r3.metric("Retrieval Strategy", "Hybrid Semantic + Keyword")

        st.markdown("#### Test Real-Time Semantic Retrieval")
        search_q = st.text_input("Enter a query term to test retrieval:", value="revenue and rating")
        if search_q:
            results = rag.retrieve_structured(search_q, top_k=4)
            if results:
                st.write("**Top Matching Definitions:**")
                for r in results:
                    st.markdown(f"- **`{r['term']}`**: {r['definition']}")
            else:
                st.info("No matching definitions found.")

        st.markdown("---")
        st.markdown("#### Complete Knowledge Index")
        if all_defs:
            st.dataframe(pd.DataFrame(all_defs), use_container_width=True)
        else:
            st.caption("No dictionary terms indexed.")

        with st.expander("Register New Definition"):
            with st.form("add_def_form"):
                new_term = st.text_input("Business Term / Column Name")
                new_def = st.text_area("Definition or Calculation Formula")
                submitted = st.form_submit_button("Index into Knowledge Base")
                if submitted:
                    if new_term.strip() and new_def.strip():
                        rag.index_definitions([{"term": new_term.strip(), "definition": new_def.strip()}])
                        st.success(f"Indexed '{new_term}' into ChromaDB.")
                        st.rerun()
                    else:
                        st.warning("Please provide both a term and a definition.")

    with tab_history:
        st.markdown("### Audit Log & History")
        hist_rows = memory.get_recent_history(limit=50)
        if hist_rows:
            st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)
            if st.button("Clear Audit Log"):
                memory.clear_history()
                st.success("Audit history cleared.")
                st.rerun()
        else:
            st.info("No query history recorded yet.")


def main():
    logger.info("Initializing Agentic Data Analyst workspace...")
    init_page()
    inject_custom_css()
    llm, memory, rag = get_shared_resources()
    render_sidebar(llm, memory, rag)
    render_navbar(llm, rag)
    render_dashboard(llm, memory, rag)


if __name__ == "__main__":
    main()
