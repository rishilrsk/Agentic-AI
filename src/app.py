"""
Agentic Data Analyst — Modern SaaS Interactive Web Application
Linear/Supabase-inspired Minimalist UI for Multi-Agent Data Analysis,
Safe DuckDB Columnar Execution, Plotly Visualizations, and ChromaDB Semantic RAG.
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
        page_title="Agentic Data Analyst | Autonomous AI Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_saas_custom_css():
    """Injects high-end, modern SaaS styling (Linear / Supabase minimalist aesthetic)."""
    st.markdown(
        """
        <style>
        /* Import clean inter font */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #0F172A;
        }

        /* App Background */
        .stApp {
            background-color: #FAFAFC;
        }

        /* Top Header Styling */
        .saas-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 0 16px 0;
            border-bottom: 1px solid #E2E8F0;
            margin-bottom: 20px;
        }
        .saas-title {
            font-size: 1.65rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #0F172A;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .saas-tagline {
            font-size: 0.88rem;
            color: #64748B;
            margin-top: 2px;
        }

        /* Status Badge Pills */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.74rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 9999px;
            background: #F1F5F9;
            color: #334155;
            border: 1px solid #E2E8F0;
        }
        .status-pill.online {
            background: #ECFDF5;
            color: #065F46;
            border-color: #A7F3D0;
        }
        .status-pill.duckdb {
            background: #FFFBEB;
            color: #92400E;
            border-color: #FDE68A;
        }
        .status-pill.rag {
            background: #EFF6FF;
            color: #1E40AF;
            border-color: #BFDBFE;
        }

        /* KPI Metric Cards */
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-bottom: 22px;
        }
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        }
        .kpi-card:hover {
            border-color: #CBD5E1;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
            transform: translateY(-1px);
        }
        .kpi-label {
            font-size: 0.76rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748B;
            margin-bottom: 6px;
        }
        .kpi-val {
            font-size: 1.55rem;
            font-weight: 700;
            color: #0F172A;
            letter-spacing: -0.02em;
        }
        .kpi-sub {
            font-size: 0.75rem;
            color: #94A3B8;
            margin-top: 4px;
        }

        /* Modern SaaS Card Containers */
        .saas-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 22px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
            margin-bottom: 20px;
        }
        .saas-card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* Pipeline Flow Stepper */
        .stepper-container {
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 10px 14px;
            margin: 12px 0 18px 0;
            font-size: 0.8rem;
            font-weight: 600;
            color: #475569;
        }
        .step-item {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 8px;
            border-radius: 6px;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
        }
        .step-arrow {
            color: #94A3B8;
            font-size: 0.85rem;
        }

        /* Grounded Insights Box */
        .insights-card {
            background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
            border-left: 4px solid #2563EB;
            border-radius: 8px;
            padding: 18px 20px;
            margin: 16px 0;
            font-size: 0.94rem;
            line-height: 1.6;
            color: #1E293B;
        }

        /* Primary Button Elevation */
        div.stButton > button[kind="primary"],
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
            color: #FFFFFF !important;
            font-weight: 600;
            border-radius: 8px;
            border: none;
            padding: 0.55rem 1.25rem;
            box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
            transition: all 0.15s ease-in-out;
        }
        div.stButton > button[kind="primary"]:hover,
        div.stButton > button:first-child:hover {
            background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%);
            box-shadow: 0 6px 14px rgba(37, 99, 235, 0.3);
            transform: translateY(-1px);
        }

        /* Secondary/Preset Button Styling */
        .chip-button > div > button {
            background-color: #F8FAFC !important;
            color: #334155 !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 20px !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            padding: 4px 12px !important;
            box-shadow: none !important;
        }
        .chip-button > div > button:hover {
            background-color: #EFF6FF !important;
            border-color: #BFDBFE !important;
            color: #1D4ED8 !important;
            transform: translateY(-1px);
        }

        /* Clean Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
        }

        /* Clean Streamlit Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 1px solid #E2E8F0;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 16px;
            border-radius: 8px 8px 0 0;
            font-weight: 600;
            font-size: 0.88rem;
            color: #64748B;
        }
        .stTabs [aria-selected="true"] {
            color: #2563EB !important;
            border-bottom: 2px solid #2563EB !important;
            background-color: transparent !important;
        }

        /* Code Block Styling */
        pre, code {
            font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
            border-radius: 8px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(llm: LLMClient, rag: RAGKnowledgeBase):
    st.markdown(
        f"""
        <div class="saas-header">
            <div>
                <h1 class="saas-title">📊 Agentic Data Analyst</h1>
                <div class="saas-tagline">Autonomous Multi-Step Data Analysis, Safe SQL Generation & Grounded Insights</div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="status-pill online">⚡ {llm.get_status_label()}</span>
                <span class="status-pill duckdb">🦆 DuckDB Columnar</span>
                <span class="status-pill rag">📚 ChromaDB RAG ({len(rag.get_all_definitions())} Terms)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(llm: LLMClient, memory: SQLiteMemory, rag: RAGKnowledgeBase):
    with st.sidebar:
        st.markdown("### 📁 Dataset Workspace")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel dataset",
            type=["csv", "xlsx", "xls"],
            help=f"File limit: {settings.MAX_FILE_SIZE_MB}MB",
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
                        st.success(f"Loaded '{uploaded_file.name}' successfully!")
                        st.rerun()
                except IngestionError as e:
                    st.error(f"Ingestion Error: {str(e)}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        sample_path = settings.PROJECT_ROOT / "data" / "samples" / "sales_data.csv"
        if sample_path.exists():
            st.markdown(
                "<div style='text-align: center; color: #94A3B8; font-size: 0.8rem; margin: 8px 0;'>— OR QUICK START —</div>",
                unsafe_allow_html=True,
            )
            if st.button("📊 Load Benchmark Dataset (Sales)", use_container_width=True):
                try:
                    with st.spinner("Loading benchmark sales dataset..."):
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
        st.markdown("### ⚙️ System Specifications")
        st.markdown(
            f"""
            <div style="font-size: 0.84rem; color: #475569; line-height: 1.8;">
                <div><strong>• Orchestrator:</strong> LangGraph Multi-Agent</div>
                <div><strong>• Analytical SQL:</strong> DuckDB In-Memory</div>
                <div><strong>• Vector Engine:</strong> ChromaDB</div>
                <div><strong>• Memory Store:</strong> SQLite Audit Log</div>
                <div><strong>• Visualizer:</strong> Plotly Graph Objects</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### 📜 Recent Audit Logs")
        recent_history = memory.get_recent_history(limit=5)
        if recent_history:
            for item in recent_history:
                with st.expander(f"Q: {item['user_query'][:24]}..."):
                    st.caption(f"📅 {item['timestamp']} | Rows: {item['row_count']}")
                    st.code(item["sql_query"], language="sql")
        else:
            st.caption("No queries recorded yet.")


def generate_dynamic_suggestions(profile: Any) -> list[str]:
    """Generates context-aware question chips based on actual columns in the dataset."""
    numeric_cols = [c.clean_name for c in profile.columns if c.inferred_type == "numeric"]
    categorical_cols = [c.clean_name for c in profile.columns if c.inferred_type in ("categorical", "text")]
    date_cols = [c.clean_name for c in profile.columns if c.inferred_type == "datetime"]

    def r(col: str) -> str:
        return col.replace("_", " ")

    suggestions = []

    if numeric_cols and categorical_cols:
        suggestions.append(f"What is the total {r(numeric_cols[0])} by {r(categorical_cols[0])}?")

    if categorical_cols and numeric_cols:
        cat = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[0]
        num = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        suggestions.append(f"Show the top 5 {r(cat)} ranked by {r(num)}.")

    if numeric_cols and categorical_cols:
        num = numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[-1]
        cat = categorical_cols[1] if len(categorical_cols) > 1 else categorical_cols[-1]
        suggestions.append(f"What is the average {r(num)} per {r(cat)}?")

    if date_cols and numeric_cols:
        suggestions.append(f"How does {r(numeric_cols[0])} trend month-over-month?")

    if categorical_cols:
        suggestions.append(f"How many records are there for each {r(categorical_cols[0])}?")

    if not suggestions:
        suggestions = [
            "Show summary statistics for this dataset.",
            "Count the total number of records.",
            "List the first 10 rows of the dataset.",
        ]

    return suggestions[:5]


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
        st.markdown(
            """
            <div class="saas-card" style="text-align: center; padding: 48px 24px;">
                <div style="font-size: 2.8rem; margin-bottom: 12px;">📊</div>
                <h2 style="font-weight: 700; color: #0F172A; margin-bottom: 8px;">Welcome to Agentic Data Analyst</h2>
                <p style="color: #64748B; max-width: 600px; margin: 0 auto 24px auto; font-size: 0.95rem;">
                    Upload any CSV/Excel file in the sidebar, or click <strong>'Load Benchmark Dataset'</strong> 
                    to explore autonomous multi-agent analysis, safe DuckDB SQL queries, and Plotly visual insights.
                </p>
                <div class="stepper-container" style="justify-content: center; max-width: 720px; margin: 0 auto;">
                    <span class="step-item">🧠 Supervisor Planner</span>
                    <span class="step-arrow">➔</span>
                    <span class="step-item">📚 ChromaDB RAG</span>
                    <span class="step-arrow">➔</span>
                    <span class="step-item">🦆 DuckDB Columnar</span>
                    <span class="step-arrow">➔</span>
                    <span class="step-item">🛡️ Safe AST Check</span>
                    <span class="step-arrow">➔</span>
                    <span class="step-item">📈 Plotly Insights</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Top KPI Metrics Cards
    st.markdown(
        f"""
        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-label">Active Dataset</div>
                <div class="kpi-val" style="font-size: 1.15rem; word-break: break-all;">{profile.file_name}</div>
                <div class="kpi-sub">Ready for analysis</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Records</div>
                <div class="kpi-val">{profile.row_count:,}</div>
                <div class="kpi-sub">Rows ingested</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Attributes</div>
                <div class="kpi-val">{profile.column_count}</div>
                <div class="kpi-sub">Columns profiled</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Data Quality</div>
                <div class="kpi-val" style="color: {'#166534' if profile.missing_cell_percentage == 0 else '#B45309'};">{100 - profile.missing_cell_percentage:.1f}%</div>
                <div class="kpi-sub">{profile.total_missing_cells} missing cells</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Memory Footprint</div>
                <div class="kpi-val">{profile.memory_usage_mb:.2f} <span style="font-size: 0.9rem; font-weight: 500;">MB</span></div>
                <div class="kpi-sub">In-memory cache</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Main Tabs
    tab_chat, tab_preview, tab_schema, tab_rag, tab_history = st.tabs([
        "💬 Agentic Analysis",
        "📋 Data Preview",
        "🔍 Schema & Stats",
        "📚 RAG Knowledge Base",
        "📜 Audit History",
    ])

    with tab_chat:
        st.markdown(
            """
            <div class="stepper-container">
                <span style="color: #64748B;">Multi-Agent Pipeline:</span>
                <span class="step-item">1. Supervisor Planner</span>
                <span class="step-arrow">➔</span>
                <span class="step-item">2. Semantic RAG Grounding</span>
                <span class="step-arrow">➔</span>
                <span class="step-item">3. DuckDB SQL Analyst</span>
                <span class="step-arrow">➔</span>
                <span class="step-item">4. Plotly Insight Generation</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dynamic Quick-Question Pill Chips
        suggestions = generate_dynamic_suggestions(profile)
        st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #64748B; margin-bottom: 6px;'>💡 Quick Analytical Prompts (1-Click):</div>", unsafe_allow_html=True)
        
        chip_cols = st.columns(len(suggestions))
        for idx, s in enumerate(suggestions):
            with chip_cols[idx]:
                short_text = s if len(s) < 28 else s[:25] + "..."
                if st.button(f"🔍 {short_text}", key=f"chip_{idx}", use_container_width=True, help=s):
                    st.session_state["active_query_input"] = s

        user_query = st.text_input(
            "Enter your analytical question in plain English:",
            value=st.session_state.get("active_query_input", ""),
            placeholder="e.g. What is the total revenue by category? or Show the top 5 products by units sold.",
        )

        col_btn1, col_btn2 = st.columns([4, 1])
        with col_btn1:
            run_clicked = st.button("🚀 Run Multi-Agent Analysis", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🔄 Clear", use_container_width=True):
                st.session_state["active_query_input"] = ""
                st.session_state["latest_analysis"] = None
                st.rerun()

        if run_clicked:
            if not user_query.strip():
                st.warning("Please enter an analytical question or click one of the quick suggestions above.")
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
                st.markdown("<div class='saas-card-title'>📚 Semantic RAG Knowledge Retrieved</div>", unsafe_allow_html=True)
                if rag_matches:
                    rag_cols = st.columns(len(rag_matches))
                    for idx, match in enumerate(rag_matches):
                        with rag_cols[idx]:
                            st.markdown(
                                f"""
                                <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-left: 3px solid #2563EB; padding: 12px; border-radius: 8px;">
                                    <div style="font-weight: 600; color: #0F172A; font-size: 0.88rem;">🔍 {match['term']}</div>
                                    <div style="font-size: 0.72rem; color: #2563EB; font-weight: 600; margin: 2px 0 6px 0;">{match['relevance']}</div>
                                    <div style="font-size: 0.8rem; color: #475569; line-height: 1.4;">{match['definition']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                else:
                    st.info("No custom business glossary terms required for this general query. Standard schema was applied.")

            # 2. Live Agent Trace Accordion
            with st.expander("🔍 Live Multi-Agent Workflow Trace (LangGraph)", expanded=False):
                for log in latest.get("agent_logs", []):
                    st.markdown(log)

            exec_res = latest.get("execution_result", {})

            # 3. Grounded Insights Card
            if latest.get("insights"):
                st.markdown(
                    f"""
                    <div class="insights-card">
                        <div style="font-size: 1.05rem; font-weight: 700; color: #1E3A8A; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                            💡 Mathematically Grounded Findings
                        </div>
                        <div>{latest['insights']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 4. Interactive Visualization
            res_df = latest.get("query_results_df")
            chart_cfg = latest.get("chart_config")
            if res_df is not None and not res_df.empty and chart_cfg:
                st.markdown("<div class='saas-card-title'>📈 Visual Analytics</div>", unsafe_allow_html=True)
                fig = build_plotly_chart(
                    df=res_df,
                    chart_type=chart_cfg.get("chart_type", "bar"),
                    x_col=chart_cfg.get("x_column"),
                    y_col=chart_cfg.get("y_column"),
                    title=chart_cfg.get("title", "Analysis Results"),
                )
                if fig:
                    # Apply clean minimalist layout styling to plotly chart
                    fig.update_layout(
                        template="plotly_white",
                        font_family="Plus Jakarta Sans, sans-serif",
                        title_font_size=15,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # 5. Query Results Table
            if res_df is not None and not res_df.empty:
                st.markdown(f"<div class='saas-card-title'>📋 Query Results ({len(res_df)} rows)</div>", unsafe_allow_html=True)
                st.dataframe(res_df, use_container_width=True)

            # 6. Executed Safe SQL Inspector
            if latest.get("sql_query"):
                with st.expander("💻 Safe SQL Inspector (DuckDB Columnar)", expanded=True):
                    col_s1, col_s2 = st.columns(2)
                    col_s1.metric("Execution Latency", f"{exec_res.get('execution_time_ms', 0):.2f} ms")
                    col_s2.metric("Result Rows", exec_res.get("row_count", 0))
                    st.code(latest["sql_query"], language="sql")

    with tab_preview:
        st.markdown(f"**Dataset Preview:** Showing first {min(len(df), 50)} rows of `{profile.file_name}`:")
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
        col_r2.metric("Vector Engine", "ChromaDB")
        col_r3.metric("Retrieval Strategy", "Hybrid Semantic + Keyword")

        st.markdown("#### 🔍 Test Real-Time Semantic Retrieval")
        search_q = st.text_input("Enter a query to test what RAG retrieves:", value="What is revenue and rating?")
        if search_q:
            results = rag.retrieve_structured(search_q, top_k=4)
            if results:
                st.write("**Top Relevant Knowledge Entries Retrieved:**")
                for r in results:
                    st.markdown(f"- **`{r['term']}`** *({r['relevance']} via {r['source']})*: {r['definition']}")
            else:
                st.info("No matching entries found.")

        st.markdown("---")
        st.markdown("#### 📖 Complete Data Dictionary & Glossary")
        if all_defs:
            st.dataframe(pd.DataFrame(all_defs), use_container_width=True)
        else:
            st.caption("No dictionary terms indexed yet.")

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
    inject_saas_custom_css()
    llm, memory, rag = get_shared_resources()
    render_sidebar(llm, memory, rag)
    render_header(llm, rag)
    render_dashboard(llm, memory, rag)


if __name__ == "__main__":
    main()
