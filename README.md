# 📊 Agentic Data Analyst

> An Intelligent Multi-Step Data Analysis and Insight Generation System powered by LangGraph, DuckDB, Local LLMs (Ollama), and Plotly.

---

## 🎯 Overview

The **Agentic Data Analyst** enables users to upload CSV or Excel datasets and ask complex analytical questions in natural language. Instead of acting as a naive text-to-SQL chatbot, the system executes an autonomous multi-agent workflow:

$$\text{User Goal} \longrightarrow \text{Planning \& Decomposition} \longrightarrow \text{Dataset Profiling} \longrightarrow \text{RAG Glossary Retrieval} \longrightarrow \text{Safe SQL Execution} \longrightarrow \text{Self-Correction \& Validation} \longrightarrow \text{Plotly Insights}$$

---

## 🚀 Key Features

* **Multi-Agent Orchestration:** Structured supervisor architecture implemented in **LangGraph**.
* **Zero Numerical Hallucination:** Computations are deterministically executed by **DuckDB** in-memory columnar engine.
* **Self-Healing SQL:** Automatic syntax validation, AST inspection, column existence checks, and bounded retry loops.
* **Security Hardening:** Strictly enforces single read-only `SELECT` statements; blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, and system procedures.
* **Semantic RAG:** Uses **ChromaDB** to index dataset dictionaries and business definitions for accurate context retrieval.
* **Persistent History:** SQLite-backed audit log tracking user questions, generated SQL, execution metrics, and insights.
* **Interactive UI:** Dynamic dashboard with live execution status, SQL inspection, and **Plotly** visual analytics built with **Streamlit**.
* **Local-First & Zero-Cost:** Designed to run entirely offline with **Ollama** (`qwen2.5:3b` or `llama3.2:3b`), with seamless deterministic fallback when offline.

---

## 🏗️ Project Architecture

```
User Input
    │
    ▼
Streamlit Frontend ───► LangGraph Orchestrator
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Supervisor / Planner Agent         Dataset Profiler
            │                                 │
            ▼                                 ▼
   RAG Knowledge Base (ChromaDB)      SQL Analyst (DuckDB)
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                Validation & Recovery Agent
                             │
                             ▼
                Insight & Visualization (Plotly)
```

---

## 📁 Repository Structure

```
.
├── .env.example               # Configuration template
├── config.py                  # Pydantic central settings
├── requirements.txt           # Pinned production dependencies
├── README.md                  # Project documentation
├── evaluate.py                # Automated benchmark evaluation suite
├── evaluation_results.json    # Real-world benchmark evaluation results
├── src/
│   ├── app.py                 # Streamlit web interface entrypoint
│   ├── profiler/              # Phase 3: Dataset Ingestion & Profiling
│   │   ├── __init__.py
│   │   ├── dataset_profiler.py # Schema inspection, statistics & LLM prompt formatter
│   │   ├── ingestion.py       # File validation, CSV/Excel parser & sanitization
│   │   └── models.py          # ColumnProfile & DatasetProfile Pydantic models
│   ├── sql_engine/            # Phase 4: DuckDB Analytical Engine & Security
│   │   ├── __init__.py
│   │   ├── executor.py        # In-memory DuckDB query runner with timing
│   │   └── validator.py       # AST validation, read-only enforcement & limits
│   ├── llm/                   # Phase 5: LLM Client & Prompt Engineering
│   │   ├── __init__.py
│   │   ├── client.py          # Ollama integration with offline fallback
│   │   └── prompts.py         # System prompts for Planner, SQL, Fixer, Insight
│   ├── agents/                # Phase 6: LangGraph Multi-Agent Orchestration
│   │   ├── __init__.py
│   │   ├── state.py           # Strongly-typed AgentState dictionary
│   │   ├── nodes.py           # Supervisor, SQL Analyst, Fixer, Insight nodes
│   │   └── graph.py           # Compiled state graph & routing logic
│   ├── visualizer/            # Phase 7: Plotly Visualization Engine
│   │   ├── __init__.py
│   │   └── chart_builder.py   # Bar, Line, Scatter, Pie chart generators
│   ├── memory/                # Phase 8: Persistence & RAG
│   │   ├── __init__.py
│   │   ├── sqlite_memory.py   # SQLite persistent query history
│   │   └── rag_store.py       # ChromaDB semantic data dictionary store
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # Structured logging utility
├── data/
│   ├── uploads/               # Uploaded user datasets
│   ├── samples/               # Benchmark test datasets (sales_data.csv)
│   ├── vector_store/          # ChromaDB persistence directory
│   └── analysis_history.db    # SQLite query history database
└── tests/
    ├── __init__.py
    ├── test_scaffold.py       # Configuration and setup test suite
    ├── test_profiler.py       # Ingestion and Profiler test suite
    ├── test_sql_engine.py     # SQL safety, limits & DuckDB test suite
    ├── test_llm.py            # LLM client & prompt test suite
    ├── test_graph.py          # LangGraph multi-agent integration test
    ├── test_visualizer.py     # Plotly visualization engine test suite
    └── test_memory_rag.py     # SQLite history & ChromaDB RAG test suite
```

---

## 🛠️ Quickstart

### 1. Prerequisites
* Python 3.12 (64-bit)
* Ollama installed (optional, autonomous fallback is active when Ollama is offline)

### 2. Virtual Environment Setup
```powershell
# Using uv (recommended)
uv venv --python 3.12 .venv
.\.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### 3. Running the Application
```powershell
streamlit run src/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.

### 4. Running Verification Tests
Execute the full test battery:
```powershell
.venv\Scripts\python.exe -m pytest tests/
```
*(All 33 unit and integration tests pass)*

### 5. Running Benchmark Evaluation
Generate real performance metrics:
```powershell
.venv\Scripts\python.exe evaluate.py
```

---

## 📊 Measured Benchmark Results

| Metric | Result | Evaluation Method |
|---|---|---|
| **Analytical Tasks Evaluated** | 5 / 5 | Full multi-agent execution battery |
| **Task Success Rate** | **100.0%** | End-to-end question to insight |
| **SQL Execution Success Rate** | **100.0%** | Zero syntax or catalog failures |
| **Security Rejection Rate** | **100.0% (5/5)** | AST validation blocks DDL/DML/PRAGMA |
| **RAG Retrieval Quality** | **100.0%** | ChromaDB semantic term match |
| **Average Pipeline Latency** | **1,899 ms** | Sub-2-second end-to-end response |
