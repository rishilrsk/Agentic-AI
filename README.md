# Agentic Data Analyst

A multi-agent data analysis platform built with LangGraph, DuckDB, ChromaDB, and Streamlit. The application allows users to upload structured datasets (CSV/Excel) and query them using natural language. Instead of prompting an LLM to guess numerical answers, the system plans analytical tasks, retrieves domain definitions, generates read-only DuckDB SQL, validates queries for safety, and renders interactive Plotly visualizations.

---

## Overview

Traditional text-to-SQL or conversational AI tools often suffer from mathematical hallucinations when calculating sums, averages, and group metrics directly. Agentic Data Analyst addresses this by separating reasoning from computation:

1. **Supervisor Planner:** Decomposes the user's analytical query into structured sub-tasks.
2. **Semantic Knowledge Retrieval (RAG):** Uses ChromaDB to look up column descriptions, calculation formulas, and business definitions to ground the query.
3. **Deterministic SQL Analyst:** Generates standard SQL executed directly against DuckDB's in-memory columnar engine.
4. **AST Safety & Validation:** Inspects queries before execution, strictly enforcing single read-only `SELECT` statements and blocking any data modification or system commands.
5. **Insights & Visuals:** Translates query outputs into evidence-based summaries and interactive Plotly charts.
6. **Audit Memory:** Records query history and execution metrics into a persistent SQLite database.

---

## System Architecture

```
User Query
    │
    ▼
Streamlit Interface ───► LangGraph Orchestrator
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Supervisor Planner Agent            Dataset Profiler
            │                                 │
            ▼                                 ▼
   ChromaDB Knowledge Store            DuckDB SQL Analyst
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                 AST Validation & Guard
                             │
                             ▼
                Plotly Visuals & Insights
                             │
                             ▼
                   SQLite Audit History
```

---

## Tech Stack

| Component | Technology | Role |
|---|---|---|
| **Frontend** | Streamlit | Web dashboard, dataset upload, and interactive UI |
| **Agent Workflow** | LangGraph | State management and multi-agent coordination |
| **SQL Engine** | DuckDB | In-memory columnar analytical query execution |
| **Knowledge Store** | ChromaDB | Vector database for data dictionary and glossary RAG |
| **Visualizations** | Plotly | Dynamic bar, line, pie, and scatter charts |
| **Persistence** | SQLite | Query history, performance metrics, and audit log |
| **LLM Support** | Ollama / Gemini API | Local-first model execution with cloud fallback |

---

## Project Structure

```
.
├── .streamlit/
│   └── config.toml             # Streamlit server and theme configuration
├── app.py                      # Root entrypoint proxy
├── config.py                   # Central settings and environment config
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/
│   ├── samples/
│   │   └── sales_data.csv      # Sample sales dataset for immediate testing
│   ├── uploads/                # Directory for user-uploaded datasets
│   └── vector_store/           # ChromaDB persistent storage directory
└── src/
    ├── app.py                  # Streamlit application layout and components
    ├── agents/                 # LangGraph state definition, nodes, and graph
    ├── llm/                    # Ollama client and prompt templates
    ├── memory/                 # SQLite audit logger and ChromaDB RAG store
    ├── profiler/               # Ingestion, file validation, and schema profiling
    ├── sql_engine/             # DuckDB executor and AST query safety validator
    ├── utils/                  # Structured logging utility
    └── visualizer/             # Plotly chart generation logic
```

---

## Getting Started

### Prerequisites
* Python 3.12 (64-bit recommended)
* (Optional) [Ollama](https://ollama.com/) running locally with `qwen2.5:3b` or `llama3.2:3b`. If Ollama is not installed or offline, the system operates in autonomous mode using its deterministic query engine.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rishilrsk/Agentic-AI.git
   cd Agentic-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .\.venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the application:
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## Configuration

Settings can be customized via environment variables or a `.env` file:

```ini
# Ollama Configuration
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="qwen2.5:3b"

# Optional Cloud LLM (for Streamlit Cloud or environments without local Ollama)
GEMINI_API_KEY=""

# Limits
MAX_FILE_SIZE_MB=50
MAX_QUERY_EXEC_SECONDS=10
MAX_RESULT_ROWS=500
```

---

## Security & Query Safety

Because the application converts natural language into executable database queries, it enforces strict security boundaries:

* **Read-Only Enforcement:** Queries must parse strictly as a single `SELECT` statement.
* **Prohibited Keywords:** Any presence of `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `ATTACH`, `COPY`, `PRAGMA`, or administrative commands triggers immediate rejection.
* **Multi-Statement Blocking:** Semicolon-separated injection payloads are blocked at the AST parser level.
* **Timeout and Row Limits:** All DuckDB executions enforce strict execution timeouts and row count limits.
