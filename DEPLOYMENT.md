# 🚀 Deployment Guide — Streamlit Community Cloud

This guide walks you through deploying the **Agentic Data Analyst** application to **Streamlit Community Cloud** with **zero cost** and a public URL for your project viva and portfolio.

---

## 📋 Deployed Repository Contents

The repository has been pruned so that **only files required for running the application in production** are tracked in Git:

```
.
├── .gitignore                  # Ignores development, test, and temporary files
├── .streamlit/
│   └── config.toml             # Streamlit cloud server & theme configuration
├── DEPLOYMENT.md               # Streamlit Cloud deployment steps
├── README.md                   # Project documentation & architecture
├── app.py                      # Root entrypoint proxy
├── config.py                   # Pydantic central application settings
├── requirements.txt            # Production dependencies (with Linux ChromaDB support)
├── data/
│   ├── samples/
│   │   ├── .gitkeep
│   │   └── sales_data.csv      # Benchmark dataset for instant demonstration
│   ├── uploads/
│   │   └── .gitkeep
│   └── vector_store/
│       └── .gitkeep
└── src/                        # Core application modules
    ├── __init__.py
    ├── app.py                  # Primary Streamlit interactive dashboard
    ├── agents/                 # LangGraph multi-agent orchestration
    ├── llm/                    # Local Ollama & Cloud Gemini LLM integration
    ├── memory/                 # SQLite audit memory & ChromaDB RAG store
    ├── profiler/               # CSV/Excel ingestion & schema profiler
    ├── sql_engine/             # DuckDB in-memory analytical query engine
    ├── utils/                  # Structured logging utility
    └── visualizer/             # Plotly interactive chart builder
```

---

## 🛠️ Step-by-Step Deployment Instructions

### Step 1: Push to GitHub

The repository is already pushed to GitHub:
**`https://github.com/rishilrsk/Agentic-AI.git`** (branch: `main`)

---

### Step 2: Deploy on Streamlit Community Cloud

1. Navigate to **[share.streamlit.io](https://share.streamlit.io/)** and sign in using your GitHub account.
2. Click the **"New app"** button.
3. Configure the deployment settings:
   * **Repository:** `rishilrsk/Agentic-AI`
   * **Branch:** `main`
   * **Main file path:** `app.py`
4. Click **"Deploy!"**.
5. Streamlit Community Cloud will automatically install dependencies from `requirements.txt` and launch your live application with a public HTTPS link.

---

### Step 3 (Optional): Enable Live Gemini Cloud AI

When deployed on Streamlit Cloud, local Ollama is not present. The system automatically runs in **Autonomous Mode** using its deterministic DuckDB query generator.

If you also want live generative AI answers in the cloud for free:
1. Obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).
2. On your deployed Streamlit Cloud dashboard, click **Settings** (⚙️) ➔ **Secrets**.
3. Add:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
4. Click **Save**. Streamlit will automatically restart with live Gemini Cloud AI enabled!

---

## 🔍 Verification on Streamlit Cloud

When your app launches:
1. In the sidebar, click **"📊 Load Benchmark Dataset (Sales)"**.
2. Select any quick analytical suggestion or type a question (e.g., *"What is the total revenue by category?"*).
3. Confirm that the multi-agent trace, DuckDB SQL, Plotly charts, and ChromaDB glossary cards appear instantly.
