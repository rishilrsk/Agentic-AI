# 🚀 Deployment Guide — Agentic Data Analyst

This document provides step-by-step instructions to deploy the **Agentic Data Analyst** application to free cloud platforms, Docker containers, or local networks with **zero cost** and **100% reliability**.

---

## 📑 Table of Contents
1. [Pre-Deployment Readiness Checklist](#1-pre-deployment-readiness-checklist)
2. [Option A: Streamlit Community Cloud (Recommended & Free)](#2-option-a-streamlit-community-cloud-recommended--free)
3. [Option B: Hugging Face Spaces (Free)](#3-option-b-hugging-face-spaces-free)
4. [Option C: Docker Containerization](#4-option-c-docker-containerization)
5. [Option D: Render / Railway Deployment](#5-option-d-render--railway-deployment)
6. [Cloud LLM & Offline Autonomous Mode](#6-cloud-llm--offline-autonomous-mode)
7. [Troubleshooting & Verification](#7-troubleshooting--verification)

---

## 1. Pre-Deployment Readiness Checklist

Before pushing to any cloud platform, ensure your project includes:
- [x] **Root Entrypoint Proxy:** [`app.py`](file:///app.py) points directly to `src/app.py`.
- [x] **Pinned Dependencies:** [`requirements.txt`](file:///requirements.txt) includes `pysqlite3-binary` for Linux ChromaDB compatibility.
- [x] **Streamlit Server Config:** [`.streamlit/config.toml`](file:///.streamlit/config.toml) with headless mode, CORS settings, and 50MB upload limits.
- [x] **Container Definition:** [`Dockerfile`](file:///Dockerfile) and [`.dockerignore`](file:///.dockerignore) ready for container builds.
- [x] **Sample Dataset:** [`data/samples/sales_data.csv`](file:///data/samples/sales_data.csv) pre-packaged for instant demonstration.
- [x] **Dual Execution Support:** Autonomous deterministic rule-based engine active when Ollama is offline, with optional Gemini API cloud fallback.

---

## 2. Option A: Streamlit Community Cloud (Recommended & Free)

Streamlit Community Cloud is the fastest, free method to host your application with a public URL (e.g., `https://agentic-data-analyst.streamlit.app`).

### Step 1: Initialize Git and Push to GitHub
```powershell
# In the project root directory:
git init
git add .
git commit -m "Initial commit: Agentic Data Analyst ready for deployment"

# Link to your GitHub repository:
git branch -M main
git remote add origin https://github.com/<your-username>/Agentic-Data-Analyst.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository: `<your-username>/Agentic-Data-Analyst`.
4. Branch: `main`.
5. Main file path: `app.py` (or `src/app.py`).
6. Click **"Deploy!"**.

### Step 3 (Optional): Configure Secrets for Live Cloud LLM
If you wish to enable live generative AI responses in the cloud without local Ollama:
1. In your Streamlit Cloud App settings, go to **Secrets**.
2. Add your free Google Gemini API key:
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```
*(If no API key is set, the application automatically runs in Autonomous Mode, generating DuckDB queries and Plotly charts deterministically).*

---

## 3. Option B: Hugging Face Spaces (Free)

Hugging Face Spaces offers free 2 CPU, 16 GB RAM instances.

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **"Create new Space"**.
2. Name your space (e.g., `agentic-data-analyst`).
3. Select **Space SDK**: **Streamlit** (or **Docker**).
4. Clone the space repository locally or push your files:
```bash
git remote add space https://huggingface.co/spaces/<your-username>/agentic-data-analyst
git push space main
```
5. The Space will build and provide a public HTTPS URL with live interactive charts and RAG capabilities.

---

## 4. Option C: Docker Containerization

Run the application anywhere Docker is supported (Windows, macOS, Linux, or Cloud VPS):

### Build Docker Image
```bash
docker build -t agentic-data-analyst .
```

### Run Container
```bash
docker run -d -p 8501:8501 --name agentic_analyst agentic-data-analyst
```
Open **`http://localhost:8501`** in your browser.

### Inspect Container Health
```bash
docker ps
docker logs -f agentic_analyst
```

---

## 5. Option D: Render / Railway Deployment

To deploy on Render:
1. Create a **New Web Service** from your GitHub repo.
2. Environment: **Python** (or **Docker**).
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Click **Deploy**.

---

## 6. Cloud LLM & Offline Autonomous Mode

| Environment | Primary LLM Provider | Fallback Engine | Cost |
|---|---|---|---|
| **Local Machine** | Ollama (`qwen2.5:3b` / `llama3.2:3b`) | Deterministic Rule-Based Engine | **$0.00** |
| **Streamlit Cloud** | Google Gemini API (optional key) | Deterministic Rule-Based Engine | **$0.00** |
| **Docker / VPS** | Local Ollama container or Gemini API | Deterministic Rule-Based Engine | **$0.00** |

> **Guarantee:** The application **never crashes** if an LLM is offline or unconfigured. The multi-agent workflow deterministically routes tasks through DuckDB and Plotly.

---

## 7. Troubleshooting & Verification

* **Issue: "Chroma requires sqlite3 >= 3.35.0" on Linux**
  * *Resolution:* Handled automatically via `pysqlite3-binary` in `requirements.txt` and the module patch in `src/memory/rag_store.py`.
* **Issue: Missing File Permissions on Read-Only Hosts**
  * *Resolution:* Handled automatically. The logger falls back to standard console output if file writes are restricted.
* **Issue: Port Binding in Cloud**
  * *Resolution:* Handled in `.streamlit/config.toml` and `Dockerfile` via `--server.address=0.0.0.0`.
