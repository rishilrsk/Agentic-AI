"""
Test suite verifying Phase 2 environment setup and configuration loading.
"""

from pathlib import Path
from config import settings


def test_configuration_loading():
    """Verify that settings load with correct defaults and valid paths."""
    assert settings.OLLAMA_MODEL == "qwen2.5:3b"
    assert settings.MAX_RETRY_COUNT == 3
    assert settings.MAX_QUERY_EXEC_SECONDS == 10


def test_directory_creation():
    """Verify that helper methods correctly resolve and create required directories."""
    uploads_dir = settings.get_uploads_path()
    samples_dir = settings.get_samples_path()
    chroma_dir = settings.get_chroma_path()
    sqlite_path = settings.get_sqlite_path()

    assert uploads_dir.exists() and uploads_dir.is_dir()
    assert samples_dir.exists() and samples_dir.is_dir()
    assert chroma_dir.exists() and chroma_dir.is_dir()
    assert sqlite_path.parent.exists() and sqlite_path.parent.is_dir()


def test_project_root_detection():
    """Verify that PROJECT_ROOT is pointing to the current workspace."""
    assert settings.PROJECT_ROOT.exists()
    assert (settings.PROJECT_ROOT / "requirements.txt").exists()


def test_core_dependencies():
    """Verify that all heavy analytical & agent libraries import without DLL/C++ issues."""
    import duckdb
    import pandas
    import plotly
    import langgraph
    import langchain
    import chromadb
    import sqlparse
    import streamlit

    assert duckdb.__version__ is not None
    assert pandas.__version__ is not None
    assert plotly.__version__ is not None
    from langgraph.graph import StateGraph
    assert StateGraph is not None
    assert chromadb.__version__ is not None


