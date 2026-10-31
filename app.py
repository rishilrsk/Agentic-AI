"""
Root Application Entrypoint Proxy for Deployment.
Allows deployment on Streamlit Community Cloud, Hugging Face Spaces, Render, and Docker
regardless of whether the host executes 'streamlit run app.py' or 'streamlit run src/app.py'.
"""

import sys
from pathlib import Path

# Ensure root workspace is available on Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import and invoke the primary application
from src.app import main

if __name__ == "__main__":
    main()
