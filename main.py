"""CLI launcher for the 3GPP RAG Assistant."""

import subprocess
import sys


def main():
    """Launch the Streamlit web application."""
    print("=" * 60)
    print("Launching 3GPP Telecom Spec Chatbot (Streamlit)...")
    print("=" * 60)
    # Start streamlit app via current python environment
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
