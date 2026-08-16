"""Entry point for the 3GPP Telecom Spec Assistant."""

import subprocess
import sys


def main():
    """Launch the Streamlit web application."""
    print("=" * 70)
    print("Starting 3GPP Telecom Spec Assistant Web UI (Streamlit)...")
    print("=" * 70)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
