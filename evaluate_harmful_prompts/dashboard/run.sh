#!/bin/bash

# ============================================
# BanglaSafe PoC Dashboard - Setup & Run Script
# ============================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

echo "============================================"
echo "  BanglaSafe PoC Results Dashboard"
echo "============================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created at: $VENV_DIR"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r "$REQUIREMENTS_FILE" --quiet

echo ""
echo "All dependencies installed successfully!"
echo ""
echo "============================================"
echo "  Starting Streamlit Dashboard..."
echo "============================================"
echo ""
echo "The dashboard will open in your browser shortly."
echo "If it doesn't open automatically, go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server."
echo ""

# Run Streamlit
cd "$SCRIPT_DIR"
streamlit run app.py --server.headless=false --browser.gatherUsageStats=false
