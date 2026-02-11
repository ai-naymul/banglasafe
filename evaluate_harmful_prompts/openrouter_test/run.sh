#!/bin/bash

# ============================================
# BanglaSafe - OpenRouter Testing Script
# ============================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"
ENV_FILE="$SCRIPT_DIR/.env"

echo "============================================"
echo "  BanglaSafe - OpenRouter Safety Testing"
echo "============================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Check for .env file
if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "Warning: .env file not found!"
    echo "Creating template .env file..."
    echo "OPENROUTER_API_KEY=your_api_key_here" > "$ENV_FILE"
    echo ""
    echo "Please edit $ENV_FILE and add your OpenRouter API key."
    echo "Get your API key from: https://openrouter.ai/keys"
    echo ""
    read -p "Press Enter after adding your API key, or Ctrl+C to exit..."
fi

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
echo "  Starting OpenRouter Tests..."
echo "============================================"
echo ""

# Run the test script
cd "$SCRIPT_DIR"
python openrouter_test.py

echo ""
echo "============================================"
echo "  Tests Complete!"
echo "============================================"
echo ""
echo "To view results in the dashboard:"
echo "  cd ../dashboard && ./run.sh"
