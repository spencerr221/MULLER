#!/bin/bash
# MULLER Streamlit Demo Launcher

echo "🗄️  Starting MULLER Streamlit Demo..."
echo ""
echo "Prerequisites:"
echo "  - Python 3.11+"
echo "  - Dependencies installed: pip install -e .[demo]"
echo ""
echo "Opening demo at http://localhost:8501"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run streamlit from the script directory
cd "$SCRIPT_DIR"
streamlit run demo_streamlit.py

