#!/usr/bin/env bash
# SignalScope One-Click Unix Runner
set -e

if [ ! -f ".venv/bin/python" ]; then
    echo "[ERROR] Virtual environment .venv not found."
    echo "Please create the Python 3.11 virtual environment first."
    exit 1
fi

if [ -z "$1" ]; then
    echo "============================================================"
    echo " SignalScope - Generative Media Forensics"
    echo "============================================================"
    echo " Usage:"
    echo "   CLI Inference:    ./run_demo.sh path/to/image.jpg"
    echo "   Streamlit Web UI: ./run_demo.sh --ui"
    echo ""
    echo " No image specified. Launching Streamlit Web UI..."
    echo "============================================================"
    .venv/bin/python -m streamlit run src/ui/app.py
    exit 0
fi

if [ "$1" = "--ui" ]; then
    echo "Starting SignalScope Streamlit Web UI..."
    .venv/bin/python -m streamlit run src/ui/app.py
    exit 0
fi

.venv/bin/python src/model/predict.py --image "$1"
