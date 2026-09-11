#!/usr/bin/env bash
# SignalScope One-Click Unix Runner
set -e

if [ ! -f ".venv/bin/python" ]; then
    echo "[ERROR] Virtual environment .venv not found."
    echo "Please create the Python 3.11 virtual environment first."
    exit 1
fi

if [ -z "$1" ]; then
    echo "[USAGE] ./run_demo.sh path/to/image.jpg"
    exit 1
fi

.venv/bin/python model/predict.py --image "$1"
