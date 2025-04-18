#!/bin/bash

echo "📦 Creating virtual environment..."
python3 -m venv venv

source venv/bin/activate

echo "⬆️ Upgrading pip..."
pip install --upgrade pip

echo "📥 Installing requirements..."
pip install -r requirements.txt

# ✅ Add this to enable local inference module
export UNSTRUCTURED_LOCAL_INFERENCE_ENABLED=true

echo "🚀 Running document loader..."
python load_doc.py
