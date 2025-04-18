#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt

# Run Streamlit app
exec streamlit run rag_v3.py >> streamlit.log 2>&1

