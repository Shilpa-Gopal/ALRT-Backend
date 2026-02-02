#!/bin/bash
export DATABASE_URL="postgresql://postgres:ipGawFDazQcvlvBVLKHSszJZnBoZMHoT@yamabiko.proxy.rlwy.net:51503/railway"
echo "DATABASE_URL set to: $DATABASE_URL"
source .venv/bin/activate
echo "Virtual environment activated"
echo "Starting Flask server..."
python backend/main.py
