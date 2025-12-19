#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Train the model if it doesn't exist
if [ ! -f "model/model.pkl" ]; then
    echo "Training model..."
    python train_model.py
fi

# Start the app with gunicorn
gunicorn --bind 0.0.0.0 --workers 1 --worker-class sync --timeout 60 app:app
