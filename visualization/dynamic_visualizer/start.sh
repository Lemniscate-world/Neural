#!/bin/bash

# Start the Flask API server
python api.py &
API_PID=$!

# Start the static file server
python -m http.server 8000 &
SERVER_PID=$!

# Function to kill both servers on exit
cleanup() {
    kill $API_PID
    kill $SERVER_PID
}

trap cleanup EXIT

# Wait for user input
echo "Servers started. Press Ctrl+C to stop."
wait