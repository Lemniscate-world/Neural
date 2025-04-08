#!/bin/bash

# Start the backend server
echo "Starting backend server..."
cd backend
python -m uvicorn api.main:app --reload --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start the frontend server
echo "Starting frontend server..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

# Function to kill both servers on exit
cleanup() {
    echo "Shutting down servers..."
    kill $BACKEND_PID
    kill $FRONTEND_PID
}

trap cleanup EXIT

# Wait for user input
echo "NeuralPaper.ai is running!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop."
wait
