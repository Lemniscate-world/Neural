from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any
import sys
import os
import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("neuralpaper")

# Environment configuration
ENVIRONMENT = os.getenv("NEURAL_ENV", "development")
PRODUCTION = ENVIRONMENT == "production"

# Add project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Always use the mock connector for now to avoid parsing issues
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
try:
    from integrations.mock_connector import MockConnector
    neural_connector = MockConnector()
    logger.info("Using mock connector for Neural DSL")
except ImportError:
    # Fallback to a very basic mock
    logger.error("Failed to import mock connector, using basic mock")
    class BasicMockConnector:
        def __init__(self):
            pass
        def load_model(self, *args, **kwargs):
            return "# Mock model\nnetwork MockModel {\n  input: (1, 28, 28, 1)\n}", {}
        def parse_dsl(self, *args, **kwargs):
            # Return a more complete mock response
            return {
                "model_data": {
                    "name": "MockModel",
                    "input": {"shape": [1, 28, 28, 1]},
                    "layers": [{"type": "Dense", "params": {"units": 128}}]
                },
                "shape_history": [
                    {"layer_id": "layer_0", "layer_type": "Dense", "input_shape": [1, 28, 28, 1], "output_shape": [1, 128]}
                ],
                "trace_data": {}
            }
        def generate_code(self, *args, **kwargs):
            return "# Mock code\n"
        def start_debug_session(self, *args, **kwargs):
            return {"session_id": "mock", "dashboard_url": "#", "process_id": 0}
        def list_models(self, *args, **kwargs):
            return []
    neural_connector = BasicMockConnector()

# Create FastAPI app
app = FastAPI(
    title="NeuralPaper.ai API",
    description="API for NeuralPaper.ai platform",
    version="0.1.0"
)

# Configure CORS
allowed_origins = [
    "http://localhost:3000",
    "https://neuralpaper.vercel.app",
    "https://neuralpaper-frontend.onrender.com"
]

if not PRODUCTION:
    # Allow all origins in development
    allowed_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class DSLCode(BaseModel):
    code: str
    backend: str = "tensorflow"

class AnnotationRequest(BaseModel):
    model_id: str
    layer_id: str

class ModelData(BaseModel):
    id: str
    name: str
    description: str
    dsl_code: str
    annotations: Dict[str, str] = {}

# Neural connector is already initialized in the try/except block above

# In-memory storage for demo purposes
# In production, use a database
running_processes = {}

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {"status": "ok", "message": "Welcome to NeuralPaper.ai API", "version": "0.1.0"}

@app.get("/api")
async def api_root():
    """API root endpoint for health checks"""
    return {"status": "ok", "message": "NeuralPaper.ai API is running", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "environment": ENVIRONMENT}

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Handle all exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": str(type(exc).__name__)},
    )

@app.post("/parse")
@app.post("/api/parse")  # Add this route to handle frontend requests
async def parse_dsl(dsl: DSLCode):
    """Parse Neural DSL code and return model data"""
    try:
        logger.info(f"Received DSL parse request with backend: {dsl.backend}")
        logger.info(f"DSL code snippet: {dsl.code[:100]}...")  # Log first 100 chars of code
        result = neural_connector.parse_dsl(dsl.code, dsl.backend)
        logger.info("DSL parsing successful")
        return result
    except Exception as e:
        logger.error(f"Failed to parse DSL: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to parse DSL: {str(e)}")

@app.post("/generate")
@app.post("/api/generate")  # Add this route to handle frontend requests
async def generate_model_code(dsl: DSLCode):
    """Generate code from Neural DSL"""
    try:
        logger.info(f"Generating code for backend: {dsl.backend}")
        code = neural_connector.generate_code(dsl.code, dsl.backend)
        logger.info("Code generation successful")
        return {"code": code}
    except Exception as e:
        logger.error(f"Failed to generate code: {str(e)}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Failed to generate code: {str(e)}")

@app.post("/debug")
@app.post("/api/debug")  # Add this route to handle frontend requests
async def start_debug_session(dsl: DSLCode):
    """Start a NeuralDbg debug session"""
    try:
        logger.info(f"Starting debug session for backend: {dsl.backend}")
        result = neural_connector.start_debug_session(dsl.code, dsl.backend)

        # Store process ID
        session_id = result["session_id"]
        running_processes[session_id] = result["process_id"]

        logger.info(f"Debug session started with ID: {session_id}")

        return {
            "session_id": session_id,
            "dashboard_url": result["dashboard_url"],
            "message": "Debug session started. Open dashboard URL to view."
        }
    except Exception as e:
        logger.error(f"Failed to start debug session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start debug session: {str(e)}")

@app.get("/models")
async def get_models():
    """Get all available annotated models"""
    try:
        models = neural_connector.list_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """Get a specific model by ID"""
    try:
        dsl_code, annotations = neural_connector.load_model(model_id)
        return {
            "id": model_id,
            "name": annotations.get("name", model_id.capitalize()),
            "description": annotations.get("description", ""),
            "dsl_code": dsl_code,
            "annotations": annotations
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Add a mock trace endpoint for the debug panel
@app.get("/trace")
async def get_trace_data():
    """Get trace data for debugging"""
    # This is a mock implementation - in a real app, this would fetch actual trace data
    return [
        {
            "layer": "Conv2D_1",
            "input_shape": [1, 224, 224, 3],
            "output_shape": [1, 112, 112, 64],
            "flops": 118013952,
            "memory": 3211264,
            "execution_time": 0.015,
            "compute_time": 0.012,
            "transfer_time": 0.003,
            "grad_norm": 0.023,
            "dead_ratio": 0.05,
            "mean_activation": 0.45
        },
        {
            "layer": "MaxPooling2D_1",
            "input_shape": [1, 112, 112, 64],
            "output_shape": [1, 56, 56, 64],
            "flops": 802816,
            "memory": 802816,
            "execution_time": 0.003,
            "compute_time": 0.002,
            "transfer_time": 0.001,
            "grad_norm": 0.018,
            "dead_ratio": 0.0,
            "mean_activation": 0.38
        },
        {
            "layer": "ResidualBlock_1",
            "input_shape": [1, 56, 56, 64],
            "output_shape": [1, 56, 56, 64],
            "flops": 115605504,
            "memory": 802816,
            "execution_time": 0.012,
            "compute_time": 0.01,
            "transfer_time": 0.002,
            "grad_norm": 0.015,
            "dead_ratio": 0.02,
            "mean_activation": 0.42,
            "anomaly": False
        },
        {
            "layer": "ResidualBlock_2",
            "input_shape": [1, 56, 56, 64],
            "output_shape": [1, 56, 56, 64],
            "flops": 115605504,
            "memory": 802816,
            "execution_time": 0.012,
            "compute_time": 0.01,
            "transfer_time": 0.002,
            "grad_norm": 0.012,
            "dead_ratio": 0.03,
            "mean_activation": 0.4,
            "anomaly": False
        },
        {
            "layer": "ResidualBlock_3",
            "input_shape": [1, 56, 56, 64],
            "output_shape": [1, 28, 28, 128],
            "flops": 51380224,
            "memory": 401408,
            "execution_time": 0.008,
            "compute_time": 0.007,
            "transfer_time": 0.001,
            "grad_norm": 0.01,
            "dead_ratio": 0.01,
            "mean_activation": 0.35,
            "anomaly": False
        },
        {
            "layer": "ResidualBlock_4",
            "input_shape": [1, 28, 28, 128],
            "output_shape": [1, 28, 28, 128],
            "flops": 51380224,
            "memory": 401408,
            "execution_time": 0.008,
            "compute_time": 0.007,
            "transfer_time": 0.001,
            "grad_norm": 0.008,
            "dead_ratio": 0.04,
            "mean_activation": 0.32,
            "anomaly": True
        },
        {
            "layer": "GlobalAveragePooling2D",
            "input_shape": [1, 28, 28, 128],
            "output_shape": [1, 128],
            "flops": 100352,
            "memory": 512,
            "execution_time": 0.001,
            "compute_time": 0.001,
            "transfer_time": 0.0,
            "grad_norm": 0.005,
            "dead_ratio": 0.0,
            "mean_activation": 0.3,
            "anomaly": False
        },
        {
            "layer": "Dense",
            "input_shape": [1, 128],
            "output_shape": [1, 1000],
            "flops": 129000,
            "memory": 4000,
            "execution_time": 0.002,
            "compute_time": 0.002,
            "transfer_time": 0.0,
            "grad_norm": 0.003,
            "dead_ratio": 0.0,
            "mean_activation": 0.25,
            "anomaly": False
        }
    ]

# Debug session endpoints
@app.get("/api/debug/{session_id}/trace")
async def get_debug_trace(session_id: str):
    """Get trace data from a running debug session"""
    if session_id not in running_processes:
        raise HTTPException(status_code=404, detail="Debug session not found")

    # In a real implementation, this would fetch data from the running NeuralDbg instance
    # For now, return the mock data
    return await get_trace_data()

@app.get("/api/models/{model_id}/trace")
async def get_model_trace(model_id: str):
    """Get trace data for a specific model"""
    try:
        # Check if model exists
        _ = neural_connector.load_model(model_id)

        # In a real implementation, this would run the model through NeuralDbg
        # and return the actual trace data
        # For now, return the mock data
        return await get_trace_data()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get trace data: {str(e)}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws/debug/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        if session_id not in running_processes:
            await websocket.send_json({"error": "Debug session not found"})
            await websocket.close()
            return

        # In a real implementation, this would stream data from the running NeuralDbg instance
        # For now, send mock data periodically
        while True:
            trace_data = await get_trace_data()
            await websocket.send_json(trace_data)
            await asyncio.sleep(1)  # Send updates every second
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
