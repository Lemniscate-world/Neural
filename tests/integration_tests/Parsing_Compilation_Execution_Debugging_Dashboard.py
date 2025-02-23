import pytest
import subprocess
import requests
from flask_socketio import SocketIOTestClient
from neural.parser import parse_neural
from neural.dashboard.dashboard import app

@pytest.fixture(scope="module")
def start_dashboard():
    """Starts NeuralDbg's dashboard for testing."""
    process = subprocess.Popen(["python", "dashboard.py"])
    yield process
    process.terminate()  # Stop dashboard after tests

def test_full_integration(start_dashboard):
    """Runs end-to-end test from parsing to visualization."""
    # 1️⃣ Parse `.neural` file
    model_config = parse_neural("tests/sample_model.neural")
    assert model_config["network"] == "MyModel"

    # 2️⃣ Compile model
    result = subprocess.run(["python", "neural.py", "compile", "tests/sample_model.neural", "--backend", "tensorflow"])
    assert result.returncode == 0

    # 3️⃣ Run model execution
    result = subprocess.run(["python", "neural.py", "run", "MyModel"])
    assert result.returncode == 0

    # 4️⃣ Test NeuralDbg WebSocket
    socket_client = SocketIOTestClient(app)
    socket_client.emit("request_trace_update")
    received = socket_client.get_received()
    assert len(received) > 0  # Ensure execution trace updates are received

    # 5️⃣ Validate Dashboard Visualization API
    response = requests.get("http://localhost:8050/trace_graph")
    assert response.status_code == 200
