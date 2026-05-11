"""Integration smoke test for MLflow logging in the demo."""

import sys
from pathlib import Path
import pytest

# Add examples to path for importing the demo
sys.path.append(str(Path(__file__).parent.parent.parent / "examples"))

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("torch")

from mlflow.tracking import MlflowClient  # noqa: E402
from demo_vanishing_gradients import (  # noqa: E402
    create_failing_model,
    create_problematic_data,
    train_with_monitoring,
)


def test_demo_logs_metrics_and_artifacts_to_mlflow(tmp_path, monkeypatch):
    """Verify the demo writes a run, metrics, params, and artifacts to MLflow."""
    tracking_dir = tmp_path / "mlruns"
    tracking_uri = tracking_dir.as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.end_run()

    mlflow.set_tracking_uri(tracking_uri)
    model = create_failing_model()
    dataloader = create_problematic_data()

    train_with_monitoring(model, dataloader, num_steps=5)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("neuraldbg-vanishing-gradients")

    assert experiment is not None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=1,
        order_by=["attribute.start_time DESC"],
    )

    assert len(runs) == 1

    run = runs[0]
    assert run.data.params["lr"] == "0.0001"
    assert "total_events" in run.data.metrics

    loss_history = client.get_metric_history(run.info.run_id, "loss")
    assert len(loss_history) == 5

    artifacts = {
        artifact.path
        for artifact in client.list_artifacts(run.info.run_id, "artifacts")
    }
    assert "artifacts/causal_graph.mmd" in artifacts
    assert "artifacts/causal_hypotheses.json" in artifacts
    assert "artifacts/semantic_events.json" in artifacts
