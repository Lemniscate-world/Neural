"""Integration smoke test for MLflow logging in the demo."""

import pytest

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("torch")

from mlflow.tracking import MlflowClient

from demo_vanishing_gradients import (
    create_failing_model,
    create_problematic_data,
    train_with_monitoring,
)


def test_demo_logs_metrics_and_artifacts_to_mlflow(tmp_path, monkeypatch):
    """Verify the demo writes a run, metrics, params, and artifacts to MLflow."""
    tracking_dir = tmp_path / "mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tracking_dir))
    mlflow.end_run()

    mlflow.set_tracking_uri(str(tracking_dir))
    model = create_failing_model()
    dataloader = create_problematic_data()

    train_with_monitoring(model, dataloader, num_steps=5)

    client = MlflowClient(tracking_uri=str(tracking_dir))
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

    artifacts = {artifact.path for artifact in client.list_artifacts(run.info.run_id, "artifacts")}
    assert "artifacts/causal_graph.mmd" in artifacts
    assert "artifacts/causal_hypotheses.json" in artifacts
    assert "artifacts/semantic_events.json" in artifacts
