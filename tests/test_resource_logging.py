import json

from ablation_study_jepa.utils.resource_logging import ResourceMonitor, collect_resource_snapshot


def test_resource_snapshot_does_not_require_gpu(tmp_path) -> None:
    snapshot = collect_resource_snapshot(tmp_path, metadata={"trial_id": "fake"})

    assert snapshot["metadata"]["trial_id"] == "fake"
    assert "cpu" in snapshot
    assert "memory" in snapshot
    assert "gpu" in snapshot
    assert "torch_cuda" in snapshot


def test_resource_monitor_writes_jsonl(tmp_path) -> None:
    output = tmp_path / "resource_usage.jsonl"
    monitor = ResourceMonitor(
        output_path=output,
        output_dir=tmp_path,
        interval_seconds=60,
        metadata={"experiment_name": "test"},
    )

    monitor.start()
    monitor.set_context(trial_id="trial_0000")
    monitor.sample_once(reason="test")
    monitor.stop()

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) >= 3
    assert records[0]["metadata"]["experiment_name"] == "test"
    assert any(record["metadata"].get("trial_id") == "trial_0000" for record in records)
