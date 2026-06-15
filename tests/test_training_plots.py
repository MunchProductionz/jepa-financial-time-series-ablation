import pandas as pd

from ablation_study_jepa.evaluation.training_plots import plot_training_history


def test_plot_training_history_writes_loss_and_gradient_svgs(tmp_path) -> None:
    history_path = tmp_path / "combined_epoch_history.csv"
    pd.DataFrame(
        {
            "window_label": ["window_000", "window_000", "window_000", "window_000"],
            "event": [
                "train_epoch_end",
                "validation_epoch_end",
                "train_epoch_end",
                "validation_epoch_end",
            ],
            "epoch": [0, 0, 1, 1],
            "train/total_loss": [0.5, 0.5, 0.3, 0.3],
            "val/prediction_loss": [None, 0.45, None, 0.35],
            "grad_norm_aux_block_0": [0.08, 0.08, 0.05, 0.05],
        }
    ).to_csv(history_path, index=False)

    outputs = plot_training_history(history_path)

    assert set(outputs) == {"gradients", "losses"}
    assert outputs["losses"].read_text(encoding="utf-8").startswith("<svg")
    assert "Training and Validation Loss" in outputs["losses"].read_text(encoding="utf-8")
    assert outputs["gradients"].read_text(encoding="utf-8").startswith("<svg")
