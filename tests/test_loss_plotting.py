import pandas as pd

from flood_uncertainty.utils.loss_plotting import collect_loss_history, read_epoch_losses


def test_epoch_losses_join_sparse_rows_and_ignore_sampled_step_loss(tmp_path):
    path = tmp_path / "metrics.csv"
    path.write_text(
        "epoch,step,loss,train_loss_epoch,val_bce_loss,val_bce_land_water\n"
        "0,0,99,,,\n"
        "0,10,,,0.6,0.4\n"
        "0,10,,1.2,,\n"
        "1,11,88,,,\n"
        "1,20,,,0.5,0.3\n"
        "1,20,,1.0,,\n",
        encoding="utf-8",
    )

    history = read_epoch_losses(path)

    assert history["epoch"].tolist() == [0, 1]
    assert history["train_loss_epoch"].tolist() == [1.2, 1.0]
    assert history["val_bce_loss"].tolist() == [0.6, 0.5]
    assert history["val_bce_land_water"].tolist() == [0.4, 0.3]


def test_resume_uses_numeric_version_order_and_removes_stale_future_epochs(tmp_path):
    old = tmp_path / "logs/version_9"
    new = tmp_path / "logs/version_10"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    pd.DataFrame({
        "epoch": [0, 1, 2, 3, 4],
        "train_loss_epoch": [5.0, 4.0, 3.0, 2.0, 1.0],
        "val_bce_loss": [0.9, 0.8, 0.7, 0.6, 0.5],
    }).to_csv(old / "metrics.csv", index=False)
    pd.DataFrame({
        "epoch": [2, 3],
        "train_loss_epoch": [2.5, 2.0],
        "val_bce_loss": [0.65, 0.55],
    }).to_csv(new / "metrics.csv", index=False)

    history = collect_loss_history(tmp_path)

    assert history["epoch"].tolist() == [0, 1, 2, 3]
    assert history["train_loss_epoch"].tolist() == [5.0, 4.0, 2.5, 2.0]
    assert history["val_bce_loss"].tolist() == [0.9, 0.8, 0.65, 0.55]


def test_missing_validation_epochs_are_not_filled_with_old_values(tmp_path):
    old = tmp_path / "logs/version_0"
    new = tmp_path / "logs/version_1"
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (old / "metrics.csv").write_text(
        "epoch,train_loss_epoch,val_bce_loss\n0,2.0,0.8\n1,1.8,0.7\n",
        encoding="utf-8",
    )
    (new / "metrics.csv").write_text(
        "epoch,train_loss_epoch,val_bce_loss\n1,1.5,\n2,1.2,0.5\n",
        encoding="utf-8",
    )

    history = collect_loss_history(tmp_path)

    assert history["epoch"].tolist() == [0, 1, 2]
    assert pd.isna(history.loc[1, "val_bce_loss"])
    assert history.loc[2, "val_bce_loss"] == 0.5


def test_step_only_logs_and_validation_only_sessions_do_not_replace_history(tmp_path):
    train = tmp_path / "logs/version_0"
    partial = tmp_path / "logs/version_1"
    validation = tmp_path / "validation_logs/version_0"
    train.mkdir(parents=True)
    partial.mkdir(parents=True)
    validation.mkdir(parents=True)
    (train / "metrics.csv").write_text(
        "epoch,train_loss_epoch\n0,2.0\n1,1.0\n", encoding="utf-8"
    )
    (partial / "metrics.csv").write_text("epoch,loss\n0,999\n", encoding="utf-8")
    (validation / "metrics.csv").write_text(
        "epoch,val_bce_loss\n0,999\n", encoding="utf-8"
    )

    history = collect_loss_history(tmp_path)

    assert history["epoch"].tolist() == [0, 1]
    assert history["train_loss_epoch"].tolist() == [2.0, 1.0]
