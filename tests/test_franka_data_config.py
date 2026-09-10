from pathlib import Path

import pytest
import yaml

from experiments.franka.make_data_config import main


def test_generated_config_uses_selected_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "chosen_dataset"
    for path in (
        dataset / "meta" / "info.json",
        dataset / "points_d455" / "data.mdb",
        dataset / "robot_state_action_stats" / "rot6d_points_d455.json",
        dataset / "franka_rgbd_manifest.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    output = tmp_path / "data.yaml"
    monkeypatch.setattr("sys.argv", ["make_data_config.py", "--dataset", str(dataset), "--output", str(output)])

    main()

    config = yaml.safe_load(output.read_text())
    entry = config["lerobot_datasets"][0]
    assert entry["root"] == str(dataset.parent)
    assert Path(entry["root"]) / entry["repo_id"] == dataset
    assert entry["repo_id"] == "chosen_dataset"
    assert entry["select_state_keys"] == []
    assert entry["state_action_norm_file"].startswith(str(dataset))


def test_generated_config_rejects_unprocessed_dataset(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["make_data_config.py", "--dataset", str(tmp_path),
                                    "--output", str(tmp_path / "data.yaml")])
    with pytest.raises(FileNotFoundError, match="processed dataset is incomplete"):
        main()
