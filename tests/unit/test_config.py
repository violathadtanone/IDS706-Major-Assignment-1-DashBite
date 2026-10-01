import pytest

from pipeline.config import load_config


@pytest.mark.unit
def test_load_config_defaults(monkeypatch):
    monkeypatch.delenv("TRAIN_EVERY_N_EVENTS", raising=False)
    monkeypatch.delenv("BATCH_SIZE", raising=False)
    monkeypatch.delenv("POLL_INTERVAL_SECONDS", raising=False)

    config = load_config()

    assert config["train_every_n_events"] == 2000
    assert config["batch_size"] == 50
    assert config["poll_interval_seconds"] == 15.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_overrides", "expected"),
    [
        ({"TRAIN_EVERY_N_EVENTS": "25"}, {"train_every_n_events": 25, "batch_size": 50, "poll_interval_seconds": 15.0}),
        ({"TRAIN_EVERY_N_EVENTS": "1500"}, {"train_every_n_events": 1500, "batch_size": 50, "poll_interval_seconds": 15.0}),
        ({"BATCH_SIZE": "5"}, {"train_every_n_events": 2000, "batch_size": 5, "poll_interval_seconds": 15.0}),
        ({"BATCH_SIZE": "75"}, {"train_every_n_events": 2000, "batch_size": 75, "poll_interval_seconds": 15.0}),
        ({"POLL_INTERVAL_SECONDS": "0.5"}, {"train_every_n_events": 2000, "batch_size": 50, "poll_interval_seconds": 0.5}),
        ({"POLL_INTERVAL_SECONDS": "4"}, {"train_every_n_events": 2000, "batch_size": 50, "poll_interval_seconds": 4.0}),
    ],
)
def test_load_config_environment_overrides(monkeypatch, env_overrides, expected):
    for env_name in ["TRAIN_EVERY_N_EVENTS", "BATCH_SIZE", "POLL_INTERVAL_SECONDS"]:
        monkeypatch.delenv(env_name, raising=False)
    for env_name, value in env_overrides.items():
        monkeypatch.setenv(env_name, value)

    config = load_config()

    assert config == expected


@pytest.mark.unit
@pytest.mark.parametrize("env_var", ["TRAIN_EVERY_N_EVENTS", "BATCH_SIZE", "POLL_INTERVAL_SECONDS"])
def test_load_config_rejects_non_positive_values(monkeypatch, env_var):
    monkeypatch.setenv(env_var, "0")
    with pytest.raises(ValueError, match="positive"):
        load_config()

    monkeypatch.setenv(env_var, "-5")
    with pytest.raises(ValueError, match="positive"):
        load_config()

    if env_var != "POLL_INTERVAL_SECONDS":
        monkeypatch.setenv(env_var, "abc")
        with pytest.raises(ValueError, match="positive"):
            load_config()
    else:
        monkeypatch.setenv(env_var, "abc")
        with pytest.raises(ValueError, match="positive"):
            load_config()


@pytest.mark.unit
def test_load_config_does_not_create_files_or_directories(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRAIN_EVERY_N_EVENTS", raising=False)
    monkeypatch.delenv("BATCH_SIZE", raising=False)
    monkeypatch.delenv("POLL_INTERVAL_SECONDS", raising=False)

    before = sorted(p.name for p in tmp_path.iterdir())
    load_config()
    after = sorted(p.name for p in tmp_path.iterdir())

    assert before == after
