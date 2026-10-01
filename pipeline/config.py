import os

DEFAULT_TRAIN_EVERY_N_EVENTS = 2000
DEFAULT_BATCH_SIZE = 50
DEFAULT_POLL_INTERVAL_SECONDS = 15.0


def _read_positive_int(env_name: str, default: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        value = default
    else:
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be a positive integer; got {raw_value!r}.") from exc

    if value <= 0:
        raise ValueError(f"{env_name} must be a positive integer; got {value!r}.")

    return value


def _read_positive_float(env_name: str, default: float) -> float:
    raw_value = os.getenv(env_name)
    if raw_value is None:
        value = default
    else:
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{env_name} must be a positive number; got {raw_value!r}.") from exc

    if value <= 0:
        raise ValueError(f"{env_name} must be a positive number; got {value!r}.")

    return value


def load_config() -> dict[str, float | int]:
    """Load pipeline configuration from environment with explicit defaults."""
    return {
        "train_every_n_events": _read_positive_int("TRAIN_EVERY_N_EVENTS", DEFAULT_TRAIN_EVERY_N_EVENTS),
        "batch_size": _read_positive_int("BATCH_SIZE", DEFAULT_BATCH_SIZE),
        "poll_interval_seconds": _read_positive_float("POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS),
    }
