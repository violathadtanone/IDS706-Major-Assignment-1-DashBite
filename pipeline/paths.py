from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANAGED_DATA_DIRS = ("raw", "features", "models", "predictions", "quality")


def repo_root() -> Path:
    return REPO_ROOT


def resolve_data_dir(name: str) -> Path:
    if name not in MANAGED_DATA_DIRS:
        valid_names = ", ".join(MANAGED_DATA_DIRS)
        raise ValueError(f"Unsupported data directory name: {name!r}. Expected one of: {valid_names}.")
    return REPO_ROOT / "data" / name


def ensure_data_dirs() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in MANAGED_DATA_DIRS:
        path = resolve_data_dir(name)
        path.mkdir(parents=True, exist_ok=True)
        paths[name] = path
    return paths
