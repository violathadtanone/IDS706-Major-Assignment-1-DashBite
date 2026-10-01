import argparse
import csv
import os
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import load_config
from pipeline.paths import resolve_data_dir

CSV_COLUMNS = [
    "order_id",
    "timestamp",
    "distance_km",
    "prep_minutes",
    "order_value",
    "was_late",
]


def generate_batch(batch_size: int, *, seed: int | None = None) -> list[dict[str, object]]:
    """Generate a synthetic batch of food-delivery orders."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}.")

    rng = random.Random(seed)
    id_rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for _ in range(batch_size):
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        rows.append(
            {
                "order_id": f"order-{id_rng.getrandbits(128):032x}",
                "timestamp": timestamp,
                "distance_km": round(rng.uniform(0.5, 30.0), 2),
                "prep_minutes": rng.randint(10, 90),
                "order_value": round(rng.uniform(5.0, 120.0), 2),
                "was_late": rng.choice([True, False]),
            }
        )
    return rows


def write_batch_csv(rows: list[dict[str, object]], output_dir: Path) -> Path:
    """Write one batch atomically to a CSV file in the chosen raw-data directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    final_path = output_dir / f"orders_{timestamp}.csv"
    temp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            dir=output_dir,
            prefix=".orders_",
            suffix=".csv.tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, final_path)
        return final_path
    except BaseException:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic raw orders for DashBite.")
    parser.add_argument("--once", action="store_true", help="run a single batch and exit")
    parser.add_argument("--output-dir", type=Path, help="override the raw output directory")
    parser.add_argument("--batch-size", type=int, help="override the configured batch size")
    parser.add_argument("--poll-interval", type=float, help="override the configured poll interval in seconds")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config()
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    poll_interval_seconds = (
        args.poll_interval if args.poll_interval is not None else config["poll_interval_seconds"]
    )
    output_dir = args.output_dir if args.output_dir is not None else resolve_data_dir("raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            rows = generate_batch(batch_size)
            final_path = write_batch_csv(rows, output_dir)
            print(f"New orders arrived: batch_size={batch_size} file={final_path.name}")
            if args.once:
                return 0
            time.sleep(poll_interval_seconds)
    except KeyboardInterrupt:
        print("Simulator stopped cleanly.")
        return 0
    except Exception as exc:  # pragma: no cover - exercised by integration if needed
        print(f"Simulator error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
