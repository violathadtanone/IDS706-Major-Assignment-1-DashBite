import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGES = ("simulator", "preprocess", "train", "infer", "dashboard")
STOP_TIMEOUT_SECONDS = 8.0


def _stage_commands() -> dict[str, list[str]]:
    python = sys.executable
    return {
        "simulator": [python, "-m", "pipeline.simulator"],
        "preprocess": [python, "-m", "pipeline.preprocess"],
        "train": [python, "-m", "pipeline.train"],
        "infer": [python, "-m", "pipeline.infer"],
        "dashboard": [
            python,
            "-m",
            "streamlit",
            "run",
            "pipeline/dashboard.py",
            "--server.port",
            "8501",
        ],
    }


def _process_details(pid: int) -> tuple[str, str] | None:
    command_result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    start_result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        check=False,
        capture_output=True,
        text=True,
    )
    command = command_result.stdout.strip()
    start_time = start_result.stdout.strip()
    if command_result.returncode != 0 or start_result.returncode != 0 or not command or not start_time:
        return None
    return command, start_time


def _record_matches_process(record: dict[str, object]) -> bool:
    try:
        pid = int(record["pid"])
        process_group = os.getpgid(pid)
    except (KeyError, TypeError, ValueError, ProcessLookupError, PermissionError):
        return False

    if process_group != pid:
        return False
    details = _process_details(pid)
    return details == (record.get("process_command"), record.get("start_time"))


@contextmanager
def _manager_lock(log_dir: Path) -> Iterator[None]:
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / ".manager.lock").open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_pid_record(pid_path: Path, record: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=pid_path.parent,
        prefix=f".{pid_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(record, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, pid_path)


def _check_dashboard_port(port: int = 8501) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(
                f"dashboard port {port} is already in use; stop its owner before running the full stack"
            ) from exc


def _read_pid_record(pid_path: Path) -> dict[str, object] | None:
    try:
        data = json.loads(pid_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _signal_group(pid: int, signum: int) -> bool:
    try:
        os.killpg(pid, signum)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def _group_exists(pid: int) -> bool:
    result = subprocess.run(
        ["ps", "-axo", "pgid=,stat="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0].isdigit() and int(fields[0]) == pid:
                return not fields[1].startswith(("Z", "X"))
        return False

    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_group_exit(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _group_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    return not _group_exists(pid)


def _stop_record(record: dict[str, object]) -> bool:
    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        return True

    if not _record_matches_process(record):
        return not _group_exists(pid)

    _signal_group(pid, signal.SIGINT)
    if _wait_for_group_exit(pid, STOP_TIMEOUT_SECONDS):
        return True

    _signal_group(pid, signal.SIGTERM)
    if _wait_for_group_exit(pid, 3.0):
        return True

    _signal_group(pid, signal.SIGKILL)
    return _wait_for_group_exit(pid, 2.0)


def _launch_stage(
    stage: str,
    command: list[str],
    log_dir: Path,
    pid_dir: Path,
    env: dict[str, str],
) -> bool:
    pid_path = pid_dir / f"{stage}.pid"
    previous = _read_pid_record(pid_path) if pid_path.exists() else None
    if previous is not None and _record_matches_process(previous):
        print(f"{stage} is already running (pid {previous['pid']}).")
        return False
    if pid_path.exists():
        pid_path.unlink()

    log_path = log_dir / f"{stage}.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    details = _process_details(process.pid)
    if details is None:
        process.terminate()
        process.wait(timeout=2)
        raise RuntimeError(f"Could not verify the started {stage} process (pid {process.pid}).")

    process_command, start_time = details
    record: dict[str, object] = {
        "stage": stage,
        "pid": process.pid,
        "process_group": process.pid,
        "command": command,
        "process_command": process_command,
        "start_time": start_time,
    }
    _write_pid_record(pid_path, record)
    display_log_path = log_path.relative_to(PROJECT_ROOT) if log_path.is_relative_to(PROJECT_ROOT) else log_path
    print(f"Started {stage} (pid {process.pid}); log: {display_log_path}")
    return True


def start_stack(log_dir: Path | None = None, commands: dict[str, list[str]] | None = None) -> int:
    log_dir = log_dir or PROJECT_ROOT / ".logs"
    pid_dir = log_dir / "pids"
    commands = commands or _stage_commands()
    env = os.environ.copy()
    env.setdefault("POLL_INTERVAL_SECONDS", "15")
    env.setdefault("PYTHONUNBUFFERED", "1")
    started: list[str] = []

    with _manager_lock(log_dir):
        pid_dir.mkdir(parents=True, exist_ok=True)
        try:
            if "dashboard" in commands:
                _check_dashboard_port()
            for stage in STAGES:
                if stage not in commands:
                    continue
                if _launch_stage(stage, commands[stage], log_dir, pid_dir, env):
                    started.append(stage)
            if started:
                time.sleep(0.5)
                for stage in started:
                    record = _read_pid_record(pid_dir / f"{stage}.pid")
                    if record is None or not _record_matches_process(record):
                        raise RuntimeError(f"{stage} exited during startup; see {log_dir / f'{stage}.log'}")
        except Exception:
            print("Startup failed; stopping stages started by this command.", file=sys.stderr)
            for stage in reversed(started):
                pid_path = pid_dir / f"{stage}.pid"
                record = _read_pid_record(pid_path)
                if record is not None and _stop_record(record):
                    pid_path.unlink(missing_ok=True)
            raise
    return 0


def stop_stack(log_dir: Path | None = None) -> int:
    log_dir = log_dir or PROJECT_ROOT / ".logs"
    pid_dir = log_dir / "pids"
    failures: list[str] = []

    with _manager_lock(log_dir):
        for stage in reversed(STAGES):
            pid_path = pid_dir / f"{stage}.pid"
            if not pid_path.exists():
                continue
            record = _read_pid_record(pid_path)
            if record is None:
                pid_path.unlink(missing_ok=True)
                print(f"Removed invalid PID record for {stage}.")
                continue
            if _stop_record(record):
                pid_path.unlink(missing_ok=True)
                print(f"Stopped {stage} (pid {record.get('pid')}).")
            else:
                failures.append(stage)
                print(f"Could not stop {stage} (pid {record.get('pid')}); PID record retained.", file=sys.stderr)

    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage the detached DashBite application stack.")
    parser.add_argument("action", choices=("start", "stop"))
    args = parser.parse_args(argv)
    try:
        return start_stack() if args.action == "start" else stop_stack()
    except Exception as exc:
        print(f"Stack {args.action} failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
