import json
import os
import socket
import sys

import pytest

from pipeline.process_manager import _check_dashboard_port, start_stack, stop_stack


@pytest.mark.integration
def test_stack_manager_detaches_logs_and_stops_process_group(tmp_path):
    log_dir = tmp_path / ".logs"
    command = [
        sys.executable,
        "-c",
        "import time; print('probe ready', flush=True); time.sleep(60)",
    ]

    assert start_stack(log_dir=log_dir, commands={"simulator": command}) == 0

    pid_path = log_dir / "pids" / "simulator.pid"
    record = json.loads(pid_path.read_text(encoding="utf-8"))
    assert record["pid"] == record["process_group"]
    assert record["command"] == command
    assert (log_dir / "simulator.log").exists()
    os.kill(record["pid"], 0)

    assert stop_stack(log_dir=log_dir) == 0
    assert not pid_path.exists()
    with pytest.raises(ProcessLookupError):
        os.killpg(record["pid"], 0)


@pytest.mark.integration
def test_startup_failure_rolls_back_started_processes(tmp_path):
    log_dir = tmp_path / ".logs"
    command = [sys.executable, "-c", "raise SystemExit(7)"]

    with pytest.raises(RuntimeError, match="startup|exited"):
        start_stack(log_dir=log_dir, commands={"simulator": command})

    assert not list((log_dir / "pids").glob("*.pid"))


@pytest.mark.integration
def test_dashboard_port_preflight_rejects_an_occupied_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        occupied_port = listener.getsockname()[1]

        with pytest.raises(RuntimeError, match=f"port {occupied_port} is already in use"):
            _check_dashboard_port(occupied_port)
