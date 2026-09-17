# Copyright (c) 2026, DatologyAI. All rights reserved.

from __future__ import annotations

import json
import os
import pickle
import socket
import subprocess
import sys
from pathlib import Path

import pytest


def _run_phase(*, mode: str, num_processes: int, num_steps: int, output_dir: Path) -> None:
    torchrun = Path(sys.executable).with_name("torchrun")
    with socket.socket() as socket_file:
        socket_file.bind(("127.0.0.1", 0))
        master_port = socket_file.getsockname()[1]
    command = [
        str(torchrun),
        "--nnodes=1",
        "--master-addr=127.0.0.1",
        f"--master-port={master_port}",
        f"--nproc-per-node={num_processes}",
        "-m",
        "tests.unit_tests.data.zephon_elastic_worker",
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--num-steps",
        str(num_steps),
    ]
    subprocess.run(
        command,
        check=True,
        cwd=Path(__file__).resolve().parents[3],
        env=os.environ | {"GLOO_SOCKET_IFNAME": "lo"},
    )


def _load_steps(path: Path) -> list[list[dict[str, list]]]:
    with path.open("rb") as output_file:
        return pickle.load(output_file)


def _canonicalize_steps(steps: list[list[dict[str, list]]]) -> list[list[dict[str, list]]]:
    return [sorted(step, key=lambda batch: json.dumps(batch, sort_keys=True)) for step in steps]


def test_elastic_resume_preserves_global_step_contents(tmp_path: Path) -> None:
    pytest.importorskip("zephon")
    baseline_dir = tmp_path / "baseline"
    elastic_dir = tmp_path / "elastic"

    _run_phase(mode="baseline", num_processes=2, num_steps=4, output_dir=baseline_dir)
    _run_phase(mode="save", num_processes=2, num_steps=2, output_dir=elastic_dir)
    _run_phase(mode="resume", num_processes=1, num_steps=2, output_dir=elastic_dir)

    reference = _load_steps(baseline_dir / "baseline.pkl")
    resumed = _load_steps(elastic_dir / "save.pkl") + _load_steps(elastic_dir / "resume.pkl")
    assert _canonicalize_steps(resumed) == _canonicalize_steps(reference)
