# Copyright (c) 2026, DatologyAI. All rights reserved.

"""Demonstrate exact Zephon data-stream resume across a DP topology change."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import socket
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECIPE = REPO_ROOT / "examples" / "zephon" / "elastic_local_jsonl.toml"


def _write_demo_tokenizer(path: Path) -> None:
    """Create small local Hugging Face assets for Zephon to load independently."""

    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        models.WordLevel({"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(path)


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as local_socket:
        local_socket.bind(("127.0.0.1", 0))
        return local_socket.getsockname()[1]


def _batch_state(batch: dict[str, Any]) -> dict[str, Any]:
    return {name: value.tolist() for name, value in batch.items()}


def _collect_global_steps(
    loader: Any, *, num_steps: int, canonical_replicas: int
) -> list[list[dict[str, Any]]]:
    import torch.distributed as dist

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    batches_per_rank, remainder = divmod(canonical_replicas, world_size)
    if remainder:
        raise ValueError(
            f"canonical_replicas={canonical_replicas} must be divisible by "
            f"world_size={world_size}"
        )

    global_steps = []
    for _ in range(num_steps):
        local_batches = [_batch_state(next(loader)) for _ in range(batches_per_rank)]
        gathered: list[list[dict[str, Any]] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_batches)
        if rank == 0:
            global_steps.append(
                [batch for rank_batches in gathered for batch in rank_batches or []]
            )
    return global_steps


def _worker(args: argparse.Namespace) -> None:
    import torch.distributed as dist

    from megatron.training.datasets.zephon_dataloader import (
        MegatronZephonDataLoader,
        apply_zephon_runtime_overrides,
        load_zephon_data_config,
    )

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recipe_config = load_zephon_data_config(args.recipe)
    canonical_replicas = recipe_config.canonical_replicas
    if canonical_replicas is None:
        raise ValueError("The elastic demo recipe must set canonical_replicas")
    config = apply_zephon_runtime_overrides(
        recipe_config,
        SimpleNamespace(
            zephon_canonical_replicas=None,
            zephon_aggregate_dir=str(args.output_dir / "aggregate"),
            zephon_run_id=args.run_id,
        ),
    )
    # Each data-only worker is already a short-lived torchrun subprocess. Keep
    # this correctness demo single-process within each worker; the GPU smoke
    # script exercises the process runner and automatic MTP default.
    config = replace(config, runner="inline", mtp_mode=False)
    loader = MegatronZephonDataLoader(
        config,
        tokenizer_id=str(args.tokenizer_id),
        bos_token_id=1,
        eos_token_id=2,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.sequence_length,
        data_parallel_rank=rank,
        data_parallel_size=world_size,
        num_batches_per_train_step=canonical_replicas,
    )

    if args.phase == "resume":
        with (args.output_dir / "checkpoint.pkl").open("rb") as checkpoint_file:
            loader.restore_state(pickle.load(checkpoint_file))

    global_steps = _collect_global_steps(
        loader, num_steps=args.num_steps, canonical_replicas=canonical_replicas
    )

    if args.phase == "save":
        checkpoint = loader.save_state()
        if rank == 0:
            with (args.output_dir / "checkpoint.pkl").open("wb") as checkpoint_file:
                pickle.dump(checkpoint, checkpoint_file)
        dist.barrier()

    if rank == 0:
        with (args.output_dir / f"{args.phase}.json").open("w", encoding="utf-8") as output_file:
            json.dump(global_steps, output_file)
    dist.barrier()
    dist.destroy_process_group()


def _run_phase(
    *,
    phase: str,
    num_workers: int,
    num_steps: int,
    output_dir: Path,
    recipe: Path,
    run_id: str,
    sequence_length: int,
    micro_batch_size: int,
    tokenizer_id: Path,
) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={num_workers}",
        "--master-addr=127.0.0.1",
        f"--master-port={_find_free_local_port()}",
        str(Path(__file__).resolve()),
        "--worker",
        "--phase",
        phase,
        "--num-steps",
        str(num_steps),
        "--output-dir",
        str(output_dir),
        "--recipe",
        str(recipe),
        "--run-id",
        run_id,
        "--sequence-length",
        str(sequence_length),
        "--micro-batch-size",
        str(micro_batch_size),
        "--tokenizer-id",
        str(tokenizer_id),
    ]
    env = os.environ | {"GLOO_SOCKET_IFNAME": "lo0" if sys.platform == "darwin" else "lo"}
    result = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False
    )
    if result.returncode:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        result.check_returncode()


def _load_steps(path: Path) -> list[list[dict[str, Any]]]:
    with path.open(encoding="utf-8") as input_file:
        return json.load(input_file)


def _canonicalize_steps(steps: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    return [
        sorted(step, key=lambda batch: json.dumps(batch, separators=(",", ":"), sort_keys=True))
        for step in steps
    ]


def _fingerprints(steps: list[list[dict[str, Any]]]) -> list[str]:
    return [
        hashlib.sha256(
            json.dumps(step, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()[:8]
        for step in steps
    ]


def _format_fingerprints(fingerprints: list[str]) -> str:
    return "  ".join(fingerprints)


def _run_demo(args: argparse.Namespace, work_dir: Path) -> bool:
    if args.checkpoint_after <= 0 or args.checkpoint_after >= args.total_steps:
        raise ValueError("checkpoint-after must be between 1 and total-steps - 1")

    with args.recipe.open("rb") as recipe_file:
        recipe_values = tomllib.load(recipe_file)
    canonical_replicas = recipe_values.get("canonical_replicas")
    if not isinstance(canonical_replicas, int) or canonical_replicas <= 0:
        raise ValueError("The elastic demo requires positive canonical_replicas")
    for num_workers in (args.initial_workers, args.resume_workers):
        if num_workers <= 0:
            raise ValueError("initial-workers and resume-workers must be positive")
        if canonical_replicas % num_workers:
            raise ValueError(
                f"canonical_replicas={canonical_replicas} must be divisible by "
                f"num_workers={num_workers}"
            )

    source_weights = ", ".join(
        f"{source['name']}={source.get('weight', 1.0):g}" for source in recipe_values["sources"]
    )
    reference_dir = work_dir / "reference"
    elastic_dir = work_dir / "elastic"
    tokenizer_id = work_dir / "tokenizer"
    _write_demo_tokenizer(tokenizer_id)
    shared = {
        "recipe": args.recipe,
        "sequence_length": args.sequence_length,
        "micro_batch_size": args.micro_batch_size,
        "tokenizer_id": tokenizer_id,
    }
    _run_phase(
        phase="reference",
        num_workers=args.initial_workers,
        num_steps=args.total_steps,
        output_dir=reference_dir,
        run_id="zephon-elastic-reference",
        **shared,
    )
    _run_phase(
        phase="save",
        num_workers=args.initial_workers,
        num_steps=args.checkpoint_after,
        output_dir=elastic_dir,
        run_id="zephon-elastic-resume",
        **shared,
    )
    _run_phase(
        phase="resume",
        num_workers=args.resume_workers,
        num_steps=args.total_steps - args.checkpoint_after,
        output_dir=elastic_dir,
        run_id="zephon-elastic-resume",
        **shared,
    )

    reference = _load_steps(reference_dir / "reference.json")
    before_checkpoint = _load_steps(elastic_dir / "save.json")
    after_resume = _load_steps(elastic_dir / "resume.json")
    actual = before_checkpoint + after_resume
    canonical_reference = _canonicalize_steps(reference)
    canonical_actual = _canonicalize_steps(actual)
    matches = canonical_actual == canonical_reference

    reference_hashes = _fingerprints(canonical_reference)
    before_hashes = _fingerprints(_canonicalize_steps(before_checkpoint))
    after_hashes = _fingerprints(_canonicalize_steps(after_resume))
    split = args.checkpoint_after
    reference_before = _format_fingerprints(reference_hashes[:split])
    reference_after = _format_fingerprints(reference_hashes[split:])
    before = _format_fingerprints(before_hashes)
    after = _format_fingerprints(after_hashes)
    print("Zephon deterministic elastic resume")
    print(f"Mixture weights:       {source_weights}")
    print(f"Data-parallel workers: {args.initial_workers} -> {args.resume_workers}")
    print(f"{'Reference:':<23}{reference_before} | {reference_after}")
    print(f"{f'{args.initial_workers}-worker stream:':<23}{before} | checkpoint")
    print(f"{f'{args.resume_workers}-worker resume:':<23}{' ' * len(before)} | {after}")
    print(f"Exact global-step match: {'YES' if matches else 'NO'}")
    return matches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove that Zephon global steps checkpointed under one DP topology "
            "resume exactly under another."
        )
    )
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--total-steps", type=int, default=4)
    parser.add_argument("--checkpoint-after", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--initial-workers", type=int, default=2)
    parser.add_argument("--resume-workers", type=int, default=1)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=["reference", "save", "resume"], help=argparse.SUPPRESS)
    parser.add_argument("--num-steps", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--tokenizer-id", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.recipe = args.recipe.resolve()
    if args.worker:
        _worker(args)
        return

    if args.work_dir is not None:
        args.work_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=args.work_dir))
        matches = _run_demo(args, run_dir)
        print(f"Artifacts:             {run_dir}")
    else:
        with tempfile.TemporaryDirectory(prefix="megatron-zephon-elastic-") as temp_dir:
            matches = _run_demo(args, Path(temp_dir))
    if not matches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
