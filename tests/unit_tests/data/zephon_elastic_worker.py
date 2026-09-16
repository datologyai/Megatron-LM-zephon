# Copyright (c) 2026, DatologyAI. All rights reserved.

"""Worker process for the Zephon elastic-resume integration test."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace

import torch.distributed as dist

from megatron.training.datasets.zephon_dataloader import (
    MegatronZephonDataLoader,
    ZephonDataConfig,
    ZephonSource,
)


class TinyTokenizer:
    """Small deterministic Hugging Face-compatible tokenizer for tests."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, texts, **_kwargs):
        return {
            "input_ids": [[3 + (ord(character) % 29) for character in text] for text in texts],
            "attention_mask": [[1] * len(text) for text in texts],
        }


def _collect_global_steps(
    loader: MegatronZephonDataLoader, *, num_steps: int, canonical_replicas: int
) -> list[list[dict[str, list]]]:
    world_size = dist.get_world_size()
    batches_per_rank, remainder = divmod(canonical_replicas, world_size)
    if remainder:
        raise ValueError("canonical_replicas must divide the test world size")

    global_steps = []
    for _ in range(num_steps):
        local_batches = []
        for _ in range(batches_per_rank):
            batch = next(loader)
            local_batches.append(
                {"tokens": batch["tokens"].tolist(), "labels": batch["labels"].tolist()}
            )
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_batches)
        if dist.get_rank() == 0:
            global_steps.append(
                [batch for rank_batches in gathered for batch in rank_batches or []]
            )
    return global_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "save", "resume"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[2] / "assets/zephon_mixture"
    config = ZephonDataConfig(
        sources=(
            ZephonSource(name="prose", path=str(source_root / "prose")),
            ZephonSource(name="code", path=str(source_root / "code")),
        ),
        chunk_size=2,
        canonical_replicas=2,
        aggregate_dir=str(args.output_dir / "aggregate"),
        run_id="megatron-elastic-resume-test",
    )
    loader = MegatronZephonDataLoader(
        config,
        tokenizer=SimpleNamespace(tokenizer=TinyTokenizer()),
        micro_batch_size=1,
        sequence_length=16,
        data_parallel_rank=rank,
        data_parallel_size=world_size,
    )

    if args.mode == "resume":
        with (args.output_dir / "checkpoint.pkl").open("rb") as checkpoint_file:
            loader.restore_state(pickle.load(checkpoint_file))

    global_steps = _collect_global_steps(loader, num_steps=args.num_steps, canonical_replicas=2)

    if args.mode == "save":
        checkpoint = loader.save_state()
        if rank == 0:
            with (args.output_dir / "checkpoint.pkl").open("wb") as checkpoint_file:
                pickle.dump(checkpoint, checkpoint_file)
        dist.barrier()

    if rank == 0:
        with (args.output_dir / f"{args.mode}.pkl").open("wb") as output_file:
            pickle.dump(global_steps, output_file)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
