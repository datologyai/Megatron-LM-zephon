# Copyright (c) 2026, DatologyAI. All rights reserved.

"""Opt-in Zephon dataloader support for GPT pretraining."""

from __future__ import annotations

import os
import pickle
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from megatron.core import mpu
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import get_checkpoint_name


@dataclass(frozen=True, slots=True)
class ZephonSource:
    """One named Zephon dataset in a weighted mixture."""

    name: str
    path: str
    fmt: str | None = None
    weight: float = 1.0


@dataclass(frozen=True, slots=True)
class ZephonDataConfig:
    """Validated values loaded from a Zephon data recipe."""

    sources: tuple[ZephonSource, ...]
    text_field: str = "text"
    cache_dir: str | None = None
    seed: int = 42
    chunk_size: int = 64
    canonical_replicas: int | None = None
    aggregate_dir: str | None = None
    run_id: str | None = None
    fetch_parallelism: int | None = None


_CONFIG_KEYS = {
    "sources",
    "text_field",
    "cache_dir",
    "seed",
    "chunk_size",
    "canonical_replicas",
    "aggregate_dir",
    "run_id",
    "fetch_parallelism",
}


def _resolve_recipe_path(value: str, recipe_dir: Path) -> str:
    if "://" in value or Path(value).is_absolute():
        return value
    return str((recipe_dir / value).resolve())


def load_zephon_data_config(path: str | os.PathLike[str]) -> ZephonDataConfig:
    """Load and validate a Zephon TOML data recipe."""

    recipe_path = Path(path).resolve()
    with recipe_path.open("rb") as recipe_file:
        values = tomllib.load(recipe_file)

    unknown_keys = set(values) - _CONFIG_KEYS
    if unknown_keys:
        raise ValueError("Unknown Zephon data recipe keys: " + ", ".join(sorted(unknown_keys)))

    raw_sources = values.pop("sources", None)
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Zephon data recipes must contain a non-empty 'sources' list")

    sources = []
    for source in raw_sources:
        if not isinstance(source, Mapping) or "name" not in source or "path" not in source:
            raise ValueError("Each Zephon source must contain 'name' and 'path' fields")
        sources.append(
            ZephonSource(
                name=str(source["name"]),
                path=_resolve_recipe_path(str(source["path"]), recipe_path.parent),
                fmt=source.get("fmt"),
                weight=float(source.get("weight", 1.0)),
            )
        )

    names = [source.name for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("Zephon source names must be unique")
    if any(not source.name or not source.path for source in sources):
        raise ValueError("Zephon source names and paths must not be empty")
    if any(source.weight <= 0 for source in sources):
        raise ValueError("Zephon source weights must all be positive")

    for key in ("cache_dir", "aggregate_dir"):
        value = values.get(key)
        if value is not None:
            if not value:
                raise ValueError(f"Zephon {key} must not be empty when set")
            values[key] = _resolve_recipe_path(str(value), recipe_path.parent)

    config = ZephonDataConfig(sources=tuple(sources), **values)
    if config.chunk_size < 1:
        raise ValueError("Zephon chunk_size must be positive")
    if config.fetch_parallelism is not None and config.fetch_parallelism < 1:
        raise ValueError("Zephon fetch_parallelism must be positive")
    return config


def _require_zephon() -> tuple[Any, Any, Any, Any]:
    try:
        from zephon import Pipeline
        from zephon.io import Dataset
        from zephon.work import MixtureSpec, StaticMixtureWorkSource
    except ImportError as exc:
        raise ImportError(
            "Zephon GPT pretraining requires the private Zephon package. "
            "See examples/zephon/README.md for installation instructions."
        ) from exc
    return Pipeline, Dataset, MixtureSpec, StaticMixtureWorkSource


def _unwrap_huggingface_tokenizer(tokenizer: Any) -> Any:
    """Return the raw Hugging Face tokenizer from current or legacy Megatron wrappers."""

    tokenizer_adapter = getattr(tokenizer, "_tokenizer", tokenizer)
    hf_tokenizer = getattr(tokenizer_adapter, "tokenizer", None)
    if hf_tokenizer is None:
        raise ValueError(
            "Zephon GPT pretraining currently requires --tokenizer-type " "HuggingFaceTokenizer"
        )
    return hf_tokenizer


class MegatronZephonDataLoader:
    """Iterator adapting a Zephon pipeline to Megatron's GPT batch schema."""

    def __init__(
        self,
        config: ZephonDataConfig,
        *,
        tokenizer: Any,
        micro_batch_size: int,
        sequence_length: int,
        data_parallel_rank: int,
        data_parallel_size: int,
    ) -> None:
        Pipeline, Dataset, MixtureSpec, StaticMixtureWorkSource = _require_zephon()
        hf_tokenizer = _unwrap_huggingface_tokenizer(tokenizer)

        canonical_replicas = config.canonical_replicas or data_parallel_size
        if canonical_replicas < data_parallel_size:
            raise ValueError(
                "Zephon canonical_replicas must be at least the current data-parallel size"
            )
        if data_parallel_size > 1 and (not config.aggregate_dir or not config.run_id):
            raise ValueError(
                "Distributed Zephon runs require aggregate_dir and run_id in the data recipe"
            )

        datasets = [
            Dataset.from_path(name=source.name, path=source.path, fmt=source.fmt)
            for source in config.sources
        ]
        work_source = StaticMixtureWorkSource(
            datasets=datasets,
            mixture=MixtureSpec({source.name: source.weight for source in config.sources}),
            chunk_size=config.chunk_size,
            seed=config.seed,
            exhausted_policy="repeat",
            shuffle_shards=True,
            shuffle_within_shard=True,
        )
        pipeline = Pipeline(work_source)
        if config.fetch_parallelism is not None:
            pipeline = pipeline.fetch_parallelism(config.fetch_parallelism)

        options: dict[str, Any] = {
            "deterministic": True,
            "dp_degree": data_parallel_size,
            "dp_group_id": data_parallel_rank,
            "canonical_replicas": canonical_replicas,
        }
        if torch.distributed.is_initialized():
            options["world_size"] = torch.distributed.get_world_size()
            options["global_rank"] = torch.distributed.get_rank()
        if config.cache_dir is not None:
            options["io_options"] = {"cache": {"enabled": True, "root": config.cache_dir}}
        if config.aggregate_dir is not None:
            options["aggregate_dir"] = config.aggregate_dir
        if config.run_id is not None:
            options["run_id"] = config.run_id

        self._pipeline = (
            pipeline.tokenize(
                tokenizer=hf_tokenizer,
                field=config.text_field,
                max_length=sequence_length + 1,
                split_long_samples=True,
                special_tokens="bos_eos",
            )
            .pack_flat(max_length=sequence_length + 1, algorithm="wrap", emit_positions=True)
            .batch(micro_batch_size, drop_last=True)
            .options(**options)
        )
        self._iterator = iter(self._pipeline)

    def __iter__(self) -> "MegatronZephonDataLoader":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        sample_batch = next(self._iterator)
        training_batch = sample_batch.to_training(
            tokens_field="input_ids", return_labels=True, dtype=torch.long, ignore_index=-100
        )
        labels = training_batch["labels"]
        return {
            "tokens": training_batch["input_ids"],
            "labels": labels,
            "loss_mask": (labels != -100).float(),
            "position_ids": training_batch["positions"],
        }

    def save_state(self) -> dict[str, bytes]:
        """Return an opaque Zephon checkpoint for Megatron checkpointing."""

        return {"zephon": pickle.dumps(self._pipeline.checkpoint())}

    def restore_state(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a state previously returned by :meth:`save_state`."""

        checkpoint = state_dict.get("zephon")
        if not isinstance(checkpoint, bytes):
            raise ValueError("Expected Zephon checkpoint state to contain bytes")
        self._pipeline.restore(pickle.loads(checkpoint))
        self._iterator = iter(self._pipeline)


def _restore_dataloader_state(loader: MegatronZephonDataLoader, args: Any) -> None:
    if args.load is None or args.dataloader_save is None:
        return
    state_path = get_checkpoint_name(
        args.dataloader_save,
        args.iteration,
        pipeline_rank=0,
        tensor_rank=0,
        basename=f"train_dataloader_dprank{mpu.get_data_parallel_rank():03d}.pt",
    )
    if not os.path.exists(state_path):
        print_rank_0(f"> Zephon dataloader state not found at {state_path}; starting fresh")
        return
    saved = torch.load(state_path, map_location="cpu", weights_only=False)
    loader.restore_state(saved["dataloader_state_dict"])
    print_rank_0(f"> restored Zephon dataloader state from {state_path}")


def zephon_train_valid_test_datasets_provider(
    _train_valid_test_num_samples: Sequence[int], vp_stage: int | None = None
) -> tuple[MegatronZephonDataLoader | None, None, None]:
    """Build the training-only Zephon loader expected by Megatron pretraining."""

    del vp_stage
    args = get_args()
    if args.eval_iters != 0:
        raise ValueError("The initial Zephon integration requires --eval-iters 0")
    if args.context_parallel_size != 1:
        raise ValueError("The initial Zephon integration requires --context-parallel-size 1")
    if mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None

    config = load_zephon_data_config(args.zephon_data_config)
    tokenizer = build_tokenizer(args)
    loader = MegatronZephonDataLoader(
        config,
        tokenizer=tokenizer,
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.seq_length,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=mpu.get_data_parallel_world_size(),
    )
    _restore_dataloader_state(loader, args)
    return loader, None, None


zephon_train_valid_test_datasets_provider.is_distributed = True
