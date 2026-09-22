# Copyright (c) 2026, DatologyAI. All rights reserved.

"""Opt-in Zephon dataloader support for GPT pretraining."""

from __future__ import annotations

import os
import pickle
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from zephon import Pipeline
from zephon.io import CacheOptions, Dataset, StoreOptions
from zephon.work import MixtureSpec, StaticMixtureWorkSource, TokenEstimation

from megatron.core import mpu
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.training import get_args, print_rank_0
from megatron.training.checkpointing import get_dataloader_checkpoint_name

TOKENS_FIELD = "input_ids"


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

    return ZephonDataConfig(sources=tuple(sources), **values)


def apply_zephon_runtime_overrides(config: ZephonDataConfig, args: Any) -> ZephonDataConfig:
    """Apply run-specific values without changing the reusable data recipe."""

    overrides = {
        name: value
        for name, value in {
            "canonical_replicas": args.zephon_canonical_replicas,
            "aggregate_dir": args.zephon_aggregate_dir,
            "run_id": args.zephon_run_id,
        }.items()
        if value is not None
    }
    return replace(config, **overrides)


def _build_zephon_runtime_options(
    config: ZephonDataConfig,
    *,
    data_parallel_rank: int,
    data_parallel_size: int,
) -> dict[str, Any]:
    """Build Zephon coordination options for the ranks that own distinct data streams."""

    options: dict[str, Any] = {
        "deterministic": True,
        "dp_degree": data_parallel_size,
        "dp_group_id": data_parallel_rank,
        "world_size": data_parallel_size,
        "global_rank": data_parallel_rank,
        "canonical_replicas": (
            config.canonical_replicas
            if config.canonical_replicas is not None
            else data_parallel_size
        ),
    }
    if config.cache_dir is not None:
        options["io_options"] = StoreOptions(
            cache=CacheOptions(enabled=True, root=config.cache_dir)
        )
    if config.aggregate_dir is not None:
        options["aggregate_dir"] = config.aggregate_dir
    if config.run_id is not None:
        options["run_id"] = config.run_id
    return options


def _zephon_tokenizer_id_from_args(args: Any) -> str:
    """Return the tokenizer identifier that Zephon should load independently."""

    if args.tokenizer_type != "HuggingFaceTokenizer":
        raise ValueError(
            "Zephon GPT pretraining currently requires --tokenizer-type HuggingFaceTokenizer"
        )
    if not isinstance(args.tokenizer_model, str) or not args.tokenizer_model:
        raise ValueError("Zephon GPT pretraining requires a non-empty --tokenizer-model")
    return args.tokenizer_model


class MegatronZephonDataLoader:
    """Iterator adapting a Zephon pipeline to Megatron's GPT batch schema."""

    def __init__(
        self,
        config: ZephonDataConfig,
        *,
        tokenizer_id: str,
        micro_batch_size: int,
        sequence_length: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        num_batches_per_train_step: int | None = None,
    ) -> None:
        canonical_replicas = (
            config.canonical_replicas
            if config.canonical_replicas is not None
            else data_parallel_size
        )
        if canonical_replicas <= 0 or canonical_replicas % data_parallel_size:
            raise ValueError(
                "Zephon canonical_replicas must be positive and divisible by the current "
                "data-parallel size so every rank owns the same number of lanes"
            )
        if (
            num_batches_per_train_step is not None
            and num_batches_per_train_step % canonical_replicas
        ):
            raise ValueError(
                "The number of batches consumed per training step must be a multiple of "
                "Zephon canonical_replicas so checkpoints land on complete lane-window "
                "boundaries"
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
            token_estimation=TokenEstimation(),
        )
        pipeline = Pipeline(work_source)
        if config.fetch_parallelism is not None:
            pipeline = pipeline.fetch_parallelism(config.fetch_parallelism)

        options = _build_zephon_runtime_options(
            config,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
        )

        self._pipeline = (
            pipeline.tokenize(
                tokenizer_id=tokenizer_id,
                field=config.text_field,
                add_attention_mask=False,
                max_length=sequence_length + 1,
                split_long_samples=True,
                special_tokens="bos_eos",
            )
            .pack_flat(max_length=sequence_length + 1, algorithm="wrap", emit_positions=True)
            .batch(micro_batch_size, drop_last=True)
            .options(**options)
        )
        self._pipeline.preflight_tokenizers()
        self._iterator = iter(self._pipeline)

    def __iter__(self) -> "MegatronZephonDataLoader":
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        sample_batch = next(self._iterator)
        training_batch = sample_batch.to_training(
            tokens_field=TOKENS_FIELD, return_labels=True, dtype=torch.long, ignore_index=-100
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
        try:
            restored_checkpoint = pickle.loads(checkpoint)
        except (pickle.PickleError, EOFError, AttributeError, ImportError, IndexError) as exc:
            raise ValueError("Unable to deserialize Zephon checkpoint state") from exc
        self._pipeline.restore(restored_checkpoint)
        self._iterator = iter(self._pipeline)


def _zephon_dataloader_checkpoint_path(args: Any) -> str:
    """Select the complete Zephon checkpoint owned by canonical DP rank 0."""

    return get_dataloader_checkpoint_name(args.dataloader_save, args.iteration, 0)


def _restore_dataloader_state(loader: MegatronZephonDataLoader, args: Any) -> None:
    if args.load is None:
        return
    if not args.dataloader_save:
        raise ValueError("Zephon resume with --load requires --dataloader-save")

    state_path = _zephon_dataloader_checkpoint_path(args)
    if not os.path.exists(state_path):
        raise FileNotFoundError(
            f"Zephon dataloader checkpoint for iteration {args.iteration} was not found at "
            f"{state_path}"
        )
    try:
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Zephon dataloader checkpoint at {state_path}") from exc
    if not isinstance(saved, Mapping):
        raise ValueError(
            f"Malformed Zephon dataloader checkpoint at {state_path}: expected a mapping"
        )
    saved_iteration = saved.get("iteration")
    if not isinstance(saved_iteration, int):
        raise ValueError(
            f"Malformed Zephon dataloader checkpoint at {state_path}: "
            "missing integer 'iteration'"
        )
    if saved_iteration != args.iteration:
        raise ValueError(
            f"Zephon dataloader checkpoint iteration mismatch at {state_path}: "
            f"expected {args.iteration}, found {saved_iteration}"
        )
    state_dict = saved.get("dataloader_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError(
            f"Malformed Zephon dataloader checkpoint at {state_path}: "
            "missing mapping 'dataloader_state_dict'"
        )
    try:
        loader.restore_state(state_dict)
    except (ValueError, TypeError, pickle.PickleError) as exc:
        raise ValueError(f"Malformed Zephon dataloader state at {state_path}: {exc}") from exc
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
    if args.rampup_batch_size is not None:
        raise ValueError("The initial Zephon integration does not support batch-size ramp-up")
    if mpu.get_tensor_model_parallel_rank() != 0:
        return None, None, None

    config = apply_zephon_runtime_overrides(load_zephon_data_config(args.zephon_data_config), args)
    data_parallel_size = mpu.get_data_parallel_world_size()
    loader = MegatronZephonDataLoader(
        config,
        tokenizer_id=_zephon_tokenizer_id_from_args(args),
        micro_batch_size=args.micro_batch_size,
        sequence_length=args.seq_length,
        data_parallel_rank=mpu.get_data_parallel_rank(),
        data_parallel_size=data_parallel_size,
        num_batches_per_train_step=get_num_microbatches() * data_parallel_size,
    )
    _restore_dataloader_state(loader, args)
    return loader, None, None


zephon_train_valid_test_datasets_provider.is_distributed = True
