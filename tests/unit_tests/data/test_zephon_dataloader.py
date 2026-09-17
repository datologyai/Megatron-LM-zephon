# Copyright (c) 2026, DatologyAI. All rights reserved.

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import megatron.training.datasets.zephon_dataloader as zephon_dataloader
from megatron.training.datasets.zephon_dataloader import (
    MegatronZephonDataLoader,
    _unwrap_huggingface_tokenizer,
    apply_zephon_runtime_overrides,
    load_zephon_data_config,
)


def test_zephon_loader_enables_token_estimation(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_work_source_options = {}
    token_estimation = object()

    class FakeDataset:
        @staticmethod
        def from_path(**kwargs):
            return kwargs

    class FakeMixtureSpec:
        def __init__(self, weights):
            self.weights = weights

    class FakeStaticMixtureWorkSource:
        def __init__(self, **kwargs):
            captured_work_source_options.update(kwargs)

    class FakeTokenEstimation:
        def __new__(cls):
            return token_estimation

    class FakePipeline:
        def __init__(self, _work_source):
            pass

        def tokenize(self, **_kwargs):
            return self

        def pack_flat(self, **_kwargs):
            return self

        def batch(self, *_args, **_kwargs):
            return self

        def options(self, **_kwargs):
            return self

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(
        zephon_dataloader,
        "_require_zephon",
        lambda: (
            FakePipeline,
            FakeDataset,
            FakeMixtureSpec,
            FakeStaticMixtureWorkSource,
            FakeTokenEstimation,
        ),
    )
    tokenizer = SimpleNamespace(_tokenizer=SimpleNamespace(tokenizer=object()))
    config = zephon_dataloader.ZephonDataConfig(
        sources=(
            zephon_dataloader.ZephonSource(name="prose", path="prose", weight=3.0),
            zephon_dataloader.ZephonSource(name="code", path="code", weight=1.0),
        )
    )

    MegatronZephonDataLoader(
        config,
        tokenizer=tokenizer,
        micro_batch_size=1,
        sequence_length=16,
        data_parallel_rank=0,
        data_parallel_size=1,
    )

    assert captured_work_source_options["token_estimation"] is token_estimation


def test_load_zephon_data_config_resolves_weighted_sources() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = load_zephon_data_config(repo_root / "examples/zephon/local_jsonl.toml")

    assert [(source.name, source.fmt, source.weight) for source in config.sources] == [
        ("prose", "jsonl", 3.0),
        ("code", "jsonl", 1.0),
    ]
    assert config.sources[0].path == str(repo_root / "tests/assets/zephon_mixture/prose")
    assert config.chunk_size == 2


def test_runtime_values_override_reusable_recipe() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = load_zephon_data_config(repo_root / "examples/zephon/elastic_local_jsonl.toml")
    args = SimpleNamespace(
        zephon_canonical_replicas=4,
        zephon_aggregate_dir="/shared/aggregate",
        zephon_run_id="example-run",
    )

    updated = apply_zephon_runtime_overrides(config, args)

    assert config.canonical_replicas == 2
    assert config.aggregate_dir is None
    assert config.run_id is None
    assert updated.canonical_replicas == 4
    assert updated.aggregate_dir == "/shared/aggregate"
    assert updated.run_id == "example-run"


def test_load_zephon_data_config_rejects_duplicate_sources(tmp_path: Path) -> None:
    recipe = tmp_path / "duplicate.toml"
    recipe.write_text("""
[[sources]]
name = "same"
path = "one"

[[sources]]
name = "same"
path = "two"
""")

    with pytest.raises(ValueError, match="names must be unique"):
        load_zephon_data_config(recipe)


def test_load_zephon_data_config_rejects_unknown_keys(tmp_path: Path) -> None:
    recipe = tmp_path / "unknown.toml"
    recipe.write_text("""
unexpected = true

[[sources]]
name = "data"
path = "data"
""")

    with pytest.raises(ValueError, match="Unknown Zephon data recipe keys"):
        load_zephon_data_config(recipe)


def test_zephon_loader_maps_batches_to_megatron_schema() -> None:
    class FakeBatch:
        def to_training(self, **kwargs):
            assert kwargs["return_labels"] is True
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[2, 3, -100]]),
                "positions": torch.tensor([[0, 1, 2]]),
            }

    loader = MegatronZephonDataLoader.__new__(MegatronZephonDataLoader)
    loader._iterator = iter([FakeBatch()])

    batch = next(loader)

    assert set(batch) == {"tokens", "labels", "loss_mask", "position_ids"}
    assert torch.equal(batch["tokens"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(batch["labels"], torch.tensor([[2, 3, -100]]))
    assert torch.equal(batch["loss_mask"], torch.tensor([[1.0, 1.0, 0.0]]))
    assert torch.equal(batch["position_ids"], torch.tensor([[0, 1, 2]]))


def test_unwrap_huggingface_tokenizer_supports_current_megatron_wrapper() -> None:
    raw_tokenizer = object()
    tokenizer = SimpleNamespace(_tokenizer=SimpleNamespace(tokenizer=raw_tokenizer))

    assert _unwrap_huggingface_tokenizer(tokenizer) is raw_tokenizer


def test_unwrap_huggingface_tokenizer_supports_legacy_wrapper() -> None:
    raw_tokenizer = object()
    tokenizer = SimpleNamespace(tokenizer=raw_tokenizer)

    assert _unwrap_huggingface_tokenizer(tokenizer) is raw_tokenizer


def test_zephon_loader_round_trips_opaque_checkpoint() -> None:
    class FakePipeline:
        def __init__(self) -> None:
            self.restored = None

        def checkpoint(self):
            return {"cursor": 17}

        def restore(self, checkpoint):
            self.restored = checkpoint

        def __iter__(self):
            return iter(())

    source = MegatronZephonDataLoader.__new__(MegatronZephonDataLoader)
    source._pipeline = FakePipeline()
    state = source.save_state()

    restored = MegatronZephonDataLoader.__new__(MegatronZephonDataLoader)
    restored._pipeline = FakePipeline()
    restored._iterator = SimpleNamespace()
    restored.restore_state(state)

    assert restored._pipeline.restored == {"cursor": 17}
