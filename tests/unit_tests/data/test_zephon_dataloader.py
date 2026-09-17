# Copyright (c) 2026, DatologyAI. All rights reserved.

import hashlib
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


def test_zephon_loader_matches_shared_training_pipeline_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_options = []
    captured_work_source_options = {}
    pipeline_operations = []
    token_estimation = object()
    tokenizer = object()

    class FakeDataset:
        @staticmethod
        def from_path(**kwargs):
            dataset_options.append(kwargs)
            return kwargs

    class FakeMixtureSpec:
        def __init__(self, weights):
            self.weights = weights

    class FakeStaticMixtureWorkSource:
        def __init__(self, **kwargs):
            captured_work_source_options.update(kwargs)

    class FakeTokenEstimation:
        def __new__(cls, *args, **kwargs):
            assert args == ()
            assert kwargs == {}
            return token_estimation

    class FakePipeline:
        def __init__(self, _work_source):
            pass

        def tokenize(self, **kwargs):
            pipeline_operations.append(("tokenize", kwargs))
            return self

        def pack_flat(self, **kwargs):
            pipeline_operations.append(("pack_flat", kwargs))
            return self

        def batch(self, *args, **kwargs):
            pipeline_operations.append(("batch", args, kwargs))
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
    config = zephon_dataloader.ZephonDataConfig(
        sources=(
            zephon_dataloader.ZephonSource(name="prose", path="prose", fmt="jsonl", weight=3.0),
            zephon_dataloader.ZephonSource(name="code", path="code", fmt="jsonl", weight=1.0),
        ),
        chunk_size=4,
    )

    MegatronZephonDataLoader(
        config,
        tokenizer=SimpleNamespace(_tokenizer=SimpleNamespace(tokenizer=tokenizer)),
        micro_batch_size=2,
        sequence_length=16,
        data_parallel_rank=0,
        data_parallel_size=1,
    )

    assert dataset_options == [
        {"name": "prose", "path": "prose", "fmt": "jsonl"},
        {"name": "code", "path": "code", "fmt": "jsonl"},
    ]
    assert captured_work_source_options["mixture"].weights == {"prose": 3.0, "code": 1.0}
    assert captured_work_source_options["chunk_size"] == 4
    assert captured_work_source_options["seed"] == 42
    assert captured_work_source_options["exhausted_policy"] == "repeat"
    assert captured_work_source_options["shuffle_shards"] is True
    assert captured_work_source_options["shuffle_within_shard"] is True
    assert captured_work_source_options["token_estimation"] is token_estimation
    assert pipeline_operations == [
        (
            "tokenize",
            {
                "tokenizer": tokenizer,
                "field": "text",
                "add_attention_mask": False,
                "max_length": 17,
                "split_long_samples": True,
                "special_tokens": "bos_eos",
            },
        ),
        ("pack_flat", {"max_length": 17, "algorithm": "wrap", "emit_positions": True}),
        ("batch", (2,), {"drop_last": True}),
    ]


def test_load_zephon_data_config_resolves_weighted_sources() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = load_zephon_data_config(repo_root / "examples/zephon/local_jsonl.toml")

    assert [(source.name, source.fmt, source.weight) for source in config.sources] == [
        ("prose", "jsonl", 3.0),
        ("code", "jsonl", 1.0),
    ]
    assert config.sources[0].path == str(repo_root / "tests/assets/zephon_mixture/prose")
    assert config.chunk_size == 4


def test_shared_zephon_artifact_hashes() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    expected_hashes = {
        "examples/zephon/local_jsonl.toml": (
            "18f7f2aeb2960a3f2a8527dcafb795a6eabdf8cd488510fd85ad558767beac98"
        ),
        "examples/zephon/elastic_local_jsonl.toml": (
            "17e7657a445598b5434a459848c4d8de0370814397cbc575e43717dd90d76ed9"
        ),
        "tests/assets/zephon_mixture/prose/data.jsonl": (
            "849b787063a051d8e4247b9006d7f16a4f4824692b2d14fae846b8ac7a0fbc1e"
        ),
        "tests/assets/zephon_mixture/code/data.jsonl": (
            "c81b60a1d641b40db0226112ea89e1bdaff42c6efb393e632bfe44c6f5d1b42a"
        ),
    }

    for relative_path, expected_hash in expected_hashes.items():
        artifact = repo_root / relative_path
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected_hash


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
