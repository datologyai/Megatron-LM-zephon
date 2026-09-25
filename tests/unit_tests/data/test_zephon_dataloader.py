# Copyright (c) 2026, DatologyAI. All rights reserved.

import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from zephon import SampleBatch, SampleMeta, SampleRecord
from zephon.io import StoreOptions

from megatron.training.checkpointing import get_dataloader_checkpoint_name
from megatron.training.datasets.zephon_dataloader import (
    TOKENS_FIELD,
    MegatronZephonDataLoader,
    ZephonDataConfig,
    ZephonSource,
    _build_zephon_runtime_options,
    _restore_dataloader_state,
    _validate_zephon_launch,
    _zephon_dataloader_checkpoint_path,
    _zephon_tokenizer_id_from_args,
    apply_zephon_runtime_overrides,
    load_zephon_data_config,
    zephon_train_valid_test_datasets_provider,
)


def test_load_zephon_data_config(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = load_zephon_data_config(repo_root / "examples/zephon/local_jsonl.toml")

    assert [(source.name, source.fmt, source.weight) for source in config.sources] == [
        ("prose", "jsonl", 3.0),
        ("code", "jsonl", 1.0),
    ]
    assert config.sources[0].path == str(repo_root / "tests/assets/zephon_mixture/prose")
    assert config.chunk_size == 4
    assert ZephonDataConfig(sources=()).chunk_size == 16_384

    invalid_recipes = {
        "duplicate.toml": (
            """
[[sources]]
name = "same"
path = "one"

[[sources]]
name = "same"
path = "two"
""",
            "names must be unique",
        ),
        "unknown.toml": (
            """
unexpected = true

[[sources]]
name = "data"
path = "data"
""",
            "Unknown Zephon data recipe keys",
        ),
        "unknown-source-field.toml": (
            """
[[sources]]
name = "data"
path = "data"
wieght = 9.0
""",
            "Unknown Zephon source fields: wieght",
        ),
        "nan-weight.toml": (
            """
[[sources]]
name = "data"
path = "data"
weight = nan
""",
            "weights must all be finite",
        ),
        "infinite-weight.toml": (
            """
[[sources]]
name = "data"
path = "data"
weight = inf
""",
            "weights must all be finite",
        ),
    }
    for filename, (contents, message) in invalid_recipes.items():
        recipe = tmp_path / filename
        recipe.write_text(contents)
        with pytest.raises(ValueError, match=message):
            load_zephon_data_config(recipe)


def test_load_zephon_data_config_normalizes_explicitly_disabled_block_shuffle(
    tmp_path: Path,
) -> None:
    recipe = tmp_path / "no-block-shuffle.toml"
    recipe.write_text("""
shuffle_block_size = "none"

[[sources]]
name = "data"
path = "data"
""")

    assert load_zephon_data_config(recipe).shuffle_block_size is None


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


@pytest.mark.parametrize("eos_token_id", [0, 7])
@pytest.mark.parametrize("eod_mask_loss", [False, True])
@pytest.mark.parametrize("reset_position_ids", [False, True])
@pytest.mark.parametrize("return_cu_seqlens", [False, True])
@pytest.mark.filterwarnings("error:eos_token_id is ignored")
def test_zephon_loader_maps_batches_to_megatron_schema(
    eos_token_id: int, eod_mask_loss: bool, reset_position_ids: bool, return_cu_seqlens: bool
) -> None:
    tokens = [[1, 10, eos_token_id, 1, 20, eos_token_id], [1, 30, 31, eos_token_id, 1, 40]]
    positions = [[0, 1, 2, 0, 1, 2], [0, 1, 2, 3, 0, 1]]
    sample_batch = SampleBatch(
        records=tuple(
            SampleRecord(
                meta=SampleMeta(sample_id=(0, 0, i), lane_id=0, chunk_id=0),
                payload={TOKENS_FIELD: token_row, "positions": position_row},
            )
            for i, (token_row, position_row) in enumerate(zip(tokens, positions))
        )
    )
    loader = MegatronZephonDataLoader.__new__(MegatronZephonDataLoader)
    loader._iterator = iter([sample_batch])
    loader._eos_token_id = eos_token_id
    loader._reset_position_ids = reset_position_ids
    loader._eos_mask_loss = eod_mask_loss
    loader._return_cu_seqlens = return_cu_seqlens

    batch = next(loader)

    expected_keys = {"tokens", "labels", "loss_mask", "position_ids"}
    if return_cu_seqlens:
        expected_keys.update(("cu_seqlens", "max_seqlen"))
        assert torch.equal(batch["cu_seqlens"], torch.tensor([[0, 3, 5], [0, 4, 5]]))
        assert torch.equal(batch["max_seqlen"], torch.tensor([3, 4]))
    assert set(batch) == expected_keys
    assert torch.equal(batch["tokens"], torch.tensor(tokens)[:, :-1])
    boundary_label = -100 if eod_mask_loss else 1
    assert torch.equal(
        batch["labels"],
        torch.tensor(
            [
                [10, eos_token_id, boundary_label, 20, eos_token_id],
                [30, 31, eos_token_id, boundary_label, 40],
            ]
        ),
    )
    boundary_loss = 0.0 if eod_mask_loss else 1.0
    assert torch.equal(
        batch["loss_mask"], torch.tensor([[1, 1, boundary_loss, 1, 1], [1, 1, 1, boundary_loss, 1]])
    )
    expected_positions = (
        [row[:-1] for row in positions] if reset_position_ids else [list(range(5))] * 2
    )
    assert torch.equal(batch["position_ids"], torch.tensor(expected_positions))


def test_zephon_loader_constructs_its_own_tokenizer_from_identifier() -> None:
    pipeline = mock.MagicMock()
    pipeline.tokenize.return_value = pipeline
    pipeline.pack_flat.return_value = pipeline
    pipeline.shuffle.return_value = pipeline
    pipeline.batch.return_value = pipeline
    pipeline.options.return_value = pipeline
    pipeline.preflight_tokenizers.return_value = None
    pipeline.__iter__.return_value = iter(())
    dataset_type = mock.Mock()
    dataset_type.from_path.return_value = object()
    config = ZephonDataConfig(
        sources=(ZephonSource(name="source", path="data"),), canonical_replicas=1
    )

    with (
        mock.patch("megatron.training.datasets.zephon_dataloader.Pipeline", return_value=pipeline),
        mock.patch("megatron.training.datasets.zephon_dataloader.Dataset", dataset_type),
        mock.patch("megatron.training.datasets.zephon_dataloader.MixtureSpec"),
        mock.patch("megatron.training.datasets.zephon_dataloader.StaticMixtureWorkSource"),
        mock.patch("megatron.training.datasets.zephon_dataloader.TokenEstimation"),
    ):
        loader = MegatronZephonDataLoader(
            config,
            tokenizer_id="example/tokenizer",
            bos_token_id=11,
            eos_token_id=12,
            micro_batch_size=2,
            sequence_length=16,
            data_parallel_rank=0,
            data_parallel_size=1,
        )

    assert pipeline.tokenize.call_args.kwargs["tokenizer_id"] == "example/tokenizer"
    assert pipeline.tokenize.call_args.kwargs["bos_token_id"] == 11
    assert pipeline.tokenize.call_args.kwargs["eos_token_id"] == 12
    assert loader._eos_token_id == pipeline.tokenize.call_args.kwargs["eos_token_id"]
    assert "tokenizer" not in pipeline.tokenize.call_args.kwargs
    pipeline.preflight_tokenizers.assert_called_once_with()
    pipeline.__iter__.assert_not_called()


def test_runtime_options_use_data_parallel_coordination_identity() -> None:
    config = SimpleNamespace(
        canonical_replicas=8,
        cache_dir="/cache",
        cache_limit_bytes="50gb",
        aggregate_dir="/aggregate",
        run_id="run",
        runner="process",
        mtp_mode=None,
    )

    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch("torch.distributed.get_world_size", return_value=16),
        mock.patch("torch.distributed.get_rank", return_value=11),
    ):
        options = _build_zephon_runtime_options(config, data_parallel_rank=2, data_parallel_size=4)

    io_options = options.pop("io_options")
    assert options == {
        "deterministic": True,
        "dp_degree": 4,
        "dp_group_id": 2,
        "world_size": 4,
        "global_rank": 2,
        "runner": "process",
        "mtp_mode": True,
        "canonical_replicas": 8,
        "aggregate_dir": "/aggregate",
        "run_id": "run",
    }
    assert isinstance(io_options, StoreOptions)
    assert io_options.cache.enabled is True
    assert str(io_options.cache.root) == "/cache"
    assert io_options.cache.limit_bytes == 50 * 1024**3


@pytest.mark.parametrize(
    ("canonical_replicas", "data_parallel_size", "num_batches_per_train_step", "message"),
    [
        (3, 2, 4, "divisible by the current data-parallel size"),
        (4, 2, 6, "complete lane-window boundaries"),
    ],
)
def test_loader_rejects_invalid_elastic_alignment(
    canonical_replicas: int, data_parallel_size: int, num_batches_per_train_step: int, message: str
) -> None:
    config = SimpleNamespace(
        sources=(), canonical_replicas=canonical_replicas, aggregate_dir="/aggregate", run_id="run"
    )

    with pytest.raises(ValueError, match=message):
        MegatronZephonDataLoader(
            config,
            tokenizer_id="example/tokenizer",
            bos_token_id=1,
            eos_token_id=2,
            micro_batch_size=1,
            sequence_length=16,
            data_parallel_rank=0,
            data_parallel_size=data_parallel_size,
            num_batches_per_train_step=num_batches_per_train_step,
        )


def test_dataloader_checkpoint_path_is_symmetric_with_pipeline_and_expert_parallelism(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(dataloader_save=tmp_path, iteration=7)

    with (
        mock.patch(
            "megatron.training.checkpointing.mpu.get_pipeline_model_parallel_world_size",
            return_value=2,
        ),
        mock.patch(
            "megatron.training.checkpointing.mpu.get_expert_model_parallel_world_size",
            return_value=8,
        ),
        mock.patch(
            "megatron.training.checkpointing.mpu.get_expert_model_parallel_rank", return_value=5
        ),
    ):
        restore_path = _zephon_dataloader_checkpoint_path(args)

    save_path = get_dataloader_checkpoint_name(tmp_path, 7, 0, pipeline_parallel=True)
    assert restore_path == save_path
    assert restore_path.endswith("iter_0000007/mp_rank_00_000/train_dataloader_dprank000.pt")


class _RecordingLoader:
    def __init__(self) -> None:
        self.restored = []

    def restore_state(self, state_dict) -> None:
        self.restored.append(state_dict)


def _resume_args(tmp_path: Path, *, iteration: int = 7, load: str | None = "/model"):
    return SimpleNamespace(load=load, dataloader_save=tmp_path, iteration=iteration)


def _write_dataloader_checkpoint(path: Path, *, iteration: int = 7, state=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "iteration": iteration,
            "dataloader_state_dict": state or {"zephon": b"complete-global-state"},
        },
        path,
    )


@pytest.mark.parametrize("saved_dp_size,resumed_dp_size", [(1, 2), (2, 1)])
def test_elastic_resume_uses_canonical_dp0_checkpoint_on_every_rank(
    tmp_path: Path, saved_dp_size: int, resumed_dp_size: int
) -> None:
    args = _resume_args(tmp_path)
    canonical_path = Path(get_dataloader_checkpoint_name(tmp_path, 7, 0, pipeline_parallel=False))
    _write_dataloader_checkpoint(canonical_path)
    if saved_dp_size > 1:
        _write_dataloader_checkpoint(
            Path(get_dataloader_checkpoint_name(tmp_path, 7, 1, pipeline_parallel=False)),
            state={"zephon": b"noncanonical-state"},
        )
    loaders = [_RecordingLoader() for _ in range(resumed_dp_size)]

    with mock.patch(
        "megatron.training.checkpointing.mpu.get_pipeline_model_parallel_world_size", return_value=1
    ):
        for loader in loaders:
            _restore_dataloader_state(loader, args)

    assert [loader.restored for loader in loaders] == [
        [{"zephon": b"complete-global-state"}] for _ in range(resumed_dp_size)
    ]


def test_requested_resume_requires_matching_dataloader_checkpoint(tmp_path: Path) -> None:
    args = _resume_args(tmp_path)
    missing_path = tmp_path / "missing.pt"

    with (
        mock.patch(
            "megatron.training.datasets.zephon_dataloader._zephon_dataloader_checkpoint_path",
            return_value=str(missing_path),
        ),
        pytest.raises(FileNotFoundError, match="iteration 7.*missing.pt"),
    ):
        _restore_dataloader_state(_RecordingLoader(), args)

    args.dataloader_save = None
    with pytest.raises(ValueError, match="--load requires --dataloader-save"):
        _restore_dataloader_state(_RecordingLoader(), args)


def test_fresh_training_does_not_require_dataloader_checkpoint(tmp_path: Path) -> None:
    args = _resume_args(tmp_path, load=None)
    loader = _RecordingLoader()

    _restore_dataloader_state(loader, args)

    assert loader.restored == []


def test_iteration_zero_load_starts_a_fresh_dataloader_stream(tmp_path: Path) -> None:
    loader = _RecordingLoader()

    _restore_dataloader_state(loader, _resume_args(tmp_path, iteration=0))

    assert loader.restored == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (["not", "a", "mapping"], "expected a mapping"),
        ({"dataloader_state_dict": {}}, "missing integer 'iteration'"),
        (
            {"iteration": 6, "dataloader_state_dict": {"zephon": b"state"}},
            "iteration mismatch.*expected 7, found 6",
        ),
        ({"iteration": 7}, "missing mapping 'dataloader_state_dict'"),
    ],
)
def test_malformed_dataloader_checkpoint_raises_useful_error(
    tmp_path: Path, payload, message: str
) -> None:
    path = tmp_path / "bad.pt"
    torch.save(payload, path)

    with (
        mock.patch(
            "megatron.training.datasets.zephon_dataloader._zephon_dataloader_checkpoint_path",
            return_value=str(path),
        ),
        pytest.raises(ValueError, match=message),
    ):
        _restore_dataloader_state(_RecordingLoader(), _resume_args(tmp_path))


def test_unreadable_dataloader_checkpoint_raises_useful_error(tmp_path: Path) -> None:
    path = tmp_path / "unreadable.pt"
    path.write_bytes(b"not a torch checkpoint")

    with (
        mock.patch(
            "megatron.training.datasets.zephon_dataloader._zephon_dataloader_checkpoint_path",
            return_value=str(path),
        ),
        pytest.raises(RuntimeError, match="Failed to load Zephon dataloader checkpoint"),
    ):
        _restore_dataloader_state(_RecordingLoader(), _resume_args(tmp_path))


def test_restore_state_rejects_malformed_opaque_checkpoint() -> None:
    loader = MegatronZephonDataLoader.__new__(MegatronZephonDataLoader)

    with pytest.raises(ValueError, match="contain bytes"):
        loader.restore_state({"zephon": "not-bytes"})
    with pytest.raises(ValueError, match="Unable to deserialize"):
        loader.restore_state({"zephon": b"not-a-pickle"})

    loader._pipeline = mock.Mock()
    loader._pipeline.restore.side_effect = RuntimeError("incompatible")
    with pytest.raises(ValueError, match="malformed or incompatible"):
        loader.restore_state({"zephon": pickle.dumps({"state": 1})})


def test_restore_state_leaves_iterator_initialization_lazy() -> None:
    loader = MegatronZephonDataLoader.__new__(MegatronZephonDataLoader)
    loader._pipeline = mock.MagicMock()
    loader._iterator = object()

    loader.restore_state({"zephon": pickle.dumps({"state": 1})})

    loader._pipeline.restore.assert_called_once_with({"state": 1})
    loader._pipeline.__iter__.assert_not_called()
    assert loader._iterator is None


def test_provider_gives_zephon_its_own_tokenizer_identifier() -> None:
    args = SimpleNamespace(
        eval_iters=0,
        context_parallel_size=1,
        rampup_batch_size=None,
        zephon_data_config=str(
            Path(__file__).resolve().parents[3] / "examples/zephon/local_jsonl.toml"
        ),
        zephon_canonical_replicas=None,
        zephon_aggregate_dir=None,
        zephon_run_id=None,
        micro_batch_size=2,
        seq_length=16,
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model="example/tokenizer",
        save=None,
        dataloader_save=None,
        virtual_pipeline_model_parallel_size=None,
        inprocess_restart=False,
        reset_attention_mask=False,
        dataloader_inter_document_masking=True,
        reset_position_ids=True,
        eod_mask_loss=True,
    )
    loader = object()
    tokenizer = SimpleNamespace(bos_id=11, eos_id=12)

    with (
        mock.patch("megatron.training.datasets.zephon_dataloader.get_args", return_value=args),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.mpu.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.mpu.is_pipeline_first_stage",
            return_value=False,
        ),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.mpu.is_pipeline_last_stage",
            return_value=False,
        ),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.mpu.get_data_parallel_rank",
            return_value=1,
        ),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.mpu.get_data_parallel_world_size",
            return_value=2,
        ),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.get_num_microbatches", return_value=2
        ),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.MegatronZephonDataLoader",
            return_value=loader,
        ) as loader_type,
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.get_tokenizer", return_value=tokenizer
        ),
        mock.patch("megatron.training.datasets.zephon_dataloader._restore_dataloader_state"),
    ):
        result = zephon_train_valid_test_datasets_provider([])

    assert loader_type.call_args.kwargs["tokenizer_id"] == "example/tokenizer"
    assert loader_type.call_args.kwargs["bos_token_id"] == 11
    assert loader_type.call_args.kwargs["eos_token_id"] == 12
    assert loader_type.call_args.kwargs["reset_position_ids"] is True
    assert loader_type.call_args.kwargs["eod_mask_loss"] is True
    assert loader_type.call_args.kwargs["return_cu_seqlens"] is True
    assert loader_type.call_args.kwargs["num_batches_per_train_step"] == 4
    assert result == (loader, None, None)


def _valid_launch_args(**overrides):
    values = {
        "eval_iters": 0,
        "context_parallel_size": 1,
        "rampup_batch_size": None,
        "step_batch_size_schedule": None,
        "save": None,
        "dataloader_save": None,
        "virtual_pipeline_model_parallel_size": None,
        "inprocess_restart": False,
        "reset_attention_mask": False,
        "dataloader_inter_document_masking": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"step_batch_size_schedule": "0:8 1000:16"}, "--step-batch-size-schedule"),
        ({"save": "/model"}, "--save requires --dataloader-save"),
        ({"virtual_pipeline_model_parallel_size": 2}, "virtual pipeline parallelism"),
        ({"inprocess_restart": True}, "--inprocess-restart"),
        ({"reset_attention_mask": True}, "--dataloader-inter-document-masking"),
    ],
)
def test_unsupported_launch_configurations_are_rejected(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_zephon_launch(_valid_launch_args(**overrides))


def test_explicit_inter_document_masking_allows_legacy_reset_attention_flag() -> None:
    _validate_zephon_launch(
        _valid_launch_args(reset_attention_mask=True, dataloader_inter_document_masking=True)
    )


def test_provider_skips_unused_middle_pipeline_stages() -> None:
    args = _valid_launch_args()

    with (
        mock.patch("megatron.training.datasets.zephon_dataloader.get_args", return_value=args),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.mpu.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.mpu.is_pipeline_first_stage",
            return_value=False,
        ),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.mpu.is_pipeline_last_stage",
            return_value=False,
        ),
        mock.patch(
            "megatron.training.datasets.zephon_dataloader.load_zephon_data_config"
        ) as load_config,
    ):
        result = zephon_train_valid_test_datasets_provider([])

    assert result == (None, None, None)
    load_config.assert_not_called()


def test_provider_rejects_batch_size_rampup() -> None:
    args = _valid_launch_args(rampup_batch_size=[4, 4, 100])

    with (
        mock.patch("megatron.training.datasets.zephon_dataloader.get_args", return_value=args),
        pytest.raises(ValueError, match="does not support batch-size ramp-up"),
    ):
        zephon_train_valid_test_datasets_provider([])


def test_unsupported_tokenizer_type_is_rejected_clearly() -> None:
    with pytest.raises(ValueError, match="requires --tokenizer-type HuggingFaceTokenizer"):
        _zephon_tokenizer_id_from_args(
            SimpleNamespace(tokenizer_type="TikTokenizer", tokenizer_model="example/tokenizer")
        )


def test_missing_zephon_tokenizer_identifier_is_rejected_clearly() -> None:
    with pytest.raises(ValueError, match="non-empty --tokenizer-model"):
        _zephon_tokenizer_id_from_args(
            SimpleNamespace(tokenizer_type="HuggingFaceTokenizer", tokenizer_model=None)
        )
