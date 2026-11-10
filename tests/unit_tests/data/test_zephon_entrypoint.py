# Copyright (c) 2026, DatologyAI. All rights reserved.

"""Check Zephon startup without allocating a GPU or downloading a tokenizer."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pretrain_gpt_zephon
from megatron.training import global_vars
from megatron.training.datasets.zephon_dataloader import zephon_train_valid_test_datasets_provider


def test_entrypoint_registers_external_loader_config_before_pretrain(monkeypatch):
    monkeypatch.setattr(global_vars, "_GLOBAL_ARGS", None)
    monkeypatch.setattr(global_vars, "_GLOBAL_RUN_CONFIG", None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pretrain_gpt_zephon.py",
            "--zephon-data-config",
            "recipe.toml",
            "--tokenizer-model",
            "local-tokenizer",
            "--num-layers",
            "2",
            "--hidden-size",
            "128",
            "--num-attention-heads",
            "4",
            "--seq-length",
            "16",
            "--max-position-embeddings",
            "16",
            "--micro-batch-size",
            "1",
            "--global-batch-size",
            "2",
            "--train-iters",
            "2",
            "--eval-iters",
            "0",
            "--lr",
            "0.0001",
            "--transformer-impl",
            "local",
            "--no-persist-layer-norm",
        ],
    )
    gpt = SimpleNamespace(
        has_nvidia_modelopt=False, forward_step=object(), get_embedding_ranks=object()
    )
    monkeypatch.setitem(sys.modules, "pretrain_gpt", gpt)

    def check_pretrain(cfg, provider, model_type, forward_step, **kwargs):
        assert global_vars.get_run_config() is cfg
        # The training runtime must accept the config without constructing stock GPT datasets.
        cfg.validate()
        cfg.finalize()
        assert cfg.data.dataset is None
        assert cfg.data.dataloader.dataloader_type == "external"
        assert provider is zephon_train_valid_test_datasets_provider
        assert forward_step is gpt.forward_step
        # Tokenizer construction and vocabulary binding belong to initialize_megatron().
        assert cfg.model.vocab_size is None

    with (
        mock.patch("megatron.training.pretrain", side_effect=check_pretrain) as pretrain,
        mock.patch.object(
            global_vars,
            "initialize_runtime_services",
            side_effect=AssertionError("Runtime services must be initialized by pretrain"),
        ),
    ):
        pretrain_gpt_zephon.main()

    pretrain.assert_called_once()


def test_spawned_worker_import_keeps_megatron_unloaded():
    entrypoint = Path(pretrain_gpt_zephon.__file__).resolve()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__mp_main__'); "
            "assert not any(name == 'megatron' or name.startswith('megatron.') "
            "for name in sys.modules)",
            str(entrypoint),
        ],
        check=True,
        timeout=30,
    )
