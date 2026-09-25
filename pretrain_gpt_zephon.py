# Copyright (c) 2026, DatologyAI. All rights reserved.

"""Pretrain Megatron GPT with an opt-in Zephon training dataloader."""

import argparse

import pretrain_gpt as gpt
from megatron.core.enums import ModelType
from megatron.training import inprocess_restart, pretrain
from megatron.training.argument_utils import (
    gpt_config_from_args,
    pretrain_cfg_container_from_args,
    resolve_tokenizer_vocab_size,
)
from megatron.training.arguments import parse_and_validate_args
from megatron.training.datasets.zephon_dataloader import zephon_train_valid_test_datasets_provider
from megatron.training.global_vars import initialize_runtime_services


def add_zephon_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments owned by the Zephon integration."""

    group = parser.add_argument_group(title="Zephon dataloader")
    group.add_argument("--zephon-data-config", required=True, help="Zephon TOML data recipe")
    group.add_argument(
        "--zephon-canonical-replicas",
        type=int,
        default=None,
        help="Stable logical data-lane count used across elastic resumes",
    )
    group.add_argument(
        "--zephon-aggregate-dir",
        default=None,
        help="Shared directory used to aggregate Zephon checkpoint state",
    )
    group.add_argument(
        "--zephon-run-id", default=None, help="Stable identity for the Zephon data stream"
    )
    group.add_argument(
        "--dataloader-save",
        default=None,
        help="Directory used to save and restore Zephon dataloader state",
    )
    if gpt.has_nvidia_modelopt:
        parser = gpt.add_modelopt_args(parser)
    return parser


if __name__ == "__main__":
    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    args = parse_and_validate_args(
        extra_args_provider=add_zephon_args,
        args_defaults={
            "tokenizer_type": "HuggingFaceTokenizer",
            "dataloader_type": "external",
            "create_attention_mask_in_dataloader": False,
        },
    )
    if gpt.has_nvidia_modelopt:
        gpt.maybe_enable_modelopt(args)
    if gpt.has_nvidia_modelopt and getattr(args, "modelopt_enabled", False):
        model_cfg = gpt_config_from_args(
            args, model_config_cls=gpt.ModelOptModelConfig, vocab_size_from_tokenizer=True
        )
    else:
        model_cfg = gpt_config_from_args(args, vocab_size_from_tokenizer=True)
    full_config = pretrain_cfg_container_from_args(args, model_cfg)
    initialize_runtime_services(args)
    resolve_tokenizer_vocab_size(full_config, args.padded_vocab_size)
    wrapped_pretrain(
        full_config,
        zephon_train_valid_test_datasets_provider,
        ModelType.encoder_or_decoder,
        gpt.forward_step,
        store=store,
        get_embedding_ranks=gpt.get_embedding_ranks,
    )
