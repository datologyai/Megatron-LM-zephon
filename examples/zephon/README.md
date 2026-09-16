# Zephon GPT dataloader

This opt-in entry point replaces only Megatron's GPT training dataloader with a
deterministic Zephon pipeline. The stock `pretrain_gpt.py` path is unchanged.

The initial integration intentionally supports fixed-length GPT pretraining
with Hugging Face tokenizers and context parallel size 1. Validation and test
loaders are disabled; launch with `--eval-iters 0`.

## Install the private dependency

Use the Megatron development container with a Git client authenticated for the
private `datologyai/zephon` repository:

```bash
uv pip install -r requirements-zephon.txt
```

Run `scripts/validate_zephon_install.sh` for a clean-environment integration
check. Set `ZEPHON_WHEEL=/path/to/zephon.whl` to validate a local build instead
of the pinned Git tag. For local Zephon development, install a sibling checkout
with `uv pip install -e`.

## Data recipe

```toml
text_field = "text"
cache_dir = "/local-ssd/zephon"

[[sources]]
name = "web"
path = "s3://example-bucket/web"
fmt = "parquet"
weight = 3.0

[[sources]]
name = "code"
path = "s3://example-bucket/code"
weight = 1.0
```

Relative paths are resolved from the recipe directory. Distributed jobs must
also set `aggregate_dir` and a unique `run_id`; set `canonical_replicas` to the
largest data-parallel size across which elastic resume must remain stable.

## Launch

Use the normal Megatron GPT model arguments, replacing the entry point and data
arguments as follows:

```bash
torchrun --nproc-per-node 1 pretrain_gpt_zephon.py \
  --zephon-data-config examples/zephon/local_jsonl.toml \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model /path/to/huggingface/tokenizer \
  --dataloader-type external \
  --no-create-attention-mask-in-dataloader \
  --eval-iters 0 \
  --context-parallel-size 1 \
  ...
```

To checkpoint the data stream alongside model checkpoints, pass
`--dataloader-save /path/to/checkpoints/dataloader`. Use the same value on a
resumed run; the entry point restores the state for each data-parallel rank.
