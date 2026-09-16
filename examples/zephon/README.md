# Zephon GPT reference integration

This opt-in reference integration replaces only Megatron's GPT training
dataloader with a deterministic Zephon pipeline. Model construction,
tokenization configuration, optimization, parallel training, and Megatron's
checkpoint coordinator remain unchanged. The stock `pretrain_gpt.py` path is
also unchanged.

The integration demonstrates the same portable data recipe and elastic
two-worker to one-worker resume used by the TorchTitan-Zephon reference
integration. Megatron-specific model, batch, and checkpoint settings remain in
Megatron's command line.

## Install

Use the Megatron development container with a Git client authenticated for the
private `datologyai/zephon` repository:

```bash
uv pip install -r requirements-zephon.txt
```

Run `scripts/validate_zephon_install.sh` for a clean-environment integration
check. Set `ZEPHON_WHEEL=/path/to/zephon.whl` to validate a local build instead
of the pinned Git tag. For local Zephon development, install a sibling checkout
with `uv pip install -e`.

## Configure the data recipe

`local_jsonl.toml` is the common base recipe shared by the Megatron and
TorchTitan reference integrations:

```toml
text_field = "text"
seed = 42
chunk_size = 4

[[sources]]
name = "prose"
path = "../../tests/assets/zephon_mixture/prose"
fmt = "jsonl"
weight = 3.0

[[sources]]
name = "code"
path = "../../tests/assets/zephon_mixture/code"
fmt = "jsonl"
weight = 1.0
```

Relative paths are resolved from the recipe directory. The weights request a
75/25 prose/code mixture and Zephon normalizes them automatically. Megatron
supplies the tokenizer, sequence length, and microbatch size, so those settings
do not appear in the reusable recipe.

## Launch GPT pretraining

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

The remaining model, optimizer, batch, and distributed arguments are the same
ones used by the corresponding `pretrain_gpt.py` launch.

## Prove elastic deterministic resume

Run the complete data-stream demonstration in a CPU-only Linux environment:

```bash
uv run --no-sync python examples/zephon/elastic_resume_demo.py
```

The command creates an uninterrupted two-worker reference stream, checkpoints
the same stream after two steps, resumes it with one worker, and compares every
field in the Megatron GPT batch. It exits unsuccessfully if any field differs.
Pass `--work-dir PATH` to keep the checkpoint and JSON stream records.

`elastic_local_jsonl.toml` fixes the logical stream at two canonical replicas.
The shared aggregate directory and run ID describe one execution, so provide
them at launch instead of storing them in the reusable recipe:

```bash
torchrun --nproc-per-node 2 pretrain_gpt_zephon.py \
  --zephon-data-config examples/zephon/elastic_local_jsonl.toml \
  --zephon-canonical-replicas 2 \
  --zephon-aggregate-dir /mnt/zephon-aggregate \
  --zephon-run-id example-run \
  --dataloader-save ./checkpoints/dataloader \
  --save ./checkpoints/model \
  ...
```

Resume the completed checkpoint with one worker by keeping the recipe,
canonical replica count, aggregate directory, run ID, tokenizer, sequence
length, and global batch definition unchanged. Point `--load` at the model
checkpoint and reuse `--dataloader-save`:

```bash
torchrun --nproc-per-node 1 pretrain_gpt_zephon.py \
  --zephon-data-config examples/zephon/elastic_local_jsonl.toml \
  --zephon-canonical-replicas 2 \
  --zephon-aggregate-dir /mnt/zephon-aggregate \
  --zephon-run-id example-run \
  --dataloader-save ./checkpoints/dataloader \
  --load ./checkpoints/model \
  --save ./checkpoints/model \
  ...
```

The physical data-parallel degree may change. The canonical replica count and
logical global batch must not.

## Training checkpoint smoke test

On a Linux machine with two CUDA GPUs, run the same bounded training and
elastic-resume workflow as the TorchTitan-Zephon reference integration:

```bash
examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke
```

The script uses a complete tiny-GPT recipe. Its first launch trains through
step 2 on two GPUs and saves model and Zephon dataloader checkpoints. Its
second launch restores the latest completed checkpoint on one GPU and trains
step 3 with the same canonical lane count, run identity, and logical global
batch. Set `TOKENIZER_MODEL` to a local Hugging Face tokenizer directory to
avoid downloading the default public tokenizer. Use a new output path for
each invocation. On a single-GPU development box, set `FIRST_PHASE_GPUS=1`;
this tests the complete checkpoint/resume path but not the physical
data-parallel resize.

## Checkpoint contract and current scope

At each checkpoint boundary, Megatron calls the external loader's
`save_state()` method and stores the opaque Zephon checkpoint under
`--dataloader-save`. On resume, the integration restores the dataloader state
from the same completed Megatron iteration before returning the iterator.

The initial integration supports fixed-length GPT pretraining with Hugging
Face tokenizers and context parallel size 1. Validation and test loaders are
disabled; launch with `--eval-iters 0`.
