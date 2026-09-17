# Zephon + Megatron-LM

This example replaces Megatron's GPT training dataloader with Zephon while
leaving model construction, tokenizer selection, optimization, distributed
training, and model checkpointing in Megatron. It demonstrates token-aware
source mixtures, deterministic checkpoint/resume, and an elastic resume at a
different data-parallel degree.

The stock `pretrain_gpt.py` path is unchanged. Zephon training uses the opt-in
`pretrain_gpt_zephon.py` entry point.

## Prerequisites

- Linux for GPU training; the data-only elastic demo also runs on macOS.
- A working Megatron development environment. The full smoke test is intended
  for the Megatron development container.
- `uv` and Python 3.12 for clean-environment validation.
- GitHub access to the private `datologyai/zephon` repository until Zephon has
  a public distribution.
- Two CUDA GPUs for the complete topology-change demonstration, or one CUDA GPU
  for a checkpoint/resume smoke test without a physical DP resize.

## Install

From the repository root, install the Zephon release pinned for this integration:

```bash
uv pip install -r requirements-zephon.txt
```

The pin currently resolves through the private Zephon GitHub repository and
uses your normal Git credentials. To validate a local release candidate instead,
set `ZEPHON_WHEEL=/path/to/zephon.whl` when running the clean-environment
validation script. For active Zephon development, install a sibling checkout
with `uv pip install -e /path/to/zephon`.

## Quick verification: CPU elastic demo

The fastest end-to-end check exercises the real Zephon pipeline and checkpoint
state without training a model:

```bash
uv run --no-sync python examples/zephon/elastic_resume_demo.py
uv run --no-sync python examples/zephon/elastic_resume_demo.py \
  --initial-workers 1 --resume-workers 2
```

The first command covers DP 2-to-1 and the second covers DP 1-to-2. Each creates
an uninterrupted reference stream, checkpoints a second stream after two global
steps, resumes under the requested worker count, and compares all fields in the
Megatron GPT batches. Lane-to-worker assignment may reorder batches after the
topology change, so comparison is order-independent within each global step.
Success ends with `Exact global-step match: YES`.

Pass `--work-dir PATH` to retain the checkpoint and the raw, emitted-order JSON
stream records for diagnosis.

## Run the full GPU demo

On a Linux machine with two CUDA GPUs, run:

```bash
examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke
```

Use a new output path for every invocation. The script runs a complete tiny-GPT
recipe in two phases:

1. Train through step 2 on two GPUs and save Megatron model and Zephon
   dataloader checkpoints.
2. Restore the latest completed checkpoint on one GPU and train step 3 with the
   same canonical lanes and logical global batch.

The final line is:

```text
Zephon training checkpoint/resume smoke test passed: ./outputs/zephon-training-smoke
```

The default tokenizer is `EleutherAI/gpt-neox-20b`. Set `TOKENIZER_MODEL` to a
local Hugging Face tokenizer directory to avoid downloading it:

```bash
TOKENIZER_MODEL=/path/to/tokenizer \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke
```

On a single-GPU development box, run:

```bash
FIRST_PHASE_GPUS=1 \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke-1gpu
```

That exercises model and dataloader checkpoint/resume but does not demonstrate
a physical data-parallel resize.

## What the demo does

1. Loads two raw-text JSONL sources from the checked-in TOML recipe.
2. Passes the configured 3:1 weights to Zephon as token proportions.
3. Uses bare `TokenEstimation()` to calibrate online source allocation.
4. Tokenizes with Megatron's Hugging Face tokenizer, splits long records, and
   packs fixed-length sequences online.
5. Converts each Zephon batch into Megatron's GPT tensor dictionary while
   preserving `[micro_batch_size, sequence_length]`.
6. Saves Zephon stream state beside the corresponding completed Megatron
   checkpoint.
7. Restores the logical stream after changing the physical worker count.

Zephon checkpoint aggregation is coordinated only across distinct DP streams.
TP/PP replicas are not independent contributors: Megatron broadcasts the TP0
batch to other TP ranks, equivalent PP streams share the DP identity, and only
TP0/PP0 invokes `save_state()`. Resume always reads the complete checkpoint
owned on disk by DP rank 0. Missing, malformed, or wrong-iteration dataloader
state fails the requested resume instead of starting a fresh stream.

For the exact pipeline and checkpoint contract, see the
[integration reference](../../docs/zephon.md).

## Use your own data

Copy `local_jsonl.toml` and replace its sources:

```toml
text_field = "text"
seed = 42
chunk_size = 4

[[sources]]
name = "web"
path = "/data/web"
fmt = "parquet"
weight = 3.0

[[sources]]
name = "code"
path = "/data/code"
fmt = "jsonl"
weight = 1.0
```

Relative paths are resolved from the recipe directory. Omit `fmt` to use
Zephon's format detection. Weights are relative token proportions: `3.0` and
`1.0` request a 75/25 token mixture. The integration passes them unchanged to
`MixtureSpec`; Zephon normalizes them.

Training always uses bare `TokenEstimation()`. There are no estimator tuning
knobs in the recipe and no post-tokenization `ensure_mixture()` operation.
Megatron supplies the tokenizer, sequence length, and microbatch size, so those
settings stay in Megatron's launch configuration.

For elastic training, start from `elastic_local_jsonl.toml`. Keep the recipe,
canonical replica count, aggregate directory, run ID, tokenizer, sequence
length, seed, and logical global batch unchanged across the resume. The physical
data-parallel degree may change.

## Advanced: launch the entry point directly

The smoke script is the canonical complete launch. For an existing Megatron GPT
configuration, replace `pretrain_gpt.py` with `pretrain_gpt_zephon.py`, remove
stock data-path arguments, and add:

```text
--zephon-data-config /path/to/recipe.toml
--dataloader-save /path/to/dataloader-checkpoints
--dataloader-type external
--no-create-attention-mask-in-dataloader
--eval-iters 0
--context-parallel-size 1
```

Elastic launches also require a stable `--zephon-canonical-replicas`,
`--zephon-aggregate-dir`, and `--zephon-run-id`. All model, optimizer, batch,
tokenizer, distributed, save, and load arguments remain normal Megatron
arguments. Resume only from a completed model checkpoint and use the matching
dataloader checkpoint directory.

## Validate the integration

Run the focused adapter test:

```bash
uv run pytest -q tests/unit_tests/data/test_zephon_dataloader.py
```

With access to the private Zephon release, validate installation and runtime
behavior in a fresh temporary environment:

```bash
scripts/validate_zephon_install.sh
```

The precise validation matrix is:

| Topology | Validation status |
| --- | --- |
| DP 2 -> 1, TP=PP=EP=1 | Real Zephon CPU demo passes with an exact global-step match. |
| DP 1 -> 2, TP=PP=EP=1 | Real Zephon CPU demo passes with an exact global-step match. |
| DP 1 -> 1, TP=PP=EP=1 | Existing one-GPU trainer smoke passed; not rerun for this change. |
| TP=2 and/or PP=2, fixed DP | DP-scoped Zephon identity and Megatron broadcast/checkpoint participation are unit-tested; no multi-GPU trainer run yet. |
| EP>1, fixed DP | Save/restore filename symmetry is unit-tested; no multi-GPU trainer run yet. |

The initial reference integration supports online raw text, fixed-length GPT
pretraining, Hugging Face tokenizers, and context parallel size 1. Validation
and test loaders are not implemented; use `--eval-iters 0`. Elastic TP/PP/EP
changes and arbitrary mixed model-parallel topologies have not been validated.
