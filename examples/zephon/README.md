# Zephon + Megatron-LM

This example replaces Megatron's GPT training dataloader with a Zephon
pipeline. Megatron continues to own model construction, optimization,
distributed training, and model checkpoint coordination. Zephon is opt-in;
the stock `pretrain_gpt.py` path is unchanged.

For Zephon concepts and API details, use the
[Zephon User Guide](https://datologyai.github.io/zephon/). This page covers the
Megatron adapter and its runnable examples.

## Install

Create a normal Megatron development environment, then install the current
Zephon `main` branch used by this integration:

```bash
uv pip install -r requirements-zephon.txt
```

The dependency currently resolves through the private `datologyai/zephon`
repository and requires GitHub access. For active Zephon development, install
a sibling checkout with `uv pip install -e /path/to/zephon`.

## CPU data-only elastic determinism demo

`elastic_resume_demo.py` tests that the dataloader stream remains deterministic
across checkpoint/resume with a changed DP degree, without constructing or
training a model:

```bash
uv run --no-sync python examples/zephon/elastic_resume_demo.py
uv run --no-sync python examples/zephon/elastic_resume_demo.py \
  --initial-workers 1 --resume-workers 2
```

The two commands cover DP 2 -> 1 and DP 1 -> 2. Each compares an uninterrupted
stream with a stream restored after two global steps. Lane assignment can
change after a resize, so batches are compared without regard to order inside
each global step. Use `--work-dir PATH` to retain the generated checkpoint and
stream records.

Each demo worker selects the inline runner with MTP disabled because the demo
tests stream correctness, not nested process orchestration. The GPU smoke test
uses the production process-runner and automatic-MTP defaults.

## GPU trainer checkpoint smoke test

`run_training_smoke.sh` trains a tiny GPT model through step 2, saves the model
and Zephon stream, restores with a configurable GPU count, and completes step
3:

```bash
examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke
```

Use a new output path for each run. The default topology is two GPUs to one
GPU. A single-GPU checkpoint/resume run is:

```bash
FIRST_PHASE_GPUS=1 SECOND_PHASE_GPUS=1 \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke-1gpu
```

Set `TOKENIZER_MODEL` to a local Hugging Face tokenizer directory to avoid a
download. `FIRST_PHASE_GPUS`, `SECOND_PHASE_GPUS`, `CANONICAL_REPLICAS`, and
`ZEPHON_RUN_ID` configure the two phases.

Additional Megatron arguments may follow the output path. For example, a fixed
TP=2 smoke test is:

```bash
FIRST_PHASE_GPUS=2 SECOND_PHASE_GPUS=2 \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke-tp2 \
  --tensor-model-parallel-size 2
```

A fixed PP=2 smoke test is:

```bash
FIRST_PHASE_GPUS=2 SECOND_PHASE_GPUS=2 \
  examples/zephon/run_training_smoke.sh ./outputs/zephon-training-smoke-pp2 \
  --pipeline-model-parallel-size 2
```

## Configure data

Copy `local_jsonl.toml` and replace its sources:

```toml
text_field = "text"
cache_dir = "/local-ssd/zephon"
cache_limit_bytes = "500gb"
seed = 42
chunk_size = 16384
shuffle_block_size = "auto"
tokenize_parallelism = 8
pack_parallelism = 8

[[sources]]
name = "web"
path = "s3://example-bucket/web"
fmt = "parquet"
weight = 3.0

[[sources]]
name = "code"
path = "hf://organization/code/train"
weight = 1.0
```

Relative paths are resolved from the recipe directory. Source weights are
relative token proportions when `token_estimation = true`, which is the
training default. Zephon normalizes the weights.

The scheduling controls are independent:

- `shuffle_shards` and `shuffle_within_shard` control source ordering.
- `shuffle_block_size` accepts `"auto"`, `"global"`, `"none"`, or a positive
  integer. `"none"` disables block shuffling.
- `token_estimation` chooses token-aware rather than record-aware proportions.
- `repeat` controls source exhaustion.
- `shuffle_after_pack` controls the shuffle over packed sequences;
  `shuffle_buffer_size` and `shuffle_parallelism` tune that stage.

Shard prefetch is off by default. Set `prefetch_buffer_size` and optionally
`prefetch_parallelism` for remote datasets. `fetch_parallelism`,
`tokenize_parallelism`, and `pack_parallelism` tune the individual stages. The
default runner is `"process"`; leaving `mtp_mode` unset enables MTP for this
tokenize-and-pack pipeline.

`cache_limit_bytes` accepts either an integer byte count or a human-readable
size such as `"50gb"`.

Megatron supplies the tokenizer identifier, sequence length, and microbatch
size, so those settings stay in Megatron's launch configuration. Zephon loads
its own tokenizer instance from the same `--tokenizer-model` value. The adapter
also forwards Megatron's resolved BOS and EOS token IDs so Zephon's document
boundaries cannot diverge from the model-facing tokenizer. Keep the tokenizer
assets unchanged across resume.

## Launch the entry point directly

Replace `pretrain_gpt.py` with `pretrain_gpt_zephon.py`, remove stock data-path
arguments, and add:

```text
--zephon-data-config /path/to/recipe.toml
--dataloader-save /path/to/dataloader-checkpoints
--dataloader-type external
--no-create-attention-mask-in-dataloader
--eval-iters 0
--context-parallel-size 1
```

Elastic launches also require stable `--zephon-canonical-replicas`,
`--zephon-aggregate-dir`, and `--zephon-run-id` values. The canonical lane
count must be divisible by every supported DP degree, and every optimizer step
must consume a whole number of canonical lane windows. Batch-size ramp-up is
not supported by this reference integration, including
`--step-batch-size-schedule`.

Keep the recipe, tokenizer, sequence length, seed, logical global batch,
canonical lane count, aggregate directory, and run ID unchanged across resume.
The physical data-parallel degree may change. Resume only from a completed
model checkpoint with the corresponding dataloader checkpoint directory.
Model checkpointing through `--save` requires `--dataloader-save`; iteration-zero
model initialization starts a fresh Zephon stream, while later resumes require
the exactly matching dataloader checkpoint.

Use `--dataloader-inter-document-masking` to restrict attention across packed
document boundaries. The legacy `--reset-attention-mask` option is accepted
only when that explicit option is also enabled. `--reset-position-ids` and
`--eod-mask-loss` retain their standard Megatron semantics.

## Integration contract

The adapter supports online raw text and fixed-length GPT pretraining. It:

1. Builds each source with `Dataset.from_path` and passes source weights
   unchanged to `MixtureSpec`.
2. Tokenizes and wraps packed sequences online with Zephon.
3. Uses `SampleBatch.to_training()` to create next-token labels while
   preserving Megatron tensors shaped `[micro_batch_size, sequence_length]`.
4. Produces the Megatron loss mask and position IDs, optionally masks EOD
   transitions, and emits packed-document sequence metadata when requested.
5. Stores Zephon's complete public checkpoint object beside the corresponding
   completed Megatron checkpoint.

Zephon coordination represents distinct data streams: DP size and DP rank are
used for its world and rank identity. Megatron broadcasts the TP0 batch to
other TP ranks, equivalent pipeline stages share the DP identity, and only the
canonical TP0/PP0 participant writes the dataloader checkpoint. Every active
DP rank restores the complete checkpoint owned on disk by DP rank 0.

## Validate the integration

Run the focused adapter and checkpoint tests inside the Megatron development
container:

```bash
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/data/test_zephon_dataloader.py
uv run python -m torch.distributed.run --nproc-per-node 8 -m pytest -q \
  tests/unit_tests/test_checkpointing.py -k maybe_save_dataloader_state
```

Then run both CPU elastic directions and the GPU smoke configurations relevant
to the topology being claimed.

| Topology | Required evidence |
| --- | --- |
| DP 2 -> 1 and DP 1 -> 2, TP=PP=EP=1 | Exact global-step match from the CPU data-only demo. |
| DP 1 -> 1, TP=PP=EP=1 | Model and dataloader checkpoint/resume through the GPU smoke test. |
| TP=2 or PP=2 with fixed DP | Successful GPU trainer checkpoint/resume; unit contracts alone are insufficient. |
| EP>1 with fixed DP | Save/restore path symmetry plus a GPU trainer run before support is claimed. |

The current reference path does not implement validation or test loaders,
context parallelism greater than one, pretokenized/prepacked input, batch-size
ramp-up or step batch-size schedules, virtual pipeline parallelism, in-process
restart, or elastic TP/PP/EP changes.
