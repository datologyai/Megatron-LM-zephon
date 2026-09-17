# Zephon dataloader integration

This opt-in reference integration replaces Megatron's GPT training dataloader
with Zephon. Megatron continues to own model construction, tokenizer selection,
optimization, distributed training, and model checkpoint coordination. The
stock `pretrain_gpt.py` path remains unchanged.

For installation and copy-paste demonstrations, start with the
[Zephon example guide](../examples/zephon/README.md).

## Integration boundary

The portable part of the integration ends at the Zephon batch:

```text
TOML recipe -> Zephon datasets and mixture -> tokenize -> pack -> batch
                                                              |
                                                              v
                     Megatron tensor dictionary -> GPT trainer
```

The recipes, fixtures, dataset construction, mixture, token estimation, and
pipeline operations match the TorchTitan-Zephon reference integration. The
implementations diverge only when they derive framework runtime topology,
adapt a Zephon batch for the trainer, and attach Zephon state to the framework
checkpoint API.

Megatron preserves the Zephon batch as tensors shaped
`[micro_batch_size, sequence_length]` and derives `tokens`, `labels`,
`loss_mask`, `position_ids`, and the optional attention mask expected by its GPT
training loop. TorchTitan has a different framework-facing representation; its
adapter behavior should not be copied into Megatron.

## Shared data contract

This reference path supports online raw text only. For training it:

1. Creates each source with `Dataset.from_path`, using the recipe's path and
   optional format.
2. Passes configured source weights unchanged to `MixtureSpec`. The weights are
   relative token proportions; Zephon performs normalization.
3. Constructs `StaticMixtureWorkSource` with the recipe's `chunk_size` and
   seed, `exhausted_policy="repeat"`, shard and within-shard shuffling enabled,
   and a bare `TokenEstimation()`.
4. Optionally applies recipe-controlled fetch parallelism.
5. Tokenizes the configured text field with Megatron's Hugging Face tokenizer,
   splits long samples, adds the shared special-token policy, and does not emit
   tokenizer attention masks.
6. Packs with `pack_flat(max_length=sequence_length + 1, algorithm="wrap",
   emit_positions=True)`.
7. Batches with Megatron's microbatch size and `drop_last=True`.

There is no post-tokenization `ensure_mixture()` operation and there are no
recipe knobs for token-estimation internals. Pretokenized and prepacked inputs
are outside the scope of this example.

## Recipe reference

Zephon-specific data configuration lives in a TOML recipe. Relative paths are
resolved from the recipe's directory.

| Field | Required | Meaning |
| --- | --- | --- |
| `text_field` | No | Raw-text field to tokenize; defaults to `text`. |
| `cache_dir` | No | Shared cache directory applied to source datasets. |
| `seed` | No | Deterministic mixture seed. |
| `chunk_size` | No | Number of work items allocated together. The shared demos use `4`. |
| `fetch_parallelism` | No | Optional parallelism for fetching records. |
| `canonical_replicas` | Elastic runs | Stable logical data-lane count across topology changes. |
| `aggregate_dir` | Elastic runs | Shared directory for aggregating lane checkpoint state. Usually supplied at launch. |
| `run_id` | Elastic runs | Stable identity for one data stream. Usually supplied at launch. |
| `sources[].name` | Yes | Stable source identity. |
| `sources[].path` | Yes | Local directory or Zephon-supported URI. |
| `sources[].fmt` | No | Explicit format passed to `Dataset.from_path`; omit for detection. |
| `sources[].weight` | No | Positive relative token proportion; defaults to `1.0`. |

The loader rejects unknown keys rather than forwarding arbitrary Zephon
options. Ordering, checkpointing, memory use, and distributed correctness can
all depend on those choices, so this example keeps its public surface small.

## Checkpoint and elastic-resume contract

At a Megatron checkpoint boundary, the external loader saves Zephon's complete
public checkpoint object beneath `--dataloader-save`. On resume, the integration
restores the dataloader state associated with the same completed Megatron
iteration before returning the iterator. Interrupted or partial checkpoints are
not valid resume points.

Zephon aggregates all active data-parallel participants into a complete logical
stream checkpoint. Megatron still writes the returned state for each DP rank,
but DP rank 0 is the canonical on-disk owner used for resume. Every active DP
rank restores that same canonical file, including after scale-out. The outer
checkpoint records its Megatron iteration; a missing file, malformed payload,
or iteration mismatch is a hard resume error. A fresh run without `--load` does
not require dataloader checkpoint state.

For an elastic resume, keep these values unchanged:

- recipe and source identities;
- tokenizer, sequence length, and logical global batch;
- seed and canonical replica count;
- aggregate directory and run ID.

The physical data-parallel world size may change. Zephon preserves the same
canonical-lane batch contents within each global training step, but lane
presentation order may change when lanes are reassigned to workers. Therefore,
elastic comparisons are order-independent within each global step.

Zephon's coordination world contains distinct data/checkpoint participants,
not every Megatron process: `world_size` and `dp_degree` are the data-parallel
size, while `global_rank` and `dp_group_id` are the data-parallel rank. TP ranks
receive the TP0 batch through Megatron's existing tensor-parallel broadcast.
Pipeline stages that advance equivalent deterministic streams use the same DP
identity, and only TP0/PP0 calls `save_state()`, so model-parallel replicas are
not additional Zephon checkpoint contributors. Dataloader files always use the
canonical TP0/PP0, non-expert-qualified directory on both save and restore.

## Behavior and defaults

| Choice | Training behavior | Reason |
| --- | --- | --- |
| Mixture unit | Tokens | Weights describe model-visible token proportions rather than document counts. |
| Source exhaustion | Repeat | Megatron training is step based. |
| Shard order | Shuffled | Avoid long runs of adjacent source data. |
| Within-shard order | Shuffled | Avoid preserving local record order during training. |
| Token estimation | Bare `TokenEstimation()` | Calibrate online sources without integration-specific tuning knobs. |
| Long samples | Split | Preserve usable tokens instead of truncating the record. |
| Packing | Wrap, with positions | Produce complete fixed-length GPT sequences. |
| Attention mask | Not emitted by tokenizer | Megatron constructs the trainer batch according to its own attention-mask settings. |
| Checkpoint state | Opaque Zephon object | Preserve the complete public Zephon checkpoint through Megatron's checkpoint lifecycle. |

Runtime rank, world-size, and framework options are Megatron concerns rather
than portable recipe settings.

## Validation and current limitations

This initial integration supports fixed-length GPT pretraining with Hugging
Face tokenizers and context parallel size 1. Zephon validation and test loaders
are not implemented; launch with `--eval-iters 0`. A future validation path
should use a separate finite, deterministic, unshuffled work source without
token estimation.

Zephon is currently private. `requirements-zephon.txt` pins the integration to
a reviewed private release tag; a public distribution should replace that pin
before this repository becomes public.

The tested topology matrix for this change is deliberately narrow:

| Topology | Coverage | Result |
| --- | --- | --- |
| DP 2 -> 1, TP=PP=EP=1 | Real Zephon CPU stream checkpoint/resume demo | Exact global-step match. |
| DP 1 -> 2, TP=PP=EP=1 | Real Zephon CPU stream checkpoint/resume demo | Exact global-step match. |
| DP 1 -> 1, TP=PP=EP=1 | Existing one-GPU trainer smoke result | Model and dataloader resume passed; not rerun for this change. |
| TP=2 and PP=2, fixed DP | Unit contract for Zephon runtime identity plus Megatron's existing TP broadcast/checkpoint-participant path | Options are DP-scoped; no multi-GPU trainer run in this change. |
| EP>1, fixed DP | Unit contract for save/restore path construction | Both resolve the same non-expert-qualified file; no multi-GPU trainer run in this change. |

Elastic TP, PP, or EP changes, context parallelism greater than one, validation
and test dataloaders, and pretokenized or prepacked inputs remain unsupported or
unvalidated. The matrix above should not be read as general model-parallel or
arbitrary-topology elastic support.

Run the public adapter test without installing Zephon:

```bash
uv run pytest -q tests/unit_tests/data/test_zephon_dataloader.py
```

Run the clean-environment release validation when private Zephon access is
available:

```bash
scripts/validate_zephon_install.sh
```

The validation script installs the pinned release, runs the adapter test with
the real package, and executes the CPU elastic demonstration in both 2-to-1 and
1-to-2 configurations.
