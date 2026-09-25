#!/usr/bin/env bash
# Copyright (c) 2026, DatologyAI. All rights reserved.

# End-to-end GPU smoke test: train, checkpoint the model and Zephon stream,
# resume with a configurable GPU count, and complete one additional step.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
dump_folder=${1:-"${repo_root}/outputs/zephon-training-smoke"}
first_phase_gpus=${FIRST_PHASE_GPUS:-2}
second_phase_gpus=${SECOND_PHASE_GPUS:-1}
first_phase_steps=${FIRST_PHASE_STEPS:-2}
total_steps=${TOTAL_STEPS:-3}
canonical_replicas=${CANONICAL_REPLICAS:-2}
run_id=${ZEPHON_RUN_ID:-zephon-training-smoke}
tokenizer_model=${TOKENIZER_MODEL:-EleutherAI/gpt-neox-20b}
python_bin=${PYTHON:-python3}
extra_args=("${@:2}")

if [[ -e "${dump_folder}" ]]; then
    echo "Refusing to reuse existing output: ${dump_folder}" >&2
    echo "Pass a new output path or move the existing directory." >&2
    exit 2
fi

cd "${repo_root}"

common_args=(
    --zephon-data-config examples/zephon/elastic_local_jsonl.toml
    --zephon-canonical-replicas "${canonical_replicas}"
    --zephon-aggregate-dir "${dump_folder}/zephon-aggregate"
    --zephon-run-id "${run_id}"
    --dataloader-save "${dump_folder}/dataloader"
    --save "${dump_folder}/model"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "${tokenizer_model}"
    --no-create-attention-mask-in-dataloader
    --num-layers 2
    --hidden-size 128
    --ffn-hidden-size 512
    --num-attention-heads 4
    --seq-length 128
    --max-position-embeddings 128
    --micro-batch-size 1
    --global-batch-size 2
    --lr 0.0001
    --min-lr 0.00001
    --lr-decay-style cosine
    --lr-decay-iters "${total_steps}"
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.02
    --clip-grad 1.0
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --transformer-impl local
    --no-persist-layer-norm
    --no-gradient-accumulation-fusion
    --no-masked-softmax-fusion
    --no-bias-gelu-fusion
    --no-bias-dropout-fusion
    --bf16
    --eval-iters 0
    --eval-interval 1000
    --log-interval 1
    --save-interval 1
    --ckpt-format torch_dist
)

echo "Phase 1: train ${first_phase_steps} steps with ${first_phase_gpus} GPU(s)"
"${python_bin}" -m torch.distributed.run \
    --nproc-per-node "${first_phase_gpus}" pretrain_gpt_zephon.py \
    "${common_args[@]}" \
    "${extra_args[@]}" \
    --train-iters "${first_phase_steps}"

echo "Phase 2: resume through step ${total_steps} with ${second_phase_gpus} GPU(s)"
"${python_bin}" -m torch.distributed.run \
    --nproc-per-node "${second_phase_gpus}" pretrain_gpt_zephon.py \
    "${common_args[@]}" \
    "${extra_args[@]}" \
    --load "${dump_folder}/model" \
    --override-opt-param-scheduler \
    --train-iters "${total_steps}"

echo "Zephon training checkpoint/resume smoke test passed: ${dump_folder}"
