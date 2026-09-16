#!/usr/bin/env bash
# Validate Megatron against an installed Zephon distribution in a clean venv.

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
validation_dir=$(mktemp -d "${TMPDIR:-/tmp}/megatron-zephon-validation.XXXXXX")
trap 'rm -rf "${validation_dir}"' EXIT

uv venv --python "${PYTHON_VERSION:-3.12}" "${validation_dir}/venv"
python_path="${validation_dir}/venv/bin/python"
cd "${validation_dir}"

if [[ "$(uname -m)" == "aarch64" ]]; then
  uv pip install --python "${python_path}" torch
else
  uv pip install --python "${python_path}" \
    --index-url https://download.pytorch.org/whl/cpu torch
fi
uv pip install --python "${python_path}" \
  pytest triton click requests pyyaml numpy
uv pip install --python "${python_path}" --no-deps -e "${repo_root}"

if [[ -n "${ZEPHON_WHEEL:-}" ]]; then
  uv pip install --python "${python_path}" "${ZEPHON_WHEEL}"
else
  uv pip install --python "${python_path}" \
    -r "${repo_root}/requirements-zephon.txt"
fi

cd "${repo_root}"
"${python_path}" -c \
  'import zephon; print(f"Validating Zephon {zephon.__version__} from {zephon.__file__}")'
"${python_path}" -m pytest -q \
  tests/unit_tests/data/test_zephon_dataloader.py \
  tests/unit_tests/data/test_zephon_elastic_resume.py
"${python_path}" examples/zephon/elastic_resume_demo.py
