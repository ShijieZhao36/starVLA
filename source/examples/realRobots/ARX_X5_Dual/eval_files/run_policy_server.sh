#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
star_vla_python="${star_vla_python:-python}"
your_ckpt="${your_ckpt:-outputs/starvla/starvla_pi05_arx_x5_dual/checkpoints/steps_30000_pytorch_model.pt}"
gpu_id="${gpu_id:-0}"
port="${port:-5694}"
USE_BF16="${USE_BF16:-1}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

CMD=(
  "${star_vla_python}" deployment/model_server/server_policy.py
  --ckpt_path "${your_ckpt}"
  --port "${port}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${gpu_id}" "${CMD[@]}"
