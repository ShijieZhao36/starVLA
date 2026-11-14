#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR="${STARVLA_DIR:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
ABot_python="${ABot_python:-python}"
your_ckpt="${your_ckpt:-/mnt/zhaoshijie/models/StarVLA/Qwen3-VL-PI-LIBERO-4in1/checkpoints/steps_100000_pytorch_model.pt}"
base_port="${base_port:-9883}"
gpu_id="${gpu_id:-0}"
USE_BF16="${USE_BF16:-1}"
# This released checkpoint was trained with historical LayerwiseFM. Set true only
# for a checkpoint trained with the canonical interleaved DiT.
USE_CANONICAL_FORWARD="${USE_CANONICAL_FORWARD:-false}"

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

CMD=(
  "${ABot_python}" deployment/model_server/server_policy.py
  --ckpt_path "${your_ckpt}"
  --port "${base_port}"
)

if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi

if [[ "${USE_CANONICAL_FORWARD}" != "true" && "${USE_CANONICAL_FORWARD}" != "false" ]]; then
  echo "USE_CANONICAL_FORWARD must be 'true' or 'false'; got '${USE_CANONICAL_FORWARD}'." >&2
  exit 2
fi
OVERRIDE="framework.action_model.diffusion_model_cfg.use_canonical_forward=${USE_CANONICAL_FORWARD}"
echo "Applying config override: ${OVERRIDE}"
CMD+=(--config_override "${OVERRIDE}")

CUDA_VISIBLE_DEVICES="${gpu_id}" "${CMD[@]}"
