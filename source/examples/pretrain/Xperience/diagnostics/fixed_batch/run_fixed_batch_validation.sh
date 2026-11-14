#!/usr/bin/env bash
# Run on neo-bj1 beside data_prepare.py, cache_vlm.py, and train_ab.py.
# Fixed 8-sample diagnostic: fresh action heads, frozen original Qwen features.
# REPO must already contain the decoder_input_norm patch; this script does not apply it.
# Example: GPU=0 RUN_LOW_LR=1 bash run_fixed_batch_validation.sh
# RUN_LOW_LR=1 also runs the matching 1e-4 / 2000-step control before 3e-5.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-/mnt/zhaoshijie/workspace/starVLA}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/CRA/bin/python}"
SOURCE_CONFIG="${SOURCE_CONFIG:-/mnt/zhaoshijie/checkpoints/starVLA/xperience_hours100_qwen3vl_pi_gr00t_80k/config.full.yaml}"
BASE_VLM="${BASE_VLM:-$REPO/playground/Pretrained_models/Qwen3-VL-4B-Instruct}"
RUN_ROOT="${RUN_ROOT:-/mnt/zhaoshijie/checkpoints/starVLA}"
GPU="${GPU:-0}"
CACHE_GPU="${CACHE_GPU:-$GPU}"
BASELINE_GPU="${BASELINE_GPU:-$GPU}"
LN_GPU="${LN_GPU:-$GPU}"
LOW_LR_GPU="${LOW_LR_GPU:-$GPU}"
RUN_EXTENDED="${RUN_EXTENDED:-0}"
RUN_LOW_LR="${RUN_LOW_LR:-0}"

for flag in "$RUN_EXTENDED" "$RUN_LOW_LR"; do
    [[ "$flag" == 0 || "$flag" == 1 ]] || { echo "RUN_EXTENDED and RUN_LOW_LR must be 0 or 1" >&2; exit 2; }
done
[[ -d "$REPO" && -d "$BASE_VLM" && -f "$SOURCE_CONFIG" ]] || {
    echo "Check REPO, BASE_VLM, and SOURCE_CONFIG (use absolute paths)." >&2; exit 2;
}
command -v "$PYTHON_BIN" >/dev/null || { echo "Python not found: $PYTHON_BIN" >&2; exit 2; }
for script in data_prepare.py cache_vlm.py train_ab.py; do
    [[ -f "$SCRIPT_DIR/$script" ]] || { echo "Missing $SCRIPT_DIR/$script" >&2; exit 2; }
done

mkdir -p -- "$RUN_ROOT"
RUN_ROOT="$(cd -- "$RUN_ROOT" && pwd)"
RUN_DIR="$RUN_ROOT/xperience_fixed_batch_validation_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_$$"
mkdir -- "$RUN_DIR"  # Refuse to reuse an existing result directory.
mkdir -- "$RUN_DIR/scripts"
cp -- "$SOURCE_CONFIG" "$RUN_DIR/source_config.full.yaml"
cp -- "$SCRIPT_DIR"/{data_prepare.py,cache_vlm.py,train_ab.py,run_fixed_batch_validation.sh} "$RUN_DIR/scripts/"
export PYTHONDONTWRITEBYTECODE=1 NO_ALBUMENTATIONS_UPDATE=1
cd -- "$REPO"
printf 'Output directory: %s\n' "$RUN_DIR"

"$PYTHON_BIN" -u "$RUN_DIR/scripts/data_prepare.py" \
    --repo "$REPO" --config "$RUN_DIR/source_config.full.yaml" \
    --output "$RUN_DIR/data" --count 8 --seed 20261104 \
    2>&1 | tee "$RUN_DIR/data_prepare.log"

CUDA_VISIBLE_DEVICES="$CACHE_GPU" "$PYTHON_BIN" -u "$RUN_DIR/scripts/cache_vlm.py" \
    --repo "$REPO" --config "$RUN_DIR/source_config.full.yaml" \
    --batch "$RUN_DIR/data/batch.pt" --base-vlm "$BASE_VLM" \
    --output "$RUN_DIR/features.pt" 2>&1 | tee "$RUN_DIR/cache_vlm.log"

# FP32 master parameters + BF16 autocast + block checkpointing are harness defaults.
# Each process restarts from seed 42; no arm loads another arm's checkpoint.
COMMON=(--repo "$REPO" --config "$RUN_DIR/source_config.full.yaml"
    --cache "$RUN_DIR/features.pt" --device cuda:0 --batch-size 8 --repeat 2
    --seed 42 --warmup 100 --schedule constant --clip 1 --weight-decay 1e-8
    --threads 4 --dead-patience 20)
run_arm() {
    local name="$1" variant="$2" gpu="$3" steps="$4" lr="$5" eval_every="$6"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$RUN_DIR/scripts/train_ab.py" \
        "${COMMON[@]}" --variant "$variant" --output-dir "$RUN_DIR/$name" \
        --steps "$steps" --lr "$lr" --eval-every "$eval_every" \
        2>&1 | tee "$RUN_DIR/$name.log"
}

# Sequential execution; baseline may stop early after 20 consecutive dead steps.
# LN arms save model weights, without optimizer state; baseline does not save weights.
run_arm baseline baseline "$BASELINE_GPU" 600 1e-4 50
run_arm ln ln "$LN_GPU" 600 1e-4 50
if [[ "$RUN_EXTENDED" == 1 || "$RUN_LOW_LR" == 1 ]]; then
    run_arm ln_2000 ln "$LN_GPU" 2000 1e-4 100
fi
if [[ "$RUN_LOW_LR" == 1 ]]; then
    run_arm ln_lr3e5_2000 ln "$LOW_LR_GPU" 2000 3e-5 100
fi
printf 'Finished. Metrics, summaries, and provenance: %s\n' "$RUN_DIR"
