#!/usr/bin/env bash
# Reuse verified three-step training; independently test eval, then train fresh.
set -Eeuo pipefail
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
: "${NODE_RANK:?Set NODE_RANK=0 or 1}"
: "${RUN_ID:?Set a new unique formal run ID}"
: "${LAUNCHER_DIR:?Set a new shared launcher directory}"
: "${REFERENCE_EVAL_DIR:?Set a new shared diagnostic eval directory}"
: "${EXPECTED_COMMIT:?Set the pinned clean checkout commit}"
DRY_RUN="${DRY_RUN:-1}"
[[ "$NODE_RANK" =~ ^[01]$ && "$DRY_RUN" =~ ^[01]$ ]] || die 'Invalid rank or DRY_RUN'
[[ "$RUN_ID" =~ ^libero_plus_direct_ln_16gpu_30k_[A-Za-z0-9_.-]+$ && "$RUN_ID" != *_smoke3 ]] || die 'Invalid new formal run ID'
[[ "$RUN_ID" != libero_plus_direct_ln_16gpu_30k_20261107_184512 ]] || die 'Never reuse the failed submission ID'
[[ "$LAUNCHER_DIR" == "/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/$RUN_ID" ]] || die 'Launcher must belong to the new run'
[[ "$REFERENCE_EVAL_DIR" == "/mnt/zhaoshijie/eval/starVLA/libero_plus/${RUN_ID}_reference_smoke64" ]] || die 'Use the dedicated reference_smoke64 directory'
REPO_DIR=/mnt/zhaoshijie/workspace/starVLA_libero_direct_ln_16gpu_30k_20261107
PYTHON_BIN=/root/miniconda3/envs/MRA/bin/python
TRAIN_FILES="$REPO_DIR/examples/pretrain/LiberoDirect16GPU/train_files"
RUNNER="$TRAIN_FILES/run_libero_direct_16gpu_30k.sh"
REUSE="$TRAIN_FILES/reuse_training_smoke.py"
COORDINATOR="$TRAIN_FILES/coordinate_nodes.py"
EVALUATOR="$TRAIN_FILES/../eval_files/run_libero_plus_16gpu.py"
REFERENCE_CHECKPOINT=/mnt/zhaoshijie/checkpoints/starVLA/libero_from_galaxea_control_ln_8gpu_60k_20261105_pair/final_model/pytorch_model.pt
FORMAL_OUTPUT="/mnt/zhaoshijie/checkpoints/starVLA/$RUN_ID"
STAGING="$LAUNCHER_DIR/reference_eval_stage"
export NODE_RANK RUN_ID LAUNCHER_DIR
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="/mnt/zhaoshijie/checkpoints/starVLA/launcher_logs/galaxea_libero_pair_validation_20261105/sim_overlay:$REPO_DIR:/root/workspace/LIBERO-plus"
export LIBERO_HOME=/root/workspace/LIBERO-plus LIBERO_CONFIG_PATH=/root/workspace/LIBERO-plus/libero
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset MUJOCO_EGL_DEVICE_ID
cd "$REPO_DIR"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" && -z "$(git status --porcelain --untracked-files=all)" ]] || die 'Checkout differs from pinned clean commit'
eval_args=(--repo "$REPO_DIR" --node-rank "$NODE_RANK" --pretrained-path "$REFERENCE_CHECKPOINT" --output-dir "$REFERENCE_EVAL_DIR" --smoke --smoke-sims-per-gpu 4 --renderer-map 0,1,2,3,4,5,6,7,0,1,2,3,5,5,6,7)
coord_args=(--run "$STAGING" --launcher "$LAUNCHER_DIR" --node-rank "$NODE_RANK")
if [[ "$DRY_RUN" == 1 ]]; then
 CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$REUSE" training --repo "$REPO_DIR"
 DRY_RUN=1 SMOKE_TEST=0 bash "$RUNNER"
 CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$EVALUATOR" "${eval_args[@]}" --dry-run
 printf 'Dry run passed; no GPU workload, output directory, or screen claim created.\n'
 exit 0
fi
mkdir -p "$LAUNCHER_DIR/node$NODE_RANK"
mkdir "$LAUNCHER_DIR/node$NODE_RANK/screen_claim" || die 'Node screen already submitted; no retry'
exec >>"$LAUNCHER_DIR/node$NODE_RANK/pipeline.log" 2>&1
phase=validating_reused_training
finish() {
 rc=$?
 printf '%s\n' "$rc" > "$LAUNCHER_DIR/node$NODE_RANK/exit_code"
 printf 'time=%s phase=%s exit_code=%s\n' "$(TZ=Asia/Shanghai date --iso-8601=seconds)" "$phase" "$rc" >> "$LAUNCHER_DIR/node$NODE_RANK/pipeline_status.txt"
}
trap finish EXIT
date --iso-8601=seconds > "$LAUNCHER_DIR/node$NODE_RANK/started_at"
[[ ! -e "$FORMAL_OUTPUT" && ! -e "${FORMAL_OUTPUT}_smoke3" && ! -e "/mnt/zhaoshijie/eval/starVLA/libero_plus/${RUN_ID}_final16" ]] || die 'Formal output already exists'
[[ ! -e "$REFERENCE_EVAL_DIR/node$NODE_RANK" && ! -e "$REFERENCE_EVAL_DIR/eval_success.json" ]] || die 'Reference eval already claimed/completed'
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$REUSE" training --repo "$REPO_DIR" --report "$LAUNCHER_DIR/node$NODE_RANK/reused_training_smoke.json"
DRY_RUN=1 SMOKE_TEST=0 CPU_PREFLIGHT_REPORT="$LAUNCHER_DIR/node$NODE_RANK/formal_cpu_preflight.json" bash "$RUNNER"
mkdir -p "$STAGING"
"$PYTHON_BIN" - "$LAUNCHER_DIR/node$NODE_RANK/reference_purpose.json" "$REFERENCE_CHECKPOINT" "$REFERENCE_EVAL_DIR" <<'PYPURPOSE'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); t=p.with_suffix('.tmp')
t.write_text(json.dumps({'diagnostic_only':True,'source_checkpoint':sys.argv[2],'output':sys.argv[3],
 'purpose':'16 policy GPUs x 4 simulators infrastructure smoke; not training initialization or formal results',
 'reused_training_smoke':'libero_plus_direct_ln_16gpu_30k_20261107_184512_smoke3'})+'\n');t.replace(p)
PYPURPOSE
phase=reference_evaluation
(
 exec 9>"/tmp/starvla_mix_pair_8gpu_$(id -u).lock"; flock -n 9 || die 'Host pipeline lock occupied'
 exec 8>"/tmp/starvla_libero_ln_8gpu_$(id -u).lock"; flock -n 8 || die 'LIBERO host lock occupied'
 "$PYTHON_BIN" - <<'PYGPU'
import subprocess,xml.etree.ElementTree as E
cards=E.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],timeout=20)).findall('gpu')
assert len(cards)==8
for g in cards:
 assert (g.findtext('gpu_recovery_action'),g.findtext('remapped_rows/remapped_row_pending'),g.findtext('remapped_rows/remapped_row_failure'))==('None','No','No')
 for key in ['dram_uncorrectable','sram_uncorrectable_parity','sram_uncorrectable_secded']:
  assert g.findtext('ecc_errors/volatile/'+key)=='0'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],timeout=20).strip()
PYGPU
 "$PYTHON_BIN" "$COORDINATOR" run "${coord_args[@]}" --timeout 21600 --log "$LAUNCHER_DIR/node$NODE_RANK/reference_eval_driver.log" -- "$PYTHON_BIN" "$EVALUATOR" "${eval_args[@]}"
)
printf '0\n' > "$LAUNCHER_DIR/node$NODE_RANK/reference_eval_exit_code.tmp"
mv "$LAUNCHER_DIR/node$NODE_RANK/reference_eval_exit_code.tmp" "$LAUNCHER_DIR/node$NODE_RANK/reference_eval_exit_code"
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 600 --path "$LAUNCHER_DIR/node0/reference_eval_exit_code" --path "$LAUNCHER_DIR/node1/reference_eval_exit_code"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" "$REUSE" reference-eval --repo "$REPO_DIR" --reference-eval-dir "$REFERENCE_EVAL_DIR" --launcher "$LAUNCHER_DIR" --report "$LAUNCHER_DIR/node$NODE_RANK/reference_eval_acceptance.json"
"$PYTHON_BIN" "$COORDINATOR" wait "${coord_args[@]}" --timeout 600 --path "$LAUNCHER_DIR/node0/reference_eval_acceptance.json" --path "$LAUNCHER_DIR/node1/reference_eval_acceptance.json"
phase=formal
# The unchanged runner rechecks clean code, both protocols, idle hardware and
# fresh output ownership; reference weights never enter its training arguments.
DRY_RUN=0 SMOKE_TEST=0 CPU_PREFLIGHT_REPORT="$LAUNCHER_DIR/node$NODE_RANK/formal_cpu_preflight.json" bash "$RUNNER"
phase=completed
