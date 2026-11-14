#!/usr/bin/env bash
# DROID CPU frame cache using the existing Galaxea/Xperience worker implementation.
# One independent job per episode x camera; each decoder and numerical library
# has one thread. Training num_workers is unrelated to CACHE_WORKERS.
# Output: FRAME_CACHE_DIR/dataset/<original_video_key>/episode_XXXXXX.npy
# DRY_RUN=1 previews metadata, job count and storage without starting workers.
set -Eeuo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
positive_int() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die "$1 must be a positive integer"; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/CRA/bin/python}"
DROID_DATASET_DIR="${DROID_DATASET_DIR:-/mnt/zhaoshijie/dataset/oxe_franka_lerobot/droid_1.0.1/stage3_unmasked_20261021_142915/job/full/dataset}"
DRY_RUN="${DRY_RUN:-1}"
CACHE_SIZE="${CACHE_SIZE:-256}"
CACHE_CHUNK="${CACHE_CHUNK:-32}"
CACHE_WORKERS="${CACHE_WORKERS:-160}"
for name in CACHE_SIZE CACHE_CHUNK; do positive_int "$name" "${!name}"; done
[[ "$DRY_RUN" =~ ^[01]$ ]] || die 'DRY_RUN must be 0 or 1'
[[ -x "$PYTHON_BIN" ]] || die "Missing Python interpreter: $PYTHON_BIN"
[[ -d "$DROID_DATASET_DIR/meta" ]] || die "Missing dataset metadata: $DROID_DATASET_DIR/meta"
DROID_DATASET_DIR="$(readlink -m "$DROID_DATASET_DIR")"
[[ "$(basename "$DROID_DATASET_DIR")" == dataset ]] || die 'The registered mixture expects a directory named dataset; select its parent with the training data_root_dir'
DATA_ROOT="$(dirname "$DROID_DATASET_DIR")"
DATA_MIX=droid_stage3_unmasked
CACHE_CAMERAS="${CACHE_CAMERAS:-front,exterior_2,wrist}"
FRAME_CACHE_DIR="${FRAME_CACHE_DIR:-/mnt/zhaoshijie/frame_cache/droid_stage3_unmasked_256}"
[[ "$FRAME_CACHE_DIR" == /* ]] || die 'FRAME_CACHE_DIR must be absolute'
[[ -f "$REPO_DIR/starVLA/dataloader/precompute_galaxea_frame_cache.py" ]] || die 'Missing shared frame-cache helper'
cd "$REPO_DIR"
export DATA_ROOT DATA_MIX FRAME_CACHE_DIR CACHE_CAMERAS CACHE_SIZE CACHE_CHUNK
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false NO_ALBUMENTATIONS_UPDATE=1

available_cpus="$("$PYTHON_BIN" -c 'import os; print(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)')"
if [[ -n "${CACHE_WORKERS:-}" ]]; then
  positive_int CACHE_WORKERS "$CACHE_WORKERS"
  (( CACHE_WORKERS <= available_cpus )) || die "CACHE_WORKERS=$CACHE_WORKERS exceeds CPU affinity ($available_cpus)"
  export CACHE_WORKERS
elif [[ "$DRY_RUN" == 0 ]]; then
  die 'Set CACHE_WORKERS explicitly before decoding the frame cache'
fi
cache_plan_json="$("$PYTHON_BIN" - "$DROID_DATASET_DIR" "$FRAME_CACHE_DIR" "$CACHE_CAMERAS" "$CACHE_SIZE" "${CACHE_WORKERS:-}" "$available_cpus" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, cache = map(Path, sys.argv[1:3])
names = sys.argv[3].split(',')
if not names or any(name not in {'front', 'exterior_2', 'wrist'} for name in names) or len(set(names)) != len(names):
    raise SystemExit('CACHE_CAMERAS must be a comma-separated subset of front,exterior_2,wrist without duplicates or whitespace')
info = json.loads((root / 'meta/info.json').read_text())
modality = json.loads((root / 'meta/modality.json').read_text())
for name in names:
    if name not in modality.get('video', {}):
        raise SystemExit(f'Missing camera in meta/modality.json: {name}')
rows = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
ids = [row['episode_index'] for row in rows]
if not rows or len(set(ids)) != len(ids) or any(row['length'] <= 0 for row in rows):
    raise SystemExit('Invalid episode index/length metadata')
frames = sum(row['length'] for row in rows)
if info['total_episodes'] != len(rows) or info['total_frames'] != frames:
    raise SystemExit('info.json episode/frame totals disagree with episodes.jsonl')
size = int(sys.argv[4])
print(json.dumps({'dataset': str(root), 'cache': str(cache), 'cameras': names,
                  'episodes': len(rows), 'frames': frames, 'job_count': len(rows) * len(names), 'size': size,
                  'info_sha256': hashlib.sha256((root / 'meta/info.json').read_bytes()).hexdigest(),
                  'episodes_sha256': hashlib.sha256((root / 'meta/episodes.jsonl').read_bytes()).hexdigest(),
                  'modality_sha256': hashlib.sha256((root / 'meta/modality.json').read_bytes()).hexdigest(),
                  'shape_per_frame': [size, size, 3], 'dtype': 'uint8', 'resize': 'cv2.INTER_AREA to square',
                  'uncompressed_payload_bytes': frames * len(names) * size * size * 3,
                  'maximum_episode_worker_array_bytes': max(row['length'] for row in rows) * size * size * 3,
                  'workers': int(sys.argv[5]) if sys.argv[5] else None,
                  'available_cpus': int(sys.argv[6]), 'decoder_threads_per_worker': 1}, indent=2))
PY
)"
printf '%s\n' "$cache_plan_json"
printf 'Command: %q -m starVLA.dataloader.precompute_galaxea_frame_cache\n' "$PYTHON_BIN"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'Dry run complete; no frame-cache directory or worker process was created.\n'
  exit 0
fi
# The shared helper writes .partial then renames; hold one output-root lock to
# prevent concurrent runs from writing the same episode-camera partial file.
command -v flock >/dev/null || die 'flock is required'
mkdir -p "$FRAME_CACHE_DIR"
exec 9>"$FRAME_CACHE_DIR/.droid_frame_cache.lock"
flock -n 9 || die "Another DROID frame-cache writer holds $FRAME_CACHE_DIR"
# Pin the source/shape contract before writing. A resume may reuse completed
# arrays only when the previous contract agrees with this run.
"$PYTHON_BIN" - "$FRAME_CACHE_DIR" "$cache_plan_json" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
root = Path(sys.argv[1])
plan = json.loads(sys.argv[2])
identity = ('dataset', 'cameras', 'size', 'episodes', 'frames', 'job_count', 'info_sha256', 'episodes_sha256', 'modality_sha256')
for filename in ('droid_cache_plan.json', 'droid_cache_complete.json'):
    path = root / filename
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(key) != plan[key] for key in identity):
            raise SystemExit(f'{path} describes different data/cameras/size; select a new FRAME_CACHE_DIR')
(root / 'droid_cache_complete.json').unlink(missing_ok=True)
fd, temporary = tempfile.mkstemp(prefix='.droid-cache-plan-', suffix='.json', dir=root)
with os.fdopen(fd, 'w') as handle:
    json.dump(plan, handle, indent=2)
    handle.write('\n')
os.replace(temporary, root / 'droid_cache_plan.json')
PY
"$PYTHON_BIN" -m starVLA.dataloader.precompute_galaxea_frame_cache
# A worker error leaves the plan but never a completion marker. Metadata is
# checked again after decoding, so a concurrent source change also fails.
"$PYTHON_BIN" - "$FRAME_CACHE_DIR" "$cache_plan_json" <<'PY'
import hashlib, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
root = Path(sys.argv[1])
plan = json.loads(sys.argv[2])
dataset = Path(plan['dataset'])
for name, key in (('info.json', 'info_sha256'), ('episodes.jsonl', 'episodes_sha256'), ('modality.json', 'modality_sha256')):
    current = hashlib.sha256((dataset / 'meta' / name).read_bytes()).hexdigest()
    if current != plan[key]:
        raise SystemExit(f'Dataset metadata changed during cache generation: {name}')
plan.update(validated=True, completed_at=datetime.now(timezone.utc).isoformat())
fd, temporary = tempfile.mkstemp(prefix='.droid-cache-complete-', suffix='.json', dir=root)
with os.fdopen(fd, 'w') as handle:
    json.dump(plan, handle, indent=2)
    handle.write('\n')
os.replace(temporary, root / 'droid_cache_complete.json')
print(f"Frame cache complete: {root / 'droid_cache_complete.json'} ({plan['job_count']} episode-camera files)")
PY
