# Integrated experiment launchers

The integration starts from `pretrain_relative` at `b21b424`; the remote backup is
`backup/pretrain_relative-20261111-1227`. The old `pretrain` branch is intentionally
excluded because its action encoding and policy decode have diverged.

Two merge commits retain the original histories: the direct LIBERO 16-GPU 60k
branch (which also contains the 30k branch), followed by the Ego50 A/B branch.
DROID and mixed 16-GPU orchestration are archived in separate commits.
Historical run directories, launch snapshots and checkpoints are not changed.

## Entrypoints

- `Ego50AB/pipeline.py`: A = Galaxea30/LIBERO20/Ego50 and
  B = Galaxea40/LIBERO10/Ego50; formal pre80k -> LIBERO60k -> 10030 evaluations.
  Use an external JSON spec and `--mode smoke` for pre3 -> post3 -> one episode.
- `DROID/train_files/run_pipeline.sh`: DROID pre30k -> LIBERO30k -> 10030 evaluations.
  `SMOKE_TEST=1` selects pre3 -> post3 -> one episode and separate smoke outputs.
- `GalaxeaLiberoMix16GPU/make_spec.py` and `deploy.py`: Galaxea/LIBERO80:20,
  two-host pre80k/global64 -> one-host LIBERO60k -> 10030 evaluations.
  The explicit smoke mode runs two-host pre3 -> one-host post3 -> one episode.
- `LiberoDirect16GPU60K/train_files/run_libero_direct_16gpu_60k_b4.sh`:
  direct LIBERO, 16 GPUs x batch4, global64, 60k updates.
- `LiberoDirect16GPU/train_files/run_libero_direct_16gpu_30k.sh`:
  direct LIBERO, 16 GPUs x batch8, global128, 30k updates.
  Both direct two-host launchers accept `SMOKE_TEST=1` for three updates;
  their shared two-host evaluator smoke covers one episode per GPU (16 total).
- `GalaxeaLiberoMix/train_files/run_libero_direct_60k_eval.sh`: the existing
  one-host direct LIBERO60k baseline, with repository and output-root overrides.

The retained `LiberoDirect16GPU/train_files/reference_eval_then_formal.sh`
and `reuse_training_smoke.py` are historical recovery drivers for the original
November 7 submission. They pin its checkout and reference checkpoints; do not
use them to launch new experiments. Use the direct launchers above with fresh
run IDs and explicit smoke mode for current validation.

Paths above are relative to this `examples/pretrain/` directory. Run launchers
from a committed, clean checkout. Shell entrypoints derive their repository
from their own location or an explicit `REPO_DIR`; resources such as models,
datasets, frame caches, runtime overlay and LIBERO installation remain explicit
cluster defaults. Inspect the family README and dry-run before actual execution.

## Validation and provenance

Use fresh run IDs, launcher/control directories, ports and evaluation outputs.
Smoke checkpoints must never be used as formal sources. Three updates test
plumbing, full-model transfer and numerical health; they do not estimate policy
quality or demonstrate that observed source proportions equal their expectations.

The direct-baseline preflight still pins the historical model, data, evaluation,
runtime, normalization and configuration protocol. Its reviewed shared helper
and mixture declarations are additionally hashed in each validator. Changing
these requires an explicit review and pin update. Run IDs/output roots are not
training hyperparameters and may vary; batch, steps, learning rates, seed,
fresh initialization and checkpoint-loading rules remain validated.

Do not reset historical experiment branches to the integration head. Reproduce
old results at their recorded commit; start future experiments from the merged
`pretrain_relative` and retain their own resolved config and source snapshots.
