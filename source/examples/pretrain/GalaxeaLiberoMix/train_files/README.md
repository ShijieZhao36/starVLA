# Galaxea / LIBERO Plus paired pretraining

Tests replacement of 20% of source samples with target-domain **training** data
at fixed optimizer updates and batch size. This is not an unseen-embodiment test.

| Arm | Host | Pretraining | Initialization and budget |
| --- | --- | --- | --- |
| mixed | neo-bj4 | Galaxea 80%, LIBERO train 20% | Original Qwen3-VL + fresh complete action head, seed42, LN, global64, 80k updates |
| control | neo-bj1 | Galaxea 100% | Same |

The YAML files differ only in mixture, LIBERO data/cache overrides, and run ID.
Both retain the original Galaxea normalization, within-Galaxea subset weights,
action mask and loss reduction. Statistics remain separate by robot tag.

## Comparisons

- Main: these matched LN80k arms, followed by identical LIBERO LN30k fine-tuning
  and the fixed final checkpoint's 10,030-episode evaluation.
- Auxiliary: existing direct LIBERO LN30k run,
  libero_plus_direct_ln_8gpu_30k_20261104_235840. Its final result was unavailable
  at setup time; do not report success until evaluation finishes.
- Historical: Galaxea80k -> LIBERO60k (7,966/10,030) and direct LIBERO60k
  (8,221/10,030). These lacked decoder LN and used older code; they are not
  matched causal controls.

Mixed expects 4,096,000 Galaxea and 1,024,000 LIBERO training slots; control
expects 5,120,000 Galaxea slots. These are sampled slots, not unique examples.
Improvement cannot establish embodiment bias as the cause of the old regression
or distinguish mixing from extra LIBERO exposure. An exposure-matched sequential
control is a separate future experiment. One seed gives preliminary evidence.

Equal slots are not equal FLOPs: sources have 3 and 2 camera views. The loader
samples whole shards, not an exact 80:20 composition in every batch. Local
masked-element loss followed by distributed rank averaging is unchanged;
sample ratios are not gradient ratios.

## Running

Each invocation owns 8 host GPUs and refuses existing GPU compute processes.
Use a fresh RUN_ID; nothing is overwritten or resumed automatically.

    DRY_RUN=1 ARM=mixed bash examples/pretrain/GalaxeaLiberoMix/train_files/run_pretrain_pair.sh
    ARM=mixed SMOKE_TEST=1 bash examples/pretrain/GalaxeaLiberoMix/train_files/run_pretrain_pair.sh
    ARM=mixed bash examples/pretrain/GalaxeaLiberoMix/train_files/run_pretrain_pair.sh
    ARM=control bash examples/pretrain/GalaxeaLiberoMix/train_files/run_pretrain_pair.sh

Default scope is pretraining only. RUN_DOWNSTREAM=1 explicitly chains the
prepared LIBERO30k + final evaluation runner after successful pretraining.
With SMOKE_TEST=1, downstream training is 3 steps and evaluation is one episode.
Smoke weights never initialize formal pretraining.

CRA Python, BF16/ZeRO2, 8 GPUs x batch8, GA1, warmup5k, updates80k, save10k;
action LR1e-4, VLM LR1e-5, base2.5e-5, cosine_with_min_lr1e-6,
AdamW(.9,.95), eps1e-8, wd1e-8, clip1. The scheduler's common final multiplier
is .04, giving action LR4e-6 / VLM LR4e-7. Qwen3 uses SDPA.

## Audit outputs

- initial_model_sha256.json: initial complete model, action head, and VLM hashes.
- data_sampling_manifest.json: paths, expected fractions, initial shard plan.
- metrics.jsonl: all-rank backward sample counts and ratios, excluding extra
  batches fetched for training-time MAE/MSE.
- dataset_statistics.json: statistics actually used by the run.
- launch/: exact config, command, commit, dirty diff and environment versions.
- pipeline_status.txt and exit_code: phase and final exit.
- pretrain_success.json: only written after successful accelerate exit and
  final checkpoint/config/statistics checks.

Checkpoints contain model weights, not full optimizer/scheduler/RNG states,
so they are not exact-resume snapshots.

## Data provenance

LIBERO source and converted metadata declare train=0:14347, 14,347 episodes,
2,238,036 frames, 40 tasks. The conversion report points to
/mnt/zhaoshijie/dataset/libero_plus_lerobot. Configured training roots do not
include evaluation rollouts. This checks local provenance, not independent
upstream episode-identity deduplication.
