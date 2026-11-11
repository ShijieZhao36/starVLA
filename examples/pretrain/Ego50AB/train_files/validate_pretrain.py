#!/usr/bin/env python3
"""Validate completed pretraining before enabling downstream training."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import yaml

EXPECTED_INITIAL = '530663757b9fb95c1f918fffaf8fff7a90b3fae43e078d17f52f2f0716778a89'
TAGS = ['galaxea_r1lite_stage1', 'libero_plus_rot6d', 'xperience_human']
FRACTIONS = {'A': [0.3, 0.2, 0.5], 'B': [0.4, 0.1, 0.5]}

def validate(run, arm, steps):
    run = Path(run)
    cfg = yaml.safe_load((run / 'config.full.yaml').read_text())
    assert cfg['seed'] == 42
    action = cfg['framework']['action_model']
    assert action['decoder_input_norm'] is True
    assert action['action_dim'] == 90 and action['action_horizon'] == 8
    data = cfg['datasets']['vla_data']
    assert data['data_mix'] == f'galaxea_libero_ego50_{arm}'
    assert data['per_device_batch_size'] == 8 and data['include_state'] is False
    trainer = cfg['trainer']
    assert trainer['pretrained_checkpoint'] is None and trainer['is_resume'] is False
    assert trainer['max_train_steps'] == steps and trainer['gradient_accumulation_steps'] == 1
    stats = json.loads((run / 'dataset_statistics.json').read_text())
    assert set(stats) == set(TAGS)
    identity = json.loads((run / 'initial_model_sha256.json').read_text())
    assert identity['parameters']['model']['sha256'] == EXPECTED_INITIAL
    records = [json.loads(s) for s in (run / 'metrics.jsonl').read_text().splitlines() if s.strip()]
    assert records and records[-1]['step'] == steps
    assert records[-1]['source_exposure/cumulative_examples'] == steps * 64
    assert all(math.isfinite(r['action_dit_loss']) for r in records)
    fractions = {tag: records[-1].get(f'source_exposure/{tag}/cumulative_examples', 0) / (steps * 64) for tag in TAGS}
    if steps == 80000:
        assert all(abs(fractions[tag] - expected) < 0.03 for tag, expected in zip(TAGS, FRACTIONS[arm])), fractions
    ckpt = run / 'final_model/pytorch_model.pt'
    assert ckpt.stat().st_size > 1_000_000
    marker = dict(completed_at=datetime.now(timezone.utc).isoformat(), arm=arm, steps=steps,
                  checkpoint=str(ckpt.resolve()), bytes=ckpt.stat().st_size,
                  observed_source_fractions=fractions, configured_source_fractions=dict(zip(TAGS, FRACTIONS[arm])),
                  initial_model_sha256=EXPECTED_INITIAL)
    (run / 'pretrain_success.json').write_text(json.dumps(marker, indent=2) + '\n')
    return marker

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--arm', choices=['A', 'B'], required=True)
    p.add_argument('--steps', type=int, choices=[3, 80000], required=True)
    args = p.parse_args()
    print(json.dumps(validate(args.run, args.arm, args.steps), indent=2))
