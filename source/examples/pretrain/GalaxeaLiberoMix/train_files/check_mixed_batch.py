"""Exercise two embodiments in ONE real model batch; no checkpoint is reused."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat

config_path, samples_path, result_path = map(Path, sys.argv[1:])
cfg = apply_config_compat(OmegaConf.load(config_path))
with samples_path.open('rb') as f:
    samples = pickle.load(f)
assert len(samples) == 2
assert sorted(len(s['image']) for s in samples) == [2, 3]
assert len({s['robot_tag'] for s in samples}) == 2
for sample in samples:
    assert np.asarray(sample['action']).shape == (8, 90)
    assert np.isfinite(sample['action']).all()
set_seed(42)
model = build_framework(cfg).cuda().train()
with torch.autocast('cuda', dtype=torch.bfloat16):
    loss = model(samples)['action_loss']
assert torch.isfinite(loss), loss
loss.backward()
grads = [(name, p.grad) for name, p in model.named_parameters() if p.grad is not None]
assert grads and all(torch.isfinite(g).all().item() for _, g in grads)
assert any('qwen' in name for name, _ in grads)
assert any('action' in name for name, _ in grads)
loss_value = float(loss.detach())
model.zero_grad(set_to_none=True)
model.eval()
with torch.autocast('cuda', dtype=torch.bfloat16):
    prediction = model.predict_action(examples=samples)['normalized_actions']
assert np.asarray(prediction).shape == (2, 8, 90)
assert np.isfinite(prediction).all()
for sample, pred in zip(samples, prediction):
    assert np.all(pred[np.asarray(sample['action_mask']) == 0] == 0)
result = dict(loss=loss_value, finite_gradients=True, gradient_parameter_count=len(grads),
              tags=[s['robot_tag'] for s in samples], views=[len(s['image']) for s in samples],
              valid_dims=[int(np.asarray(s['action_mask'])[0].sum()) for s in samples],
              prediction_shape=list(np.asarray(prediction).shape), masked_prediction_zero=True,
              peak_gpu_bytes=torch.cuda.max_memory_allocated())
result_path.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
