#!/usr/bin/env python3
"""Freeze the original pretrained Qwen VLM and cache real observation features."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--batch', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--base-vlm', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    sys.path.insert(0, a.repo)
    os.chdir(a.repo)
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
    torch.set_num_threads(4)
    torch.manual_seed(42)
    output = Path(a.output)
    assert not output.exists(), 'Refusing to overwrite an existing feature cache'
    batch = torch.load(a.batch, map_location='cpu', weights_only=False)
    cfg = OmegaConf.load(a.config)
    cfg.framework.qwenvl.base_vlm = a.base_vlm
    model = _QWen3_VL_Interface(cfg).to('cuda').eval().requires_grad_(False)
    n_layers = model.model.config.text_config.num_hidden_layers
    hidden_dim = model.model.config.text_config.hidden_size
    assert n_layers == 36 and hidden_dim == 2560
    inputs = model.build_qwenvl_inputs(
        images=[[Image.fromarray(x.numpy())] for x in batch['images']],
        instructions=batch['instructions'],
    )
    with torch.no_grad():
        result = model(**inputs, output_hidden_states=True,
                       output_attentions=False, return_dict=True, use_cache=False)
    features = [h.detach().cpu().contiguous() for h in result.hidden_states[-n_layers:]]
    assert all(torch.isfinite(h).all() for h in features)
    batch['vl_embs_list'] = features
    batch['attention_mask'] = inputs['attention_mask'].detach().cpu().bool()
    batch['metadata'] = {'base_vlm': a.base_vlm, 'base_vlm_frozen': True,
        'vlm_mode': 'eval', 'vlm_checkpoint_loaded': False,
        'hidden_state_selection': 'last 36, H1..H36; same as QwenPI',
        'source_batch': a.batch, 'config': a.config,
        'dtype': str(features[0].dtype),
        'feature_shapes': [list(h.shape) for h in features]}
    torch.save(batch, output)
    h = hashlib.sha256()
    with output.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    report = {**batch['metadata'], 'sha256': h.hexdigest(),
        'action_shape': list(batch['actions'].shape),
        'feature_rms_by_layer': [float(x.float().square().mean().sqrt()) for x in features],
        'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    del model, inputs, result
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
