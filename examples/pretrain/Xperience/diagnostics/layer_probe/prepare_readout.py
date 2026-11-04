#!/usr/bin/env python3
"""Derive supervised-output-only readout cache and exact-input positive control."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    assert not a.output.exists()
    data=torch.load(a.input,map_location='cpu',mmap=True,weights_only=True)
    batch=torch.load(a.cache,map_location='cpu',mmap=True,weights_only=True)
    meta=data['metadata']; idx=meta['times'].index(.5)
    actions=batch['actions'].repeat(2,1,1).to(device=a.device,dtype=torch.bfloat16)
    mask=batch['action_mask'].repeat(2,1,1).to(device=a.device,dtype=torch.bfloat16)
    valid=torch.tensor(meta['valid_indices'],device=a.device)
    clean=actions*mask
    time=torch.full((16,1,1),.5,device=a.device,dtype=torch.bfloat16)
    controls=[]
    for rnd in range(data['target'].shape[0]):
        torch.manual_seed(meta['args']['seed']+idx*100000+rnd)
        noise=torch.randn_like(actions)*mask
        target=-(noise[8:,:,valid].float()-noise[:8,:,valid].float()).reshape(64,48).cpu()
        assert torch.equal(target,data['target'][rnd]), 'Reconstructed seed stream does not match exported target'
        xt=((1-time)*noise+time*clean).float()
        controls.append((xt[8:,:,valid]-xt[:8,:,valid]).reshape(64,48).cpu())
    original=data['features'].pop('decoder_output')
    pred=original[:,:,meta['valid_indices']].contiguous()
    assert torch.equal(pred,data['native_prediction_delta'])
    data['features']['decoder_output_valid48']=pred
    data['features']['delta_xt_valid48']=torch.stack(controls)
    control_error=(-2*data['features']['delta_xt_valid48']-data['target']).square().mean()/data['target'].square().mean()
    data['metadata']['readout_preparation']={'source':str(a.input),'changes':['decoder_output restricted to supervised 48 dimensions','add reconstructed exact delta_xt positive control'],
        'target_reconstruction_exact':True,'native_output_selection_exact':True,'oracle_minus2_delta_xt_relative_mse':control_error.item()}
    torch.save(data,a.output)
    a.output.with_suffix('.json').write_text(json.dumps(data['metadata']['readout_preparation'],indent=2))
    print(json.dumps(data['metadata']['readout_preparation']))

if __name__=='__main__': main()
