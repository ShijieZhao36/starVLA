#!/usr/bin/env python3
"""Keep original train/validation rounds and replace test with unseen noise."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['reference','fresh','output']: p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    import torch
    torch.set_num_threads(4)
    assert not a.output.exists()
    old=torch.load(a.reference,map_location='cpu',mmap=True,weights_only=True)
    new=torch.load(a.fresh,map_location='cpu',mmap=True,weights_only=True)
    assert old['target'].shape[0]==96 and new['target'].shape[0]==16
    assert set(old['features'])==set(new['features'])
    assert old['metadata']['args']['checkpoint']==new['metadata']['args']['checkpoint']
    assert old['metadata']['valid_indices']==new['metadata']['valid_indices']
    def seeds(d, count):
        meta=d['metadata']; base=meta['args']['seed']+meta['times'].index(.5)*100000
        return list(range(base,base+count))
    old_seeds,new_seeds=seeds(old,96),seeds(new,16)
    assert not set(old_seeds).intersection(new_seeds)
    out={'features':{k:torch.cat([v[:80],new['features'][k]],0) for k,v in old['features'].items()},
         'target':torch.cat([old['target'][:80],new['target']],0),
         'native_prediction_delta':torch.cat([old['native_prediction_delta'][:80],new['native_prediction_delta']],0),
         'metadata':{'reference':str(a.reference),'fresh':str(a.fresh),'train_seeds':old_seeds[:64],
           'validation_seeds':old_seeds[64:80],'confirmation_test_seeds':new_seeds,
           'note':'Original 16 test rounds were inspected during exploration; excluded here. Lambda grid extended using original validation only, then confirmed on new noise.'}}
    torch.save(out,a.output)
    a.output.with_suffix('.json').write_text(json.dumps(out['metadata'],indent=2))
    print('saved',str(a.output))

if __name__=='__main__': main()
