"""Validate the new fork with its actual dataloader, weights, and 14-D loss."""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors import safe_open

REPO=Path('/mnt/vla-pretrain/zh1112/workspace/starVLA_pi05_sft35_20261114')
PREP=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO))


def main():
    Accelerator()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    from starVLA.dataloader import build_dataloader
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
    from deployment.model_server.tools.image_tools import resize_with_pad
    from torchcodec.decoders import VideoDecoder
    cfg=OmegaConf.load(PREP/'pi05_socks35_30k.yaml')
    cache_status=json.loads((PREP/'cache_full/SUCCESS.json').read_text())
    for name,digest in cache_status['source_metadata_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
    rows=json.loads((REPO/'examples/realRobots/ARX_X5_Socks35/train_files/data_registry/mix_weights.json').read_text())['datasets']
    fps=set()
    for row in rows:
        info=json.loads((Path(cfg.datasets.vla_data.data_root_dir)/row['alias']/'meta/info.json').read_text())
        fps.add(info['fps'])
    pixel_checks=[]
    for row in [rows[0],rows[len(rows)//2],rows[-1]]:
        data=Path(cfg.datasets.vla_data.data_root_dir)/row['alias']
        info=json.loads((data/'meta/info.json').read_text())
        modalities=json.loads((data/'meta/modality.json').read_text())
        episode=json.loads((data/'meta/episodes.jsonl').read_text().splitlines()[0])
        idx,n=int(episode['episode_index']),int(episode['length'])
        indices=[0,n//2,n-1]
        for camera in ['front','left_wrist','right_wrist']:
            key=modalities['video'][camera]['original_key']
            path=data/info['video_path'].format(episode_chunk=idx//info['chunks_size'],video_key=key,episode_index=idx)
            raw=VideoDecoder(str(path),device='cpu',dimension_order='NHWC',num_ffmpeg_threads=1).get_frames_at(indices=indices).data.numpy()
            reference=resize_with_pad(raw,224,224)
            cached=np.load(Path(cfg.datasets.vla_data.frame_cache_dir)/row['alias']/key/f'episode_{idx:06d}.npy',mmap_mode='r',allow_pickle=False)
            assert np.array_equal(reference,cached[indices])
            pixel_checks.append({'subset':row['subset'],'camera':camera,'episode':idx,'frame_indices':indices,'exact':True})
    cfg.output_dir=str(PREP/'validated_loader')
    Path(cfg.output_dir).mkdir(exist_ok=True)
    cfg.datasets.vla_data.num_workers=0
    cfg.datasets.vla_data.per_device_batch_size=1
    loader=build_dataloader(cfg,'gr00t_sharded')
    batch=next(iter(loader))
    assert len(batch)==1 and batch[0]['action'].shape==(50,14)
    assert batch[0]['state'].shape==(1,14)
    assert [im.size for im in batch[0]['image']]==[(224,224)]*3
    model=build_framework(cfg)
    # Exercise the SAME strict preload path used by the official trainer.
    model=TrainerUtils.load_pretrained_backbones(model,cfg.trainer.pretrained_checkpoint,strict=True)
    with safe_open(cfg.trainer.pretrained_checkpoint,framework='pt',device='cpu') as f:
        source=f.get_tensor('action_in_proj.weight')
        assert torch.equal(model.action_head.action_in_proj.weight,source.to(model.action_head.action_in_proj.weight.dtype))
    model.to('cuda:0').train()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(batch,return_debug=True)
    manual=(out['u_t'][...,:14]-out['v_t'][...,:14]).square().mean()
    assert torch.allclose(manual,out['action_loss'])
    out['v_t'].retain_grad()
    out['action_loss'].backward()
    assert torch.all(out['v_t'].grad[...,14:]==0)
    assert torch.any(out['v_t'].grad[...,:14]!=0)
    grad_norms={}
    for name,module in [('VLM',model.vlm),('Action',model.action_head)]:
        norms=[p.grad.float().norm() for p in module.parameters() if p.grad is not None]
        assert norms and all(bool(torch.isfinite(x)) for x in norms)
        grad_norms[name]=float(torch.stack(norms).norm())
        assert grad_norms[name]>0
    prepared=model._prepare_examples(batch,include_actions=True)
    assert torch.equal(prepared['image'][0,0],torch.from_numpy(np.array(batch[0]['image'][0])).permute(2,0,1).float()/255*2-1)
    result={'passed':True,'repository_commit':'cefee908d71a4c1fb5cc037260a4f3d0dccb8d37',
            'weight_load_strict':True,'weight_values_checked':True,
            'dataloader':'starVLA.dataloader.build_dataloader(gr00t_sharded)',
            'all_35_registered':len(loader.dataset.mixture.datasets)==35,
            'input_images':[list(im.size) for im in batch[0]['image']],
            'state_shape':list(batch[0]['state'].shape),'action_shape':list(batch[0]['action'].shape),
            'loss_14d':float(out['action_loss'].detach()),'loss_matches_first14_mse':True,
            'padding18_output_gradient_exactly_zero':True,'gradient_norms':grad_norms,
            'model_image_processing_does_no_additional_resize':True,
            'source_metadata_unchanged':True,'dataset_fps':sorted(fps),
            'source_to_cache_pixel_checks':pixel_checks}
    (PREP/'mask14_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
