"""Generate raw-video caches using StarVLA's actual PI05 resize_with_pad."""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import numpy as np

REPO=Path('/mnt/vla-pretrain/zh1112/workspace/starVLA_pi05_sft35_20261114')
REGISTRY=REPO/'examples/realRobots/ARX_X5_Socks35/train_files/data_registry/mix_weights.json'
DATA=Path('/mnt/vla-pretrain/zh1112/dataset/training_views/socks_sft35_abs_h50_20261112')
CACHE=Path('/mnt/vla-pretrain/zh1112/frame_cache/socks_sft35_224_pi05_pil_20261114')
CAMERAS=('front','left_wrist','right_wrist')


def dump(path,obj):
    partial=path.with_name(path.name+'.tmp')
    partial.write_text(json.dumps(obj,indent=2)+'\n')
    os.replace(partial,path)


def valid(path,length):
    try:
        a=np.load(path,mmap_mode='r',allow_pickle=False)
        return a.dtype==np.uint8 and a.shape==(length,224,224,3) and path.stat().st_size==a.offset+a.nbytes
    except (OSError,ValueError):
        return False


def jobs():
    mix=json.loads(REGISTRY.read_text())
    assert len(mix['datasets'])==35 and mix['horizon']==50
    result=[]
    hashes={}
    for row in mix['datasets']:
        source=DATA/row['alias']
        for name in ['info.json','modality.json','episodes.jsonl']:
            p=source/'meta'/name
            hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        info=json.loads((source/'meta/info.json').read_text())
        modality=json.loads((source/'meta/modality.json').read_text())
        episodes=[json.loads(line) for line in (source/'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
        for ep in episodes:
            idx,n=int(ep['episode_index']),int(ep['length'])
            for camera in CAMERAS:
                key=modality['video'][camera]['original_key']
                video=source/info['video_path'].format(episode_chunk=idx//info['chunks_size'],video_key=key,episode_index=idx)
                if not video.is_file(): raise FileNotFoundError(video)
                result.append((str(video),str(CACHE/row['alias']/key/f'episode_{idx:06d}.npy'),n))
    return result,hashes


def init_worker():
    sys.path.insert(0,str(REPO))
    import torch
    torch.set_num_threads(1)


def one(job):
    from torchcodec.decoders import VideoDecoder
    from deployment.model_server.tools.image_tools import resize_with_pad
    video,out,n=job
    path=Path(out)
    if valid(path,n): return 'skipped',0,n
    decoder=VideoDecoder(video,device='cpu',dimension_order='NHWC',num_ffmpeg_threads=1)
    if int(decoder.metadata.num_frames or 0)<n: raise ValueError(f'Video shorter than metadata: {video}')
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_name(path.name+f'.{os.getpid()}.partial')
    # Buffer an episode in RAM and issue a sequential write. Page-by-page mmap
    # writes stall on GPFS despite abundant CPU and memory on this machine.
    frames=np.empty((n,224,224,3),dtype=np.uint8)
    for start in range(0,n,128):
        stop=min(start+128,n)
        raw=decoder.get_frames_at(indices=list(range(start,stop))).data.numpy()
        frames[start:stop]=resize_with_pad(raw,224,224)
    with partial.open('wb') as handle:
        np.save(handle,frames)
    del frames
    os.replace(partial,path)
    if not valid(path,n): raise ValueError(f'Invalid written cache: {path}')
    return 'written',n*224*224*3,n


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=96)
    p.add_argument('--max-jobs',type=int)
    p.add_argument('--run-dir',type=Path,required=True)
    args=p.parse_args()
    args.run_dir.mkdir(parents=True,exist_ok=True)
    work,hashes=jobs()
    chosen=work if args.max_jobs is None else work[:args.max_jobs]
    state={'phase':'running','pid':os.getpid(),'dataset_count':35,'workers':args.workers,
           'expected_jobs':len(work),'selected_jobs':len(chosen),'expected_camera_frames':sum(j[2] for j in work),
           'expected_pixel_bytes':sum(j[2] for j in work)*224*224*3,'cache_root':str(CACHE),
           'resize_implementation':'deployment.model_server.tools.image_tools.resize_with_pad, PIL bilinear',
           'source_metadata_sha256':hashes,'completed':0,'written':0,'skipped':0,'failed':0,
           'completed_frames':0,'written_bytes':0,'started_at':datetime.now().astimezone().isoformat(),'errors':[]}
    dump(args.run_dir/'status.json',state)
    started=time.monotonic()
    last_dump=started
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        pending={}
        it=iter(chosen)
        def fill():
            while len(pending)<args.workers*2:
                try: job=next(it)
                except StopIteration: return
                pending[pool.submit(one,job)]=job
        fill()
        while pending:
            done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for f in done:
                job=pending.pop(f)
                try:
                    outcome,b,n=f.result()
                    state[outcome]+=1
                    state['completed_frames']+=n
                    state['written_bytes']+=b
                except Exception as e:
                    state['failed']+=1
                    state['errors'].append({'video':job[0],'error':repr(e)})
                state['completed']+=1
            fill()
            now=time.monotonic()
            state['elapsed_seconds']=now-started
            if now-last_dump>=5 or not pending:
                dump(args.run_dir/'status.json',state)
                last_dump=now
    if not state['failed']:
        # Re-read every output header before authorizing the training pipeline.
        invalid=[j[1] for j in chosen if not valid(Path(j[1]),j[2])]
        if invalid: state['failed']+=len(invalid); state['invalid_outputs']=invalid
    state['phase']='failed' if state['failed'] else ('sample_complete' if args.max_jobs else 'complete')
    state['ended_at']=datetime.now().astimezone().isoformat()
    state['elapsed_seconds']=time.monotonic()-started
    dump(args.run_dir/'status.json',state)
    if state['phase']=='complete': dump(args.run_dir/'SUCCESS.json',state)
    print(json.dumps({k:v for k,v in state.items() if k!='source_metadata_sha256'}),flush=True)
    return bool(state['failed'])


if __name__=='__main__':
    raise SystemExit(main())
