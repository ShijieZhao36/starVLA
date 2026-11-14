"""Measure all eight GPUs at batch 8, then start a fresh 40k run if safe."""
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess

PREP=Path(__file__).resolve().parent
REPO=Path('/mnt/vla-pretrain/zh1112/workspace/starVLA_pi05_sft35_20261114')
ROOT=Path('/mnt/vla-pretrain/zh1112/checkpoints/starVLA')
PY='/root/miniconda3/envs/starvla_pi05/bin/python'
RUN='pi05_socks35_abs_h50_state14_mask14_pil224_lr5e5_bs8_8gpu_40k_20261114'
SMOKE=RUN+'_memory_smoke20'


def write(name,data):
    target=PREP/name
    tmp=target.with_suffix(target.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n')
    os.replace(tmp,target)


def status(phase,**kwargs):
    record={'phase':phase,'updated_at':datetime.datetime.now().astimezone().isoformat(),
            'controller_pid':os.getpid(),'run_id':RUN,**kwargs}
    write('status.json',record)
    print(json.dumps(record),flush=True)


def command(run,port):
    return [PY,'-m','accelerate.commands.launch',
            '--config_file',str(REPO/'starVLA/config/deepseeds/deepspeed_zero2.yaml'),
            '--num_processes','8','--main_process_port',str(port),
            '--deepspeed_config_file',str(REPO/'starVLA/config/deepseeds/ds_config.yaml'),
            str(REPO/'starVLA/training/train_starvla.py'),
            '--config_yaml',str(PREP/'train_40k_bs8.yaml'),
            '--run_root_dir',str(ROOT),'--run_id',run]


def main():
    # Keep the descriptor open for this controller's entire lifetime.
    lock=(PREP/'controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy()
    env.update(PYTHONPATH=str(REPO),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               WANDB_MODE='disabled',PYTHONUNBUFFERED='1',NCCL_DEBUG='WARN',
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if busy: raise RuntimeError('GPUs are occupied: '+busy)
    status('testing_batch8',test_steps=20,per_gpu_batch=8,global_batch=64)
    test=command(SMOKE,29861)+['--trainer.max_train_steps','20','--trainer.num_warmup_steps','0',
                             '--trainer.logging_frequency','1']
    write('smoke_command.json',test)
    with (PREP/'smoke.log').open('a') as log:
        subprocess.run(test,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    smoke_dir=ROOT/SMOKE
    memory=[]
    for rank in range(8):
        item=json.loads((smoke_dir/f'cuda_memory_rank{rank}.json').read_text())
        assert item['step']==20
        external=max(0,item['device_total_bytes']-item['device_free_bytes']-item['reserved_bytes'])
        item['estimated_peak_device_used_bytes']=item['peak_reserved_bytes']+external
        item['estimated_headroom_bytes']=item['device_total_bytes']-item['estimated_peak_device_used_bytes']
        memory.append(item)
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(smoke_dir/'tensorboard')).Reload()
    loss=events.Scalars('loss/action_fm')
    assert len(loss)==20 and loss[-1].step==20 and all(math.isfinite(r.value) for r in loss)
    assert events.Scalars('source_exposure/cumulative_examples')[-1].value==1280
    assert (smoke_dir/'final_model/pytorch_model.pt').is_file()
    minimum_margin=min(item['estimated_headroom_bytes'] for item in memory)
    safe=minimum_margin>=8*2**30
    result={'safe':safe,'test_steps':20,'per_device_batch':8,'global_batch':64,
            'minimum_estimated_headroom_gib':minimum_margin/2**30,
            'criterion':'all ranks completed training and save, finite losses; >=8 GiB headroom',
            'maximum_peak_allocated_gib':max(m['peak_allocated_bytes'] for m in memory)/2**30,
            'maximum_peak_reserved_gib':max(m['peak_reserved_bytes'] for m in memory)/2**30,
            'last_test_loss':loss[-1].value,'ranks':memory}
    write('batch8_safety.json',result)
    if not safe:
        status('batch8_headroom_insufficient',safety=result)
        return
    status('batch8_safe',safety=result)
    destination=ROOT/RUN
    if destination.exists(): raise RuntimeError('Formal run directory already exists: '+str(destination))
    destination.mkdir(parents=True)
    cmd=command(RUN,29862)
    write('formal_command.json',cmd)
    with (destination/'train.log').open('a') as log:
        proc=subprocess.Popen(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status('formal_launched',training_pid=proc.pid,train_log=str(destination/'train.log'),
           tensorboard=str(destination/'tensorboard'),from_original_pi05_base=True)


if __name__=='__main__':
    try: main()
    except Exception as exc:
        status('failed',error=repr(exc))
        raise
