"""Wait for verified raw->224 caches, validate, smoke, then start the approved run."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time

REPO=Path('/mnt/vla-pretrain/zh1112/workspace/starVLA_pi05_sft35_20261114')
PREP=Path(__file__).resolve().parent
PYTHON='/root/miniconda3/envs/starvla_pi05/bin/python'
RUN='pi05_socks35_abs_h50_state14_mask14_pil224_lr5e5_8gpu_30k_20261114'
ROOT=Path('/mnt/vla-pretrain/zh1112/checkpoints/starVLA')


def status(phase,**kwargs):
    d={'phase':phase,'updated_at':datetime.now().astimezone().isoformat(),'pipeline_pid':os.getpid(),
       'run_id':RUN,'repository':str(REPO),'environment':'starvla_pi05',**kwargs}
    tmp=PREP/'pipeline_status.json.tmp'
    tmp.write_text(json.dumps(d,indent=2)+'\n')
    os.replace(tmp,PREP/'pipeline_status.json')
    print(json.dumps(d),flush=True)


def cmd(run,port):
    return [PYTHON,'-m','accelerate.commands.launch','--config_file',str(REPO/'starVLA/config/deepseeds/deepspeed_zero2.yaml'),
            '--num_processes','8','--main_process_port',str(port),
            '--deepspeed_config_file',str(REPO/'starVLA/config/deepseeds/ds_config.yaml'),
            str(REPO/'starVLA/training/train_starvla.py'),'--config_yaml',str(PREP/'pi05_socks35_30k.yaml'),
            '--run_root_dir',str(ROOT),'--run_id',run]


def run_checked(command,log,env):
    with log.open('a') as f:
        subprocess.run(command,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)


def main():
    env=os.environ.copy()
    env.update(PYTHONPATH=str(REPO),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',PYTHONUNBUFFERED='1',
               NCCL_DEBUG='WARN',TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    status('waiting_cache')
    while True:
        p=PREP/'cache_full/status.json'
        if p.exists():
            d=json.loads(p.read_text())
            if d['phase']=='failed': raise RuntimeError('Raw224 cache failed: '+str(d.get('errors')))
            if d['phase']=='complete':
                assert d['failed']==0 and d['completed']==d['expected_jobs']
                assert (PREP/'cache_full/SUCCESS.json').exists()
                break
        time.sleep(10)
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active: raise RuntimeError('GPUs occupied; refusing to affect another job: '+active)
    status('validating_mask14_and_strict_preload')
    env_single={**env,'CUDA_VISIBLE_DEVICES':'0'}
    run_checked([PYTHON,str(PREP/'validate_mask14.py')],PREP/'mask14_validation.log',env_single)
    valid=json.loads((PREP/'mask14_validation.json').read_text())
    assert valid['passed'] and valid['all_35_registered']
    smoke=RUN+'_smoke2'
    status('running_8gpu_smoke',smoke_run=smoke)
    smoke_command=cmd(smoke,29851)+['--trainer.max_train_steps','2','--trainer.num_warmup_steps','0',
                                  '--trainer.logging_frequency','1','--trainer.eval_interval','2',
                                  '--datasets.vla_data.num_workers','0']
    run_checked(smoke_command,PREP/'smoke8.log',env)
    assert (ROOT/smoke/'final_model/pytorch_model.pt').is_file()
    # Read actual events, not merely the event filename, before formal launch.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(ROOT/smoke/'tensorboard')).Reload()
    for tag in ['loss/action_fm','loss/total','learning_rate/vlm','learning_rate/action_head']:
        rows=events.Scalars(tag)
        assert len(rows)>=2 and rows[-1].step==2
    status('smoke_passed')
    out=ROOT/RUN
    if out.exists(): raise RuntimeError('Formal run already exists; refusing an accidental duplicate: '+str(out))
    out.mkdir(parents=True)
    formal=cmd(RUN,29852)
    (PREP/'formal_command.json').write_text(json.dumps(formal,indent=2)+'\n')
    with (out/'train.log').open('a') as f:
        p=subprocess.Popen(formal,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    status('formal_launched',training_pid=p.pid,train_log=str(out/'train.log'),
           tensorboard=str(out/'tensorboard'),formal_command=formal)


if __name__=='__main__':
    try: main()
    except Exception as e:
        status('failed',error=repr(e))
        raise
