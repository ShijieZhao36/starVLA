#!/usr/bin/env python3
"""Prepare, check, submit and inspect a pinned two-node experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

from pipeline_node import stage_commands

HERE = Path(__file__).resolve().parent
ARTIFACT_FILES = ('deploy.py', 'make_spec.py', 'pipeline_node.py', 'test_pipeline_node.py',
                  'train_files/coordinate_nodes.py', 'train_files/run_mixed_16gpu_pretrain.sh',
                  'train_files/validate_mixed16.py', 'train_files/accelerate_mixed_2nodes.yaml',
                  'train_files/test_mixed16.py')


def ssh(host, program, payload, timeout=180):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12',
                             '-o', 'ServerAliveInterval=15', host, 'python3', '-c', shlex.quote(program)],
                            input=json.dumps(payload), text=True, capture_output=True, timeout=timeout)
    return dict(host=host, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


CHECK_PIN = r'''
for name in ['pretrain','posttrain']:
 repo=s[name+'_repo']
 assert subprocess.check_output(['git','-C',repo,'rev-parse','HEAD'],text=True).strip()==s[name+'_commit'],'Code revision changed'
 assert not subprocess.check_output(['git','-C',repo,'status','--porcelain'],text=True).strip(),'Checkout is dirty'
'''
CHECK_ARTIFACT = r'''
for rel,digest in json.loads((root/'artifact_sha256.json').read_text()).items():
 assert hashlib.sha256((Path(s['artifact_dir'])/rel).read_bytes()).hexdigest()==digest,'Launcher snapshot changed'
'''
PREAMBLE = "import hashlib,json,os,shlex,subprocess,sys\nfrom pathlib import Path\nd=json.load(sys.stdin);s=d['spec'];root=Path(s['control_dir'])\n"
PREPARE = PREAMBLE + CHECK_PIN + r'''
for p in [Path(s['run_root'])/s['pretrain_run_id'],Path(s['run_root'])/(s['pretrain_run_id']+'_smoke3'),
          Path(s['run_root'])/s['posttrain_run_id'],Path(s['run_root'])/(s['posttrain_run_id']+'_smoke3'),
          Path(s['eval_output_dir']),Path(s['eval_output_dir']+'_smoke3')]:
 assert not p.exists(),f'Refusing existing run: {p}'
root.mkdir(parents=True,exist_ok=False)
artifacts=Path(s['artifact_dir']);artifacts.mkdir()
hashes={}
for relative,content in d['files'].items():
 p=Path(relative);assert not p.is_absolute() and '..' not in p.parts
 target=artifacts/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
 hashes[relative]=hashlib.sha256(target.read_bytes()).hexdigest()
 if target.suffix in ['.py','.sh']:target.chmod(0o755)
(root/'spec.json').write_text(json.dumps(s,indent=2,ensure_ascii=False)+'\n')
(root/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
print(json.dumps(dict(prepared=True,control_dir=str(root),files=len(hashes))))
'''
DRY_RUN = PREAMBLE + CHECK_PIN + CHECK_ARTIFACT + r'''
rank=d['rank'];mode=d['mode'];mroot=root/mode
preflight=[s['python_bin'],str(Path(s['artifact_dir'])/'train_files/validate_mixed16.py'),'preflight',
 '--repo',s['pretrain_repo'],'--expected-commit',s['pretrain_commit'],'--reference-run',s['reference_run'],
 '--run',str(Path(s['run_root'])/s['pretrain_run_id']), '--launcher',str(mroot/'pretrain'),'--node-rank',str(rank),
 '--per-device-batch-size',str(s['pretrain_per_device_batch']),'--global-batch',str(s['pretrain_global_batch']),
 '--report',str(root/f'preflight_preview_node{rank}.json')]
if mode=='smoke':preflight+=['--smoke']
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONPATH=s['pretrain_repo'],PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(preflight,env=env,text=True,capture_output=True,timeout=120)
print('SOURCE_PREFLIGHT',r.returncode,r.stdout,r.stderr);assert r.returncode==0
for phase,command,overrides in d['stages']:
 env=os.environ.copy();env.update(overrides);env['DRY_RUN']='1'
 r=subprocess.run(command,env=env,text=True,capture_output=True,timeout=120)
 print(phase,r.returncode,r.stdout,r.stderr);assert r.returncode==0
'''
READY = PREAMBLE + CHECK_PIN + CHECK_ARTIFACT + r'''
import xml.etree.ElementTree as ET
rank=d['rank'];mode=d['mode'];mroot=root/mode
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],timeout=20).strip(),'GPU processes are active'
cards=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=20)).findall('gpu');assert len(cards)==8
for g in cards:
 assert g.findtext('gpu_recovery_action')=='None'
 assert g.findtext('remapped_rows/remapped_row_pending')=='No'
 assert g.findtext('remapped_rows/remapped_row_failure')=='No'
 assert g.findtext('ecc_errors/volatile/dram_uncorrectable')=='0'
assert not (mroot/f'pipeline_node{rank}').exists(),'This pipeline already started'
assert not (mroot/f'launch_claim_node{rank}').exists(),'This node already submitted'
if mode=='formal':assert json.loads((root/'smoke_success.json').read_text())['validated'] is True,'Full smoke is required'
print(json.dumps(dict(ready=True,rank=rank,gpus=8,mode=mode)))
'''
LAUNCH = PREAMBLE + r'''
rank=d['rank'];mode=d['mode'];mroot=root/mode;mroot.mkdir(exist_ok=True)
(mroot/f'launch_claim_node{rank}').mkdir(exist_ok=False)
command=[s['python_bin'],'-E','-u',str(Path(s['artifact_dir'])/'pipeline_node.py'),'--spec',str(root/'spec.json'),
         '--node-rank',str(rank),'--mode',mode]
shell='exec '+shlex.join(command)+' > '+shlex.quote(str(mroot/f'pipeline_node{rank}.log'))+' 2>&1'
screen=s['screen_names'][rank]+'_'+mode
result=subprocess.run(['screen','-dmS',screen,'bash','-c',shell],capture_output=True,text=True)
assert result.returncode==0,(result.stdout,result.stderr)
print(json.dumps(dict(submitted=True,rank=rank,mode=mode,screen=screen,command=command)))
'''
PROBE = PREAMBLE + r'''
rank=d['rank'];result={}
for mode in ['smoke','formal']:
 for rel in [f'{mode}/pipeline_node{rank}/status.json',f'{mode}/pipeline_node{rank}/failure.json',
             f'{mode}/pipeline_node{rank}/exit_code',f'{mode}/exit_code',f'{mode}_success.json']:
  p=root/rel
  if p.is_file():
   try:result[rel]=json.loads(p.read_text())
   except (ValueError,OSError):result[rel]=p.read_text()[-3000:]
for base in [s['pretrain_run_id'],s['posttrain_run_id']]:
 for suffix in ['','_smoke3']:
  run=Path(s['run_root'])/(base+suffix);row={'exists':run.exists()}
  for name in ['pretrain_success.json','posttrain_success.json','eval_success.json','exit_code']:
   p=run/name
   if p.is_file():
    try:row[name]=json.loads(p.read_text())
    except (ValueError,OSError):row[name]=p.read_text()[-2000:]
  metrics=run/'metrics.jsonl'
  if metrics.is_file():
   with metrics.open('rb') as f:
    f.seek(max(0,metrics.stat().st_size-30000));lines=f.read().decode(errors='replace').splitlines()
   valid=[]
   for line in lines:
    try:valid.append(json.loads(line))
    except ValueError:pass
   row['last_metrics']=valid[-1:]
  result[run.name]=row
result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True,timeout=20)
print(json.dumps(result,ensure_ascii=False))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','dry-run','launch','probe'])
    parser.add_argument('--spec',required=True)
    parser.add_argument('--mode',choices=['smoke','formal'],default='smoke')
    parser.add_argument('--report',help='Optional JSON report outside the Git checkout')
    args=parser.parse_args()
    spec=json.loads(Path(args.spec).read_text())
    payloads=[dict(spec=spec,rank=rank,mode=args.mode,stages=stage_commands(spec,rank,args.mode)) for rank in range(2)]
    if args.action=='prepare':
        files={name:(HERE/name).read_text() for name in ARTIFACT_FILES}
        results=[ssh(spec['hosts'][0],PREPARE,dict(spec=spec,files=files))]
    elif args.action=='launch':
        with ThreadPoolExecutor(max_workers=2) as pool:
            ready=list(pool.map(lambda pair:ssh(pair[0],READY,pair[1]),zip(spec['hosts'],payloads)))
        assert all(row['returncode']==0 for row in ready),ready
        results=[]
        for rank in [1,0]:
            row=ssh(spec['hosts'][rank],LAUNCH,payloads[rank]);results.append(row)
            if row['returncode']:break
    else:
        program=DRY_RUN if args.action=='dry-run' else PROBE
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda pair:ssh(pair[0],program,pair[1]),zip(spec['hosts'],payloads)))
    report=dict(at=datetime.now(timezone.utc).isoformat(),action=args.action,mode=args.mode,results=results)
    if args.report:
        output=Path(args.report).resolve()
        repo=Path(spec['pretrain_repo']).resolve()
        if output.is_relative_to(repo):parser.error('--report must be outside the Git checkout')
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))
    if not all(row['returncode']==0 for row in results):sys.exit(1)


if __name__=='__main__':main()
