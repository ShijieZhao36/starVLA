"""Build paired configs from the committed Galaxea recipe; no training starts here."""
from copy import deepcopy
from pathlib import Path
import sys
import yaml

repo = Path(sys.argv[1]).resolve()
dest = repo/'examples/pretrain/GalaxeaLiberoMix/train_files'
dest.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load((repo/'examples/pretrain/GalaxeaOpenWorld/train_files/starvla_galaxea_open_world_stage1.yaml').read_text())
cfg.update(run_id='galaxea_libero_control_ln_8gpu_80k', run_root_dir='/mnt/zhaoshijie/checkpoints/starVLA',
           wandb_entity='jasperz-georgia-institute-of-technology', wandb_project='pretrain')
cfg['framework']['qwenvl']['base_vlm'] = '/mnt/zhaoshijie/workspace/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct'
cfg['framework']['action_model']['decoder_input_norm'] = True
cfg['trainer'].update(pretrained_checkpoint=None, is_resume=False, audit_source_exposure=True,
                      audit_initial_model=True)
cfg['datasets']['vla_data']['audit_sampling'] = True
mixed = deepcopy(cfg)
mixed['run_id'] = 'galaxea_libero_mixed_ln_8gpu_80k'
d = mixed['datasets']['vla_data']
d['data_mix'] = 'galaxea_libero_plus_stage1_80_20'
d['data_root_dirs_by_robot'] = {'libero_plus_rot6d': '/mnt/zhaoshijie/dataset/_processed/libero_plus/stage1'}
d['frame_cache_dirs_by_robot'] = {'libero_plus_rot6d': '/mnt/zhaoshijie/frame_cache/libero_plus_stage1_256'}
for arm, obj in [('control', cfg), ('mixed', mixed)]:
    (dest/f'starvla_{arm}_ln_80k.yaml').write_text(yaml.safe_dump(obj, sort_keys=False))
print('Created paired configs: identical model/optimizer/budget; different dataset mixture only.')
