"""Opt-in full-shape WM32 GPU test; two updates only, not research training. Pass a fresh output path."""
import hashlib,os,sys,time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from ral_eight_hour import ROOT,atomic_json,read,execute,gpu_lock
from ral_online_night import verify_training,archive_pair
from partymimic.experiments.cli import training_command
from partymimic.experiments.protocol import freeze
out=Path(sys.argv[1]).resolve()
out.mkdir(parents=True,exist_ok=False)
a=SimpleNamespace(params=ROOT/'release/ral/params.csv',motions=ROOT/'data/motions/z01_lafan_v2_clean',responses=ROOT/'data/ral_pairs_v1/manifest.json')
cfg=read(ROOT/'configs/ral/diagnose_nine_hour_v1.json')
cfg['training'].update(iterations=2,wm_window=32,ckpt_every=2,log_every=1)
atomic_json(out/'config.json',cfg)
p=freeze(out/'config.json',a.params,a.motions,a.responses,out/'protocol.json')
p.update(groups=['B'],training_seeds=[303],expected_environment_transitions=70656,smoke_only=True)
atomic_json(out/'protocol.json',p)
job=dict(id='s303_B',group='B',seed=303)
run=out/'jobs/s303_B/run';run.parent.mkdir(parents=True)
cmd=training_command(SimpleNamespace(config=out/'config.json',params=a.params,motion_dir=a.motions,responses=a.responses,group='B',seed=303,protocol=out/'protocol.json',out=run))
atomic_json(out/'jobs/s303_B/command.json',cmd)
lock=ROOT/'logs/ral_8h'/('.gpu_'+hashlib.sha256(os.environ.get('CUDA_VISIBLE_DEVICES','0').encode()).hexdigest()[:12]+'.lock')
with gpu_lock(lock) as fd:
 status=execute(cmd,run.parent/'console.log',time.monotonic()+600,[False],fd)
 if status!='completed':raise RuntimeError(status)
 ledger=verify_training(run,p,job);job['validated']=True
 record=archive_pair(out,dict(id='fullsize',jobs=[job]),2)
 atomic_json(out/'validation.json',dict(status='completed',smoke_only=True,formal_results=False,gpu='RTX3090',iterations=2,full_training_shapes=True,horizon=32,wm_window=32,retention=record,env_steps=ledger['env_steps']))
print('Full-shape WM32 GPU smoke passed')
