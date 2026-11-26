"""Fit-only world-model diagnostic sweep; cal selection, never policy/test evaluation."""
import argparse
import copy
import json
from pathlib import Path
import time

import torch

import partymimic.networks as networks
from partymimic.dr import XiLayout
from partymimic.experiments.cli import common_robot
from partymimic.experiments.models import make_world_model
from partymimic.experiments.parameters import sha256
from partymimic.experiments.protocol import code_hash
from partymimic.experiments.response import (
    ResponseData, initialize_pair_normalizers, paired_losses, paired_predictions,
    validate_response_objective,
)
from partymimic.experiments.runtime import object_hash, state_hash
from partymimic.mdp import unflatten_state
from partymimic.optim import MuonAdam


@torch.no_grad()
def evaluate(wm, dataset, layout):
    forward = response8 = response32 = qerror = 0.
    for i in range(0, dataset.count, 128):
        b = {k:v[i:i+128] for k,v in dataset.data.items()}
        f, _, e = paired_losses(wm,b,layout.normalize,dataset.q_scale,dataset.v_scale)
        plus,minus = paired_predictions(wm,b,layout.normalize)
        q = ((plus[4]-unflatten_state(b['s_plus'])[4]).square().mean()
             +(minus[4]-unflatten_state(b['s_minus'])[4]).square().mean())/2
        fraction=len(b['s0'])/dataset.count
        forward += float(f)*fraction
        response8 += float(e[:,:8].square().mean())*fraction
        response32 += float(e.square().mean())*fraction
        qerror += float(q)*fraction
    return dict(forward_mse=forward,response_rmse8=response8**.5,
                response_rmse32=response32**.5,q_rmse32_rad=qerror**.5)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--params',default='release/ral/params.csv')
    parser.add_argument('--responses',default='data/ral_pairs_v1/manifest.json')
    a=parser.parse_args()
    plan=json.loads(Path(a.plan).read_text())
    output=Path(a.out)
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    networks.AMP_DTYPE=None
    networks.RunningNorm.no_sync=True
    robot=common_robot(a.params)
    layout=XiLayout(robot,'cuda:0')
    fit=ResponseData(a.responses,'fit',robot,layout,sha256(a.params),'cuda:0')
    cal=ResponseData(a.responses,'cal',robot,layout,sha256(a.params),'cuda:0')
    if min(plan['updates'],plan['batch_size'])<1:
        raise ValueError('Positive diagnostic budget required')
    for c in plan['candidates']:
        validate_response_objective(c['response_horizon'],c['response_loss'],c['response_huber_delta'],fit.horizon)
        if not 0 <= c['response_weight'] < float('inf'):
            raise ValueError('Invalid response weight')
    report=dict(schema_version=1,purpose='world-model-only diagnostic; not full C training',
                formal_results=False,test_outcomes_inspected=False,plan=plan,plan_sha256=sha256(a.plan),
                code_sha256=code_hash(),tool_sha256=sha256(__file__),
                response_sha256=sha256(a.responses),params_sha256=sha256(a.params),
                gpu=torch.cuda.get_device_name(),results=[])
    (output/'plan.json').write_text(json.dumps(report,indent=2))
    for seed in plan['seeds']:
        torch.manual_seed(seed)
        args=dict(experiment_model='actuator',wm_hidden=plan['wm_hidden'],wm_structured=True,
                  wm_amp='none',friction_speed=.01)
        initial=make_world_model(robot,layout,args).cuda()
        initialize_pair_normalizers(initial,fit,layout.normalize)
        initial_state=copy.deepcopy(initial.state_dict())
        initial_hash=state_hash({'wm':initial})
        del initial
        # Store the actual shared draw schedule before running any candidate.
        generator=torch.Generator(device='cuda:0').manual_seed(seed+7301)
        schedule=torch.randint(fit.count,(plan['updates'],plan['batch_size']),
                               generator=generator,device='cuda:0')
        torch.save(schedule.cpu(),output/f'schedule_s{seed}.pt')
        for candidate in plan['candidates']:
            torch.manual_seed(seed)
            wm=make_world_model(robot,layout,dict(args,wm_amp=candidate['wm_amp'])).cuda()
            wm.load_state_dict(initial_state)
            assert state_hash({'wm':wm})==initial_hash
            optimizer=MuonAdam(wm,[wm.net],lr=plan['lr'],betas=(.9,.999),
                               fused=True,capturable=True,flat_adam=True)
            curves=[]
            started=time.monotonic()
            for it,ids in enumerate(schedule):
                batch={k:v[ids] for k,v in fit.data.items()}
                f,r,errors=paired_losses(wm,batch,layout.normalize,fit.q_scale,fit.v_scale,
                    **{k:candidate[k] for k in ('response_horizon','response_loss','response_huber_delta')})
                loss=f+candidate['response_weight']*r
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(wm.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                if (it+1)%20==0 or it==0:
                    curves.append(dict(update=it+1,forward=float(f.detach()),response=float(r.detach()),
                                       gradient_norm=float(norm)))
            wm.eval()
            result=dict(seed=seed,**candidate,init_sha256=initial_hash,
                        sampling_sha256=sha256(output/f'schedule_s{seed}.pt'),
                        updates=plan['updates'],batch_size=plan['batch_size'],
                        elapsed_s=time.monotonic()-started,cal=evaluate(wm,cal,layout),
                        fit=evaluate(wm,fit,layout),curve=curves)
            torch.save(dict(wm=wm.state_dict(),args=dict(args,**candidate),
                            diagnostic_only=True,updates=plan['updates']),
                       output/f"{candidate['name']}_s{seed}.pt")
            report['results'].append(result)
            (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
            print(json.dumps({k:result[k] for k in ('name','seed','cal','elapsed_s')}),flush=True)
            del wm,optimizer,loss,f,r,errors
            torch.cuda.empty_cache()
    print(output/'results.json')


if __name__=='__main__':
    main()
