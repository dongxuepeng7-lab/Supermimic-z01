"""Explicit empirical-risk windows; strict calibration records remain unchanged."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path


def empirical_calibration(strict, spec):
    """Rank cal-only risks; never convert a failed threshold check into a pass."""
    out = copy.deepcopy(strict)
    short, long = spec['short'], spec['long']
    ranked = []
    for row in out['bins']:
        errors = row['prefix_error'][:long]
        finite = len(errors) == long and all(x is not None and math.isfinite(x) and x < spec['unreliable_sentinel'] for x in errors)
        enough = row['eligible_blocks'] >= out['min_blocks'] and row['independent_sessions'] >= 2
        row['empirical_risk'] = max(errors) if finite else None
        row['empirical_rank_eligible'] = enough and finite
        if enough and finite:
            ranked.append((row['empirical_risk'], row['bin']))
        for field, h in [('short_validated', short), ('long_validated', long)]:
            row[field] = enough and len(row['prefix_error']) >= h and all(
                x is not None and math.isfinite(x) and x <= out['tolerance'] for x in row['prefix_error'][:h])
    ranked.sort()
    nlong = math.floor(len(ranked) * spec['long_fraction_of_reliable_bins'])
    long_bins = {b for _, b in ranked[:nlong]}
    for row in out['bins']:
        row['horizon'] = long if row['bin'] in long_bins else short
        row['assignment_is_certified'] = False
    out.update(short=short, long=long, empirical_protocol='cal_risk_rank_v1',
               empirical_spec=spec, empirical_long_bins=sorted(long_bins),
               original_short=strict['short'], original_long=strict['long'],
               certified_policy_return_bound=False, formal_results=False,
               response_test_accessed=False)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=('risk','live','window'))
    for key in ('checkpoint','params','motions','out','config'):
        p.add_argument('--'+key,type=Path,required=True)
    for key in ('calibration','starts'):
        p.add_argument('--'+key,type=Path)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--variant',choices=('D_emp','C_perm_emp','C_fixed16','C_fixed32'))
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.kind=='window' and (a.variant is None or a.starts is None or a.calibration is None):p.error('window requires variant, starts and calibration')
    if a.kind=='risk' and a.calibration is None:p.error('risk requires calibration')
    from ral_eight_hour import atomic_json,digest
    if a.out.exists():raise FileExistsError(a.out)
    if a.kind=='risk':
        strict=json.loads(a.calibration.read_text())
        if strict['checkpoint_sha256'] != digest(a.checkpoint) or strict['split']!='cal':
            raise ValueError('Calibration source mismatch')
        result=empirical_calibration(strict,cfg['empirical_windows'])
        result['original_strict_calibration_sha256']=digest(a.calibration)
        atomic_json(a.out,result);return
    import torch
    torch.manual_seed(a.seed)
    if a.kind=='live':
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        if 'pi_live' not in ck:raise ValueError('Live actor absent')
        ck['pi']=copy.deepcopy(ck['pi_live'])
        a.out.mkdir(parents=True)
        torch.save(ck,a.out/'last.pt')
        atomic_json(a.out/'args.json',ck['args'])
        atomic_json(a.out/'run.json',dict(status='completed',kind='live_parent_export',
            source_checkpoint_sha256=digest(a.checkpoint),checkpoint_sha256=digest(a.out/'last.pt'),
            actor_weights='live',new_training_updates=0,formal_results=False))
        return
    import numpy as np
    from partymimic.experiments.cli import load_context
    from partymimic.experiments.windows import train_windows,state_bins
    context=load_context(a.checkpoint,a.params,a.motions,'cuda:0')
    ck,robot,layout,wm,motion,mdp=context
    cal=json.loads(a.calibration.read_text())
    if cal.get('empirical_protocol')!='cal_risk_rank_v1':raise ValueError('Empirical mapping required')
    meta=json.loads(a.starts.read_text());data=a.starts.parent/meta['file']
    if digest(data)!=meta['sha256']:raise ValueError('Start pool changed')
    with np.load(data,allow_pickle=False) as z:states=torch.as_tensor(z['s'],device='cuda:0')
    bins=state_bins(states,torch.tensor(cal['speed_edges_rad_s'],device='cuda:0'),cal['binning'],mdp.kin,mdp.feet)
    assignment=torch.tensor([r['horizon'] for r in cal['bins']],device='cuda:0')[bins]
    unique=sorted(assignment.unique().cpu().tolist())
    mixed=len(unique)==2
    fixed=int(a.variant.replace('C_fixed','')) if a.variant.startswith('C_fixed') else None
    # A constant empirical map can be executed as an explicit degenerate control,
    # but can never support a state-dependent horizon-selection claim.
    if fixed is None and not mixed:fixed=int(unique[0])
    group='C-perm' if a.variant=='C_perm_emp' else 'D'
    del states,bins,assignment,wm,context,ck
    w=cfg['windows']
    train_windows(a.checkpoint,a.calibration,a.starts,a.out,group,robot,motion,mdp,layout,
                  w['iterations'],w['batch_size'],a.seed,w['lr'],'cuda:0',fixed_horizon=fixed)
    run=json.loads((a.out/'run.json').read_text())
    run.update(experiment_variant=a.variant,window_protocol='empirical_cal_risk_rank_v1',
               original_certified_D=False,empirical_assignment_identifiable=mixed,
               declared_horizons=[cal['short'],cal['long']],actual_pool_horizons=unique,
               original_strict_calibration_sha256=cal['original_strict_calibration_sha256'],
               note='No strict calibration guarantee. Constant maps are degenerate controls, not evidence of adaptive selection.')
    atomic_json(a.out/'run.json',run)


if __name__=='__main__':main()
