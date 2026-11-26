"""Audit the returned fixed-model precision ladder without treating it as policy evidence."""
import argparse
import math
from pathlib import Path
import numpy as np
from ral_eight_hour import ROOT, read, digest, atomic_json


def audit(folder):
    from ral_backend_diagnose import select_blocks
    from ral_mechanism import summarize
    from ral_mechanism_probe import FACTORS, HORIZONS
    plan = read(ROOT/'configs/ral/mechanism_probe_v1.json')
    status = read(folder/'status.json')
    assert status['machine']=='5080' and status['status']=='completed' and status['new_training_runs']==0
    assert read(folder/'protocol.json')==read(ROOT/'release/ral/mechanism_probe_protocol_v1.json')
    assert status['protocol_sha256']==digest(ROOT/'release/ral/mechanism_probe_protocol_v1.json')
    retention=plan['machines']['5080']['archives'][0]['retention']
    assert read(folder/'horizon32_verified_members.json')==retention['members']
    original=read(folder/'judgment.json');assert summarize(folder)==original
    manifest=read(ROOT/plan['responses'])
    expected={}
    for split in ('fit','cal'):
        entry=manifest['splits'][split];path=(ROOT/plan['responses']).parent/entry['file']
        assert digest(path)==entry['sha256']
        with np.load(path,allow_pickle=False) as z:
            expected[split]={str(x) for x in z['block_id'][select_blocks(z['block_id'],24)]}
    out=dict(status='audited',formal_results=False,training_seed=301,source_status_sha256=digest(folder/'status.json'),
        wall_seconds=status['finished_unix']-status['started_unix'],checkpoint_tensor_bytes_verified_locally=False,
        scope='One frozen A+/B pair; 24 fit + 24 cal blocks each; precisions/amplitudes are repeated measurements, not extra training seeds. Model FD versus own AD, not model versus physics or policy success.',
        response_test_accessed=False,new_training_runs=0,models={})
    assert {j['id'] for j in status['jobs']}=={'horizon32_Aplus','horizon32_B'}
    for job in status['jobs']:
        path=folder/job['output'];assert job['status']=='completed' and digest(path)==job['sha256']
        d=read(path);label=job['id'].split('_',1)[1]
        assert d['checkpoint_sha256']==retention['members'][f'jobs/s301_{label}/run/ckpt_005000.pt']
        assert d['status']=='completed' and d['optimizer_steps']==0 and d['weights_and_normalizers_unchanged'] and not d['response_test_accessed']
        assert len(d['blocks'])==96
        stats={}
        for split in ('fit','cal'):
            for precision in ('float32','float64'):
                rows=[r for r in d['blocks'] if r['split']==split and r['precision']==precision]
                assert len(rows)==24 and {r['block_ids'][0] for r in rows}==expected[split]
                assert all(r['actual_parameter_dtype']=='torch.'+precision for r in rows)
                for row in rows:
                    assert tuple(r['epsilon_factor'] for r in row['epsilon_ladder'])==FACTORS
                    assert all(set(r['horizons'])=={str(h) for h in HORIZONS} for r in row['epsilon_ladder'])
                horizons={}
                for h in map(str,HORIZONS):
                    vals=np.array([[r['horizons'][h]['fd_vs_ad_scaled_rmse'] for r in row['epsilon_ladder']] for row in rows])
                    rel=np.array([row['epsilon_ladder'][-1]['horizons'][h]['relative_error'] for row in rows],dtype=float)
                    assert np.isfinite(vals).all() and np.isfinite(rel).all()
                    horizons[h]=dict(median_rmse_by_factor=vals.mean(0).tolist())
                    horizons[h].update(median_rmse_by_factor=np.median(vals,axis=0).tolist(),
                        final_error_lower_count=int((vals[:,-1]<vals[:,0]).sum()),
                        final_error_max=float(vals[:,-1].max()),final_relative_error_max=float(rel.max()),
                        final_worst_block=rows[int(vals[:,-1].argmax())]['block_ids'][0])
                stats[split+'_'+precision]=dict(blocks=24,horizons=horizons,
                    channel_medians={k:float(np.median([r['algebraic_channels'][k] for r in rows])) for k in
                        ('near_zero_velocity_fraction','friction_slope_max','saturation_fraction','near_clamp_boundary_fraction')})
        out['models'][label]=dict(checkpoint_sha256=d['checkpoint_sha256'],report_sha256=digest(path),summaries=stats)
    out['interpretation']=[
        'FP64 errors decrease at every sampled block for h8/h32; supports finite-amplitude nonlinearity rather than a pervasive AD implementation failure. Worst blocks remain reported.',
        'FP32 very small central differences suffer roundoff; smaller epsilon alone is not a remedy.',
        'Friction smoothing is a prespecified candidate ablation, not an established cause or fix. Keep the physical plant and clamp unchanged.',
        'No new policy evaluation, no superiority claim, no automatic C. 4090 archived weights reported deleted; its fixed-batch loss task is cancelled, historical metadata retained.']
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',type=Path,default=ROOT/'logs/ral_mechanism/import/mechanism01_5080')
    p.add_argument('--out',type=Path,default=ROOT/'release/ral/mechanism01_5080_review_20261124.json')
    a=p.parse_args();atomic_json(a.out,audit(a.session));print(a.out)
