"""Read-only mechanism probes. No policy optimization, test labels, or automatic C."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from ral_eight_hour import atomic_json, digest, read

HORIZONS = (1, 2, 4, 8, 16, 32)
FACTORS = (1., .5, .25, .125, .0625, .03125, .015625, .00390625)


def rmse(x):
    if not bool(torch.isfinite(x).all()):
        raise ValueError("Nonfinite probe: stop and retain partial results")
    return float(x.detach().double().square().mean().sqrt())


def loss_statistics(errors, mask, gradient_norm):
    """The trainer sums time and divides by B; alternatives are scalar rescalings.

    Reports hypothetical pre-clip norms, NOT Muon update sizes or training clip rates.
    """
    valid = float(mask.sum())
    if valid <= 0:
        raise ValueError("No valid timesteps")
    batch, horizon = errors.shape
    summed = float((errors * mask).sum() / batch)
    def variant(divisor):
        norm = gradient_norm / divisor
        return dict(loss=summed/divisor, preclip_grad_norm=norm,
                    would_clip_at_1=norm > 1., clip_multiplier=min(1., 1./(norm+1e-6)))
    return dict(valid_steps=valid, possible_steps=mask.numel(), valid_fraction=valid/mask.numel(),
                per_sequence_sum=variant(1.), divide_by_horizon=variant(horizon),
                divide_by_valid_steps=variant(valid/batch))


def roll_states(wm, state, previous, context, actions):
    from partymimic.mdp import unflatten_state
    state, states = unflatten_state(state), []
    for action in actions.unbind(1):
        state = wm.step(state, action, previous, context)
        states.append(state)
        previous = action
    return tuple(torch.stack([s[k] for s in states], 1) for k in range(6))


def canonical_inputs(ds, ids, device, dtype):
    # All precisions share the same FP32-valued initial states/context. Compute
    # midpoint/direction once in double so FP32 midpoint rounding is not confused
    # with a different dataset. Never renormalize truncated sequence directions.
    d = {k: v[ids].double() for k, v in ds.data.items()}
    d['base'] = (d['a_plus'] + d['a_minus']) / 2
    d['direction'] = (d['a_plus'] - d['a_minus']) / (2*d['epsilon'][:, None, None])
    return {k: v.to(device=device, dtype=dtype) for k, v in d.items()}


def derivative_probe(wm, layout, d, q_scale, v_scale):
    from ral_gradient_diagnose import rollout
    from partymimic.mdp import unflatten_state
    scales = d['base'].new_tensor([q_scale]*wm.num_j + [v_scale]*wm.num_j)
    context = layout.normalize(d['xi'])
    fn = lambda u: rollout(wm, d['s0'], d['a_prev'], context, u)
    _, ad = torch.autograd.functional.jvp(fn, d['base'], d['direction'])
    rmse(ad)
    results = []
    with torch.no_grad():
        for factor in FACTORS:
            step = d['epsilon'][:, None, None] * factor
            plus, minus = d['base'] + step*d['direction'], d['base'] - step*d['direction']
            fd = (fn(plus)-fn(minus))/(2*step)
            achieved = (plus-minus)/(2*step)
            rows = {}
            for h in HORIZONS:
                err, signal = rmse((fd-ad)[:, :h]/scales), rmse(ad[:, :h]/scales)
                rows[str(h)] = dict(fd_vs_ad_scaled_rmse=err, ad_scaled_rms=signal,
                    relative_error=err/signal if signal > 1e-12 else None)
            results.append(dict(epsilon_factor=factor, horizons=rows,
                direction_roundoff_rmse=rmse(achieved-d['direction']),
                collapsed_action_fraction=float(((plus == minus) & (d['direction'] != 0)).float().mean())))
        # Diagnose the algebraic channels at rolled states, without changing B.
        s, previous = unflatten_state(d['s0']), d['a_prev']
        velocities, raws = [], []
        for a in d['base'].unbind(1):
            velocities.append(s[5])
            target = wm.q0 + wm.action_scale*a + d['xi'][..., wm.offset_slice]
            raws.append(wm.kp*(target-s[4])-wm.kd*s[5])
            s = wm.step(s, a, previous, context); previous = a
        velocity, raw = torch.stack(velocities, 1), torch.stack(raws, 1)
        slope = wm.friction/wm.friction_speed * (1-torch.tanh(velocity/wm.friction_speed).square())
        channels = dict(near_zero_velocity_fraction=float((velocity.abs() < wm.friction_speed).float().mean()),
            friction_slope_max=float(slope.max()), saturation_fraction=float((raw.abs() >= wm.limit).float().mean()),
            near_clamp_boundary_fraction=float(((raw.abs()-wm.limit).abs() < .01*wm.limit).float().mean()),
            friction_local_fd=[])
        for factor in (1., .1, .01, .001):
            dv = wm.friction_speed*factor
            fd = wm.friction*(torch.tanh((velocity+dv)/wm.friction_speed)-torch.tanh((velocity-dv)/wm.friction_speed))/(2*dv)
            channels['friction_local_fd'].append(dict(velocity_step=dv, slope_rmse=rmse(fd-slope)))
    return dict(epsilon_ladder=results, algebraic_channels=channels)


def loss_probe(wm, layout, d, horizon):
    from partymimic.mdp import unflatten_state
    # Both stored branches, identical inputs across all four frozen checkpoints.
    actions = torch.cat((d['a_plus'], d['a_minus']))[:, :horizon]
    targets = torch.cat((d['s_plus'], d['s_minus']))[:, :horizon]
    pred = roll_states(wm, d['s0'].repeat(2, 1), d['a_prev'].repeat(2, 1),
                       layout.normalize(d['xi'].repeat(2, 1)), actions)
    floors = actions.new_tensor([.003]*3 + [.005]*3 + [.02]*3 + [.05]*3 + [.003]*wm.num_j + [.05]*wm.num_j)
    err = (wm.state_error(pred, unflatten_state(targets))/torch.maximum(wm.inc_norm.std(), floors)).square().mean(-1)
    mask = torch.ones_like(err)  # CPU paired trajectories have no episode-validity mask.
    loss = (err*mask).sum()/err.shape[0]
    wm.zero_grad(set_to_none=True)
    loss.backward()
    norm = float(torch.stack([p.grad.double().square().sum() for p in wm.parameters() if p.grad is not None]).sum().sqrt())
    rmse(err)
    if not np.isfinite(norm):
        raise ValueError("Nonfinite WM gradient")
    result = loss_statistics(err.detach(), mask, norm)
    result.update(horizon=horizon, first8_mean_error=float(err[:, :8].detach().mean()),
        late_mean_error=float(err[:, 8:].detach().mean()) if horizon > 8 else None,
        q_rmse_rad=rmse(pred[4]-targets[..., 13:13+wm.num_j]),
        mean_error_by_step=err.detach().mean(0).double().tolist())
    wm.zero_grad(set_to_none=True)
    return result


def model_probe(a):
    from partymimic.experiments.cli import load_context
    from partymimic.experiments.response import ResponseData
    from ral_backend_diagnose import select_blocks
    from ral_eight_hour_worker import state_digest
    import partymimic.networks as networks
    sha = digest(a.checkpoint)
    ck, robot, layout, wm = load_context(a.checkpoint, a.params, device=a.device)
    if wm.mode not in ('matched', 'actuator') or ck['args']['wm_amp'] != 'none':
        raise ValueError('Requires frozen FP32 A+/B world models')
    # The production "none" path explicitly casts x.float(). This diagnostic-only
    # route disables autocast without that cast, permitting genuine FP64 matmuls.
    networks.AMP_DTYPE = None
    wm.mlp_amp = 'inherit'
    wm.eval()
    del ck
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = dict(status='running', task=a.task, checkpoint_sha256=sha, model=wm.mode,
        response_test_accessed=False, formal_results=False, optimizer_steps=0,
        scope='Frozen model on CPU-generated fit/cal pairs; not online replay, actor-gradient, or success-rate evidence',
        precision_scope='Same stored FP32 weights/inputs promoted to FP64; diagnostic MLP path bypasses forced FP32 cast; autocast/TF32 disabled',
        loss_scope='All paired timesteps valid; hypothetical clipping on these fixed small batches is NOT historical online clipping frequency or Muon effective update',
        channel_scope='Algebraic channel observations, not a friction/clamp intervention or causal attribution; A+ uses raw action', blocks=[])
    # Load each split once on CPU, never access the locked test NPZ.
    for split in ('fit', 'cal'):
        ds = ResponseData(a.responses, split, robot, layout, digest(a.params), 'cpu')
        ids = select_blocks(ds.arrays['block_id'], a.blocks_per_split)
        for precision in (('float32', 'float64') if a.task == 'derivative' else ('float32',)):
            dtype = getattr(torch, precision)
            wm.to(dtype=dtype)
            before = state_digest({'wm': wm.state_dict()}, ('wm',))
            wm.requires_grad_(a.task == 'loss')
            size = 1 if a.task == 'derivative' else 8
            for start in range(0, len(ids), size):
                sub = ids[start:start+size]
                d = canonical_inputs(ds, sub, a.device, dtype)
                row = dict(split=split, precision=precision,
                           block_ids=[str(ds.arrays['block_id'][i]) for i in sub],
                           actual_parameter_dtype=str(next(wm.parameters()).dtype))
                if a.task == 'derivative':
                    row.update(derivative_probe(wm, layout, d, ds.q_scale, ds.v_scale))
                else:
                    row['windows'] = [loss_probe(wm, layout, d, h) for h in (8, 32)]
                report['blocks'].append(row)
                atomic_json(a.out, report)
                print(f"{a.task}: {split} {precision} {start+len(sub)}/{len(ids)}", flush=True)
            if state_digest({'wm': wm.state_dict()}, ('wm',)) != before:
                raise ValueError('Model weights/normalizers changed during diagnosis')
    if digest(a.checkpoint) != sha:
        raise ValueError('Checkpoint changed')
    report.update(status='completed', weights_and_normalizers_unchanged=True)
    atomic_json(a.out, report)


def constraint_features(nefc, solver, repeats, branches, horizon):
    n = nefc.reshape(repeats, branches, horizon, -1)
    it = solver.reshape(repeats, branches, horizon, -1)
    return dict(any_constraints=bool(n.any()),
                branch_constraint_count_differs=bool((n[:, 1:] != n[:, :1]).any()),
                constraint_count_changes_in_time=bool((np.diff(n, axis=2) != 0).any()),
                solver_reached_100=bool((it >= 100).any()),
                max_constraints=int(n.max()), max_solver_iterations=int(it.max()))


def backend_probe(a):
    from ral_backend_diagnose import compare
    from ral_diagnose_review import equal
    state = read(a.session/'status.json')
    report = dict(status='running', task='backend_strata', source_status_sha256=digest(a.session/'status.json'),
        response_test_accessed=False, formal_results=False, new_simulator_rollouts=0,
        scope='Existing eager_single r0 raw data; r1 is repeatability, not extra independent samples. Full original and failure-enriched new directions remain separate.',
        constraint_scope='Counts of all solver constraints, NOT contact identities or proof of a contact-caused failure; traces sampled after each control step, not every substep.',
        cells=[], strata={})
    for cell in state['backend']:
        if cell['status'] != 'completed':
            continue
        record = next(j for j in cell['jobs'] if j['profile'] == 'eager_single' and j['repeat'] == 0)
        folder = a.session/record['folder']
        if digest(folder/'summary.json') != record['summary_sha256']:
            raise ValueError('Historical backend summary changed')
        old = read(folder/'summary.json')
        if old['status'] != 'completed' or old['response_test_accessed'] or old['smoke_only']:
            raise ValueError('Incomplete or non-development input')
        paths = {digest(p): p for p in folder.glob('*_trajectories.npz')}
        result = dict(id=cell['id'], direction_seed=cell['direction_seed'], blocks=[])
        for block in old['blocks']:
            with np.load(paths[block['trajectory_sha256']], allow_pickle=False) as z:
                rows = compare(z['train_cpu'], z['train_gpu'], z['scales'], z['epsilon'], old['config'])
                equal(rows, block['backend'])
                r, b, h, _ = z['train_gpu'].shape
                features = constraint_features(z['gpu_nefc'], z['gpu_solver_niter'], r, b, h)
            result['blocks'].append(dict(split=block['split'], block_id=block['block_id'],
                overall_passed=block['passed'], features=features, measurements=rows,
                trajectory_sha256=block['trajectory_sha256']))
        report['cells'].append(result)
        atomic_json(a.out, report)
        print(f"backend strata: {cell['id']} {len(result['blocks'])} blocks", flush=True)
    if not report['cells'] or report['cells'][0]['id'] != 'full_original':
        raise ValueError('Full original coverage missing')
    # Each stratum counts blocks once per cell; amplitude/horizon repeated measurements
    # are reported separately, never used as independent training seeds.
    for population, cells in (('full_original', report['cells'][:1]), ('failure_enriched_directions', report['cells'][1:])):
        buckets = defaultdict(lambda: dict(block_direction_count=0, overall_failed=0))
        measurements = defaultdict(list)
        for cell in cells:
            for block in cell['blocks']:
                for feature in ('any_constraints', 'branch_constraint_count_differs', 'constraint_count_changes_in_time', 'solver_reached_100'):
                    key = f"{feature}={block['features'][feature]}"
                    buckets[key]['block_direction_count'] += 1
                    buckets[key]['overall_failed'] += not block['overall_passed']
                for m in block['measurements']:
                    measurements[f"h{m['horizon']}_eps{m['factor']}"].append(m)
        report['strata'][population] = dict(features=dict(buckets), amplitude_horizon={
            key: dict(block_direction_count=len(rows), backend_failed=sum(not r['passed'] for r in rows),
                      median_cpu_response_rms=float(np.median([r['cpu_response_rms'] for r in rows])),
                      median_cross_response_rmse=float(np.median([r['cross_response_rmse'] for r in rows])))
            for key, rows in measurements.items()})
    if digest(a.session/'status.json') != report['source_status_sha256']:
        raise ValueError('Historical session changed')
    report['status'] = 'completed'
    atomic_json(a.out, report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('derivative', 'loss', 'backend'), required=True)
    p.add_argument('--out', type=Path, required=True)
    for key in ('checkpoint', 'params', 'responses', 'session'):
        p.add_argument('--'+key, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--blocks-per-split', type=int, default=24)
    a = p.parse_args()
    if a.out.exists() or a.blocks_per_split < 1:
        p.error('Fresh output and positive count required')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.task == 'backend':
        if a.session is None: p.error('--session required')
        backend_probe(a)
    else:
        if any(x is None for x in (a.checkpoint, a.params, a.responses)): p.error('Model/data paths required')
        model_probe(a)


if __name__ == '__main__':
    main()
