"""5080-only C response-gradient-cap experiment; nine-hour active budget.

Adds an isolated entry point without changing files frozen by cd02 on 3090.
All primary B/C cells are evaluated before any conditional D work begins.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import ral_cd_nine_hour as queue

ROOT = queue.ROOT
CONFIG = ROOT / 'configs/ral/cd_strength_v1.json'
ADDITIONS = ('tools/ral_cd_strength.py', 'configs/ral/cd_strength_v1.json',
             'scripts/ral_5080_cd_strength.sh')
ORIGINAL_TRAINING_PLAN = queue.training_plan
ORIGINAL_SUMMARIZE = queue.summarize
ORIGINAL_DIAGNOSTICS = queue.training_diagnostics
ORIGINAL_VERIFY = queue.base.verify_training


def validate_plan(plan):
    if plan['hours'] != 9 or plan['seeds'] != [501, 505, 509, 513, 517]:
        raise ValueError('Expected the preregistered nine-hour/five-seed plan')
    candidates = plan['machines']['5080']['candidates']
    expected = [('B', 'B', 0., .1), ('C_h32_cap10', 'C', .01, .1),
                ('C_h32_cap30', 'C', .01, .3)]
    actual = [(c['name'], c['group'], c['response_weight'], c['response_grad_ratio']) for c in candidates]
    if actual != expected or any(c['response_horizon'] != 32 for c in candidates):
        raise ValueError('Expected B / C_h32_cap10 / C_h32_cap30, differing only in declared response settings')
    if plan['machines']['5080']['window_parent'] != 'C_h32_cap30':
        raise ValueError('Window parent must be fixed before outcomes are observed')
    original = queue.read(ROOT / 'configs/ral/cd_nine_hour_v1.json')
    for key in ('training_overrides', 'source_archive_sha256', 'pairs', 'windows', 'evaluation'):
        if plan[key] != original[key]:
            raise ValueError(f'Unexpected change to the matched protocol: {key}')



def check_dependencies(plan):
    for name, fingerprint in plan['required_existing_files_sha256'].items():
        path = ROOT/name
        if not path.is_file() or queue.digest(path) != fingerprint:
            raise ValueError(f'Existing cd02 dependency missing or changed: {name}')
    source = {str(p.relative_to(ROOT)): queue.digest(p) for p in sorted((ROOT/'src').rglob('*.py'))}
    fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if fingerprint != plan['required_code_sha256']:
        raise ValueError('Training source differs from the reviewed native-response implementation')


def verify_training(run, parent, plan, group):
    result = ORIGINAL_VERIFY(run, parent, plan, group)
    args = queue.read(run/'args.json')
    for key in ('response_horizon', 'response_grad_ratio'):
        if args[key] != plan['training_overrides'][key]:
            raise ValueError(f'Trained candidate differs from its assigned plan: {key}')
    expected_weight = plan['C_response_weight'] if group == 'C' else 0.
    if args['response_weight'] != expected_weight:
        raise ValueError('Trained response weight differs')
    return result


def training_plan(plan, candidate):
    result = ORIGINAL_TRAINING_PLAN(plan, candidate)
    result['training_overrides']['response_grad_ratio'] = candidate['response_grad_ratio']
    return result


def diagnostics(run):
    result = ORIGINAL_DIAGNOSTICS(run)
    args = queue.read(run / 'args.json')
    result.update(response_horizon=args['response_horizon'],
                  configured_response_weight=args['response_weight'],
                  configured_gradient_cap=args['response_grad_ratio'])
    return result


def summarize(q):
    ORIGINAL_SUMMARIZE(q)
    path = q.a.out / 'summary.json'
    result = queue.read(path)
    cells = result['completion_percent']
    a, b = 'C_h32_cap30', 'C_h32_cap10'
    paired = {}
    for name, scenarios in [('identified', ['identified']), ('nominal', ['nominal']),
                           ('five_nonreference_mean', ['nominal', *q.plan['evaluation']['remaining_scenarios']])]:
        values = []
        for seed in q.plan['seeds']:
            c = cells.get(str(seed), {})
            if all(sc in c.get(v, {}) for v in (a, b) for sc in scenarios):
                values.append(dict(seed=seed, difference_pp=statistics.mean(c[a][sc]-c[b][sc] for sc in scenarios)))
        diffs = [v['difference_pp'] for v in values]
        paired[name] = dict(n=len(values), seeds=values,
                            mean_pp=statistics.mean(diffs) if diffs else None,
                            sample_sd_pp=statistics.stdev(diffs) if len(diffs)>1 else None)
    result['paired_effects'][a+'_minus_'+b] = paired
    metrics = {}
    for metric in ('response_rmse4', 'response_rmse8', 'response_rmse32', 'q_rmse32_rad', 'qd_rmse32_rad_s'):
        values = []
        for seed in q.plan['seeds']:
            c = result['cal_responses']
            ka, kb = f's{seed}_{a}', f's{seed}_{b}'
            if ka in c and kb in c and c[kb][metric] > 0:
                values.append(dict(seed=seed, relative_change_percent=100*(c[ka][metric]/c[kb][metric]-1)))
        diffs = [x['relative_change_percent'] for x in values]
        metrics[metric] = dict(n=len(values), seeds=values,
            mean_relative_change_percent=statistics.mean(diffs) if diffs else None,
            sample_sd_percent=statistics.stdev(diffs) if len(diffs)>1 else None,
            improved_seeds=sum(x<0 for x in diffs))
    result['cap30_minus_cap10_cal_response'] = metrics
    result['primary_contrast'] = 'C_h32_cap30 versus C_h32_cap10 on the same 5080'
    result['all_five_seeds_full_primary_policy_cells'] = paired['five_nonreference_mean']['n'] == 5
    result['interpretation'] = ('Development intervention; cap is measured before common clipping/Muon, not the actual '
        'parameter-step fraction. No cross-GPU pooling, no selecting favorable seeds/conditions; '
        'cal response improvement alone is not policy superiority. Partial/blocked is not zero.')
    queue.atomic_json(path, result)


def run_work(q):
    # Preserve all five seeds even when early results are unfavorable.
    for seed in q.plan['seeds']:
        queue.train_seed(q, seed)
    # Finish the six-condition primary comparison before spending budget on D.
    for scenario in q.plan['evaluation']['remaining_scenarios']:
        for seed in q.plan['seeds']:
            for c in q.plan['machines']['5080']['candidates']:
                queue.evaluate(q, seed, c['name'], scenario)
            summarize(q)
    for seed in q.plan['seeds']:
        queue.window_seed(q, seed)
        if q.state['windows'].get(str(seed), {}).get('status') == 'completed':
            for scenario in q.plan['evaluation']['remaining_scenarios']:
                for variant in ('D', 'C_perm', 'C_fixed32', 'C_fixed8'):
                    queue.evaluate(q, seed, variant, scenario)
                summarize(q)
    q.state['status'] = ('completed_with_blocked_windows' if any(
        x['status'].startswith('blocked') for x in q.state['windows'].values()) else 'completed')


def check_inputs(plan, inbox):
    missing = []
    for name in (*queue.ADDON_FILES, *ADDITIONS, 'release/ral/params.csv', '.venv/bin/python'):
        if not (ROOT/name).is_file(): missing.append(name)
    for seed in plan['seeds']:
        path = inbox/f"{plan['source_session']}_s{seed}.tar.gz"
        if not path.is_file(): missing.append(str(path))
        elif queue.digest(path) != plan['source_archive_sha256'][str(seed)]:
            raise ValueError(f'Original parent handoff hash differs: {path}')
    motions = list((ROOT/'data/motions/z01_lafan_v2_clean').glob('*.npz'))
    if len(motions) != 163: missing.append(f'real motion files: expected 163, found {len(motions)}')
    if missing: raise FileNotFoundError('Required inputs missing:\n'+'\n'.join(missing))
    print('输入检查通过：5 份原始交接包哈希正确，163 个动作文件及依赖入口存在。', flush=True)
    print('启动时还会校验源代码、参数、动作内容和 GPU；本次检查未占用 GPU。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--machine', choices=('5080',), default='5080')
    parser.add_argument('--session', default='cd03_strength')
    parser.add_argument('--check-inputs', action='store_true', help='Verify local inputs without GPU use')
    parser.add_argument('--inbox', type=Path, default=ROOT/'logs/ral_cd/cd01_5080/handoff')
    parser.add_argument('--continue-session', action='store_true')
    parser.add_argument('--status', action='store_true')
    known = parser.parse_args()
    plan = queue.read(CONFIG)
    validate_plan(plan)
    check_dependencies(plan)
    if known.check_inputs:
        check_inputs(plan, known.inbox.resolve())
        return 0
    sys.argv = [sys.argv[0], '--machine', '5080', '--session', known.session, '--inbox', str(known.inbox)]
    if known.continue_session: sys.argv.append('--continue-session')
    if known.status: sys.argv.append('--status')
    # Changes are local to this process. Imported frozen files are never edited.
    queue.CONFIG = CONFIG
    queue.ADDON_FILES = (*queue.ADDON_FILES, *ADDITIONS)
    queue.training_plan = training_plan
    queue.training_diagnostics = diagnostics
    queue.base.verify_training = verify_training
    queue.summarize = summarize
    queue.run_work = run_work
    return queue.main()


if __name__ == '__main__':
    raise SystemExit(main())
