#!/usr/bin/env python3
"""Metric recalculation template for DI-2026-0099 v15.

Input: one row per independent hardware trial after raw encoder, IMU, command,
and current streams have been aligned and summarized over the analysis window.
This script aggregates trial-level metrics; it intentionally does not treat
control-loop samples within one trial as independent observations.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import pandas as pd
try:
    from scipy import stats
except Exception:
    stats = None

def ci_half_width(values, alpha=0.05):
    vals = pd.Series(values).dropna().astype(float)
    n = len(vals)
    if n <= 1:
        return float('nan')
    sd = vals.std(ddof=1)
    tcrit = stats.t.ppf(1-alpha/2, n-1) if stats is not None else (2.262 if n == 10 else 1.96)
    return float(tcrit * sd / math.sqrt(n))

def percent_reduction(dr_mean, hybrid_mean):
    return (dr_mean - hybrid_mean) / dr_mean * 100.0 if dr_mean else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trial_csv', type=Path)
    ap.add_argument('--out', type=Path, default=Path('hardware_statistics_recomputed_v15.csv'))
    args = ap.parse_args()
    df = pd.read_csv(args.trial_csv)
    metrics = ['peak_current_A','pitch_RMS_deg','pitch_variance_deg2','pitch_peak_deg','joint_position_RMSE_rad','current_chattering_index_A_per_s']
    rows = []
    for metric in metrics:
        row = {'metric': metric}
        means = {}
        for method, sub in df.groupby('method'):
            vals = sub[metric].dropna().astype(float)
            means[method] = vals.mean()
            row[f'{method}_mean'] = vals.mean()
            row[f'{method}_sd'] = vals.std(ddof=1)
            row[f'{method}_ci_half_width'] = ci_half_width(vals)
        if 'DR-only' in means and 'Hybrid-RTI' in means:
            row['reduction_percent'] = percent_reduction(means['DR-only'], means['Hybrid-RTI'])
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'Wrote {args.out}')

if __name__ == '__main__':
    main()
