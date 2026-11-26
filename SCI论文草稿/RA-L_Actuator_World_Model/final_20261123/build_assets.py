"""Rebuild the scientific plot from the bundled audit (no training or network)."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/'evidence/online_night01_review_20261123.json').read_text())
plt.rcParams.update({'font.family':'serif', 'font.serif':['DejaVu Serif'],
    'font.size':9, 'axes.labelsize':9, 'axes.titlesize':10,
    'pdf.fonttype':42, 'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7.05,2.65),sharex=True,sharey=True)
for ax,machine in zip(axes,['4090','5080']):
    pairs=data['machines'][machine]['pairs']
    for p in pairs:
        x=p['prediction_diagnostics']['cal']['q32_B_reduction_percent']
        y=p['primary_B_minus_Aplus_completion_pp']
        failed=not p['gate']['passed']
        ax.scatter(x,y,s=38 if failed else 24,marker='x' if failed else 'o',
                   color='#A83D31' if failed else '#205F7B',linewidths=1.5 if failed else .5,zorder=3)
        if failed:ax.annotate(f"seed {p['seed']}",(x,y),xytext=(-4,9),textcoords='offset points',ha='right',fontsize=8)
    ax.axhline(0,color='.5',linewidth=.7);ax.axvline(0,color='.5',linewidth=.7)
    ax.set_title(f"RTX {machine}: {len(pairs)} paired training seeds")
    ax.set_xlabel('B reduction in calibration angle RMSE (%)')
    ax.set_xlim(-12,16);ax.set_ylim(-8,9);ax.set_yticks([-8,-4,0,4,8])
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('B minus A+ completion\n(percentage points)')
fig.tight_layout(pad=.6,w_pad=1.4)
fig.savefig(ROOT/'figures/prediction_control.pdf')
fig.savefig(ROOT/'figures/prediction_control.png',dpi=200)
plt.close(fig)
