"""Publication figures from existing frozen replays; no fitting or robot writes.
Run from repository: .venv/bin/python analysis/paper_figures_20261121/make_figures.py
"""
from pathlib import Path
import json, hashlib, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
A=ROOT/'analysis'
SOURCES={}
def track(p):
    p=Path(p)
    if str(p) not in SOURCES:
        SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return p
def nz(p):
    with np.load(track(p)) as z: return {k:z[k] for k in z.files}
def js(p): return json.loads(track(p).read_text())

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
 'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
 'lines.linewidth':1.1,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'savefig.dpi':300})
BLUE='#0072B2'; ORANGE='#D55E00'; GREY='#777777'; BLACK='#222222'
def save(fig,name):
    for ext in ['png','pdf','svg']: fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',facecolor='white')
    plt.close(fig)
def grid(ax): ax.grid(alpha=.18,linewidth=.5)

# Comparisons are within each joint/record, never an average over incompatible runs.
specs=[('L knee','left_knee_scale03_T2_validation_20261120','0.3 / 2 s'),
 ('R knee','right_knee_scale03_T2_validation_20261120','0.3 / 2 s'),
 ('L hip pitch','left_hip_pitch_scale03_T2_validation_20261120','0.3 / 2 s'),
 ('R hip pitch','right_hip_pitch_scale03_T2_validation_20261120','0.3 / 2 s'),
 ('L hip roll','left_hip_roll_scale03_T3_validation_20261120','0.3 / 3 s'),
 ('R hip yaw','right_hip_yaw_scale01_T2_validation_20261120','0.1 / 2 s'),
 ('R ankle pitch','right_ankle_pitch_scale03_T3_validation_20261120','0.3 / 3 s'),
 ('L ankle roll','left_ankle_roll_scale03_T2_validation_20261120','0.3 / 2 s')]
records=[]
for label,directory,run in specs:
    d=A/directory
    if label=='R hip yaw':
        coarse=d/'full_original_right_baseline.npz'; bf=d/'fine_original_right_baseline.npz'; cf=d/'fine_right_candidate.npz'; cand='right_candidate'
    elif label=='R ankle pitch':
        coarse=d/'full_scale0.3_T3_baseline_0.002.npz';bf=d/'full_scale0.3_T3_baseline_0.0005.npz';cf=d/'full_scale0.3_T3_right_candidate_0.0005.npz';cand='right_candidate'
    else:
        coarse=d/'runtime_limits_baseline_0.002.npz';bf=d/'runtime_limits_baseline_0.0005.npz';cf=d/'runtime_limits_current_rounded_0.0005.npz';cand='current_rounded'
    ref=nz(coarse); t=ref['time'];mask=ref['score_mask'].astype(bool)
    models=[]
    for f in [bf,cf]:
        x=nz(f)
        assert t[0]>=x['time'][0]-1e-8 and t[-1]<=x['time'][-1]+1e-8
        out={k:np.interp(t,x['time'],x[k]) for k in ['q_sim','v_sim','tau_sim']}
        met={k:float(np.sqrt(np.mean((out[k+'_sim'][mask]-ref[k+'_real'][mask])**2))) for k in ['q','v','tau']}
        assert all(np.isfinite(v) for v in met.values())
        models.append((out,met))
    report=js(d/'validation.json')
    for idx,name in enumerate(['original_right_baseline' if label=='R hip yaw' else 'baseline',cand]):
        if label=='R hip yaw':
            row=next(r for r in report['fine_records'] if r['candidate']==name)
            expected=row['metrics_at_original_2ms_scoring_times']
        else:
            row=next(r for r in report['rows'] if r['candidate']==name and r['max_step_sec']==.0005 and r.get('mode','runtime_limits')=='runtime_limits')
            expected=row['metrics_same_2ms_grid']
        for key,long in [('q','position_rmse_rad'),('v','velocity_rmse_rad_s'),('tau','torque_rmse_Nm')]:
            assert np.isclose(models[idx][1][key],expected[long],rtol=1e-7,atol=1e-10),(label,key)
    records.append(dict(label=label,directory=directory,run=run,candidate=cand,ref=ref,models=models))

# Fig 1: timestamp evidence, not a reconstruction of motor execution times.
raw=nz(A/'right_ankle_pitch_scale03_T3_validation_20261120/scale0.3_T3.npz')
r=records[6]; lo=r['ref']['time'][0];hi=r['ref']['time'][-1]
ct=raw['ct'];ft=raw['st'];c=ct[(ct>=lo)&(ct<=hi)];f=ft[(ft>=lo)&(ft<=hi)]
ci=np.searchsorted(ft,c,side='left')-1
ages=(c-ft[ci])*1000
timing={'record':'right_ankle_pitch_scale0.3_T3','interval_s':[float(lo),float(hi)],
 'command_hz':float((len(c)-1)/(c[-1]-c[0])),'feedback_hz':float((len(f)-1)/(f[-1]-f[0])),
 'command_max_interval_ms':float(np.diff(c).max()*1000),'feedback_max_interval_ms':float(np.diff(f).max()*1000),
 'feedback_age_mean_ms':float(ages.mean()),'feedback_age_p99_ms':float(np.quantile(ages,.99)),
 'age_assumption':'Recorded receipt timestamps; zero extra command delay; strict earlier feedback on ties; not physical sensor age.'}
fig,ax=plt.subplots(2,2,figsize=(7.2,4.6),layout='constrained')
origin=lo+3
for tt,y,col in [(ct,1,BLUE),(ft,0,ORANGE)]:
    a=tt[(tt>=origin)&(tt<=origin+.05)]-origin
    ax[0,0].vlines(a*1000,y-.18,y+.18,color=col)
ax[0,0].set(yticks=[0,1],yticklabels=['Feedback','Command'],xlabel='Time within fixed 50 ms window (ms)',ylim=(-.5,1.5),title='(a) Recorded asynchronous events')
for tt,col,label in [(c,BLUE,'Command'),(f,ORANGE,'Feedback')]:
    x=np.sort(np.diff(tt)*1000); ax[0,1].plot(x,np.arange(1,len(x)+1)/len(x),color=col,label=label)
ax[0,1].set(xlabel='Inter-event interval (ms)',ylabel='Empirical CDF',title='(b) All intervals in replay span',xlim=(0,23));ax[0,1].legend()
ax[1,0].hist(ages,bins=np.linspace(0,max(ages)*1.001,65),weights=np.ones(len(ages))*100/len(ages),color=BLUE)
ax[1,0].set(xlabel='Recorded feedback age at command (ms)',ylabel='Commands (%)',title='(c) Latest strictly earlier feedback')
for tt,col,label in [(c,BLUE,'Command'),(f,ORANGE,'Feedback')]:
    ax[1,1].plot(tt[1:]-lo,np.diff(tt)*1000,'.',ms=.5,alpha=.45,color=col,label=label,rasterized=True)
ax[1,1].set(xlabel='Time from replay start (s)',ylabel='Inter-event interval (ms)',title='(d) Full-span timing irregularity')
for a in ax.flat:grid(a)
save(fig,'fig01_recorded_timing')

# Fig 2: include failed validation, no invented confidence intervals.
fig,axes=plt.subplots(1,3,figsize=(7.2,4.4),layout='constrained')
y=np.arange(len(records));labels=[x['label']+'\n'+x['run'] for x in records]
for ax,key,title in zip(axes,['q','v','tau'],['Position','Velocity','Torque']):
    ratio=np.array([r['models'][1][1][key]/r['models'][0][1][key] for r in records])
    ax.barh(y,ratio,color=[BLUE if z<1 else ORANGE for z in ratio],height=.6)
    ax.axvline(1,color=BLACK,ls='--',lw=.8)
    for yy,rr in zip(y,ratio):ax.text(rr+.025,yy,f'{rr:.2f}',va='center',fontsize=7,bbox={'facecolor':'white','edgecolor':'none','pad':.2})
    ax.set(yticks=y,yticklabels=labels if key=='q' else [],title=title,xlabel='RMSE / original-model RMSE',xlim=(0,max(1.4,ratio.max()+.22)))
    ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
fig.suptitle('Independent recorded trajectories: parameter-set comparison',fontsize=10)
save(fig,'fig02_holdout_error_ratios')

# Fig 3: deterministic window = first three scored periods, not selected by best fit.
fig,axes=plt.subplots(3,3,figsize=(7.2,6.0),layout='constrained')
for col,idx in enumerate([0,4,6]):
    r=records[idx];ref=r['ref'];t=ref['time'];T=2 if idx==0 else 3
    start=t[ref['score_mask'].astype(bool)][0];sel=(t>=start)&(t<=start+3*T);xx=t[sel]-start
    for row,key in enumerate(['q','tau']):
        a=axes[row,col];a.plot(xx,ref[key+'_real'][sel],color=BLACK,label='Measured',lw=.9)
        for (out,met),color,label in zip(r['models'],[GREY,BLUE],['Original model','Identified set']):
            a.plot(xx,out[key+'_sim'][sel],color=color,label=label,lw=1,ls='--' if color==GREY else '-')
        a.set_ylabel('Position (rad)' if key=='q' else 'Torque (N m)');grid(a)
    a=axes[2,col]
    for (out,met),color,label in zip(r['models'],[GREY,BLUE],['Original model','Identified set']):
        a.plot(xx,(out['q_sim'][sel]-ref['q_real'][sel])*1000,color=color,label=label)
    a.axhline(0,color=BLACK,lw=.5);a.set(xlabel='Time from first scored sample (s)',ylabel='Position residual (mrad)');grid(a)
    axes[0,col].set_title(r['label']+' | '+r['run'])
axes[0,0].legend(fontsize=6,loc='best')
save(fig,'fig03_heldout_trajectories')

# Fig 4: retain limit mismatch and physical shaking, without claiming root cause.
fig,ax=plt.subplots(2,2,figsize=(7.2,4.8),layout='constrained')
r=records[7];ref=r['ref'];t=ref['time'];start=t[ref['score_mask'].astype(bool)][0];s=(t>=start)&(t<=start+6)
ax[0,0].plot(t[s]-start,ref['q_real'][s],color=BLACK,label='Measured')
ax[0,0].plot(t[s]-start,r['models'][1][0]['q_sim'][s],color=BLUE,label='Identified set')
ax[0,0].axhline(-.25,color=ORANGE,ls='--',label='XML lower limit')
ax[0,0].set(title='(a) L ankle roll: range mismatch',xlabel='Time from first scored sample (s)',ylabel='Position (rad)');ax[0,0].legend(fontsize=6)
j=js(A/r['directory']/'validation.json')
for ix,mode in enumerate(['runtime_limits','no_roll_limit_diagnostic']):
    vals=[]
    for name in ['baseline','current_rounded']:
        row=next(v for v in j['rows'] if v['mode']==mode and v['candidate']==name and v['max_step_sec']==.0005)
        vals.append(row['metrics_same_2ms_grid']['position_rmse_rad']*1000)
    for offset,v,color,label in zip([0,1],vals,[GREY,BLUE],['Original model','Identified set']):
        ax[1,0].bar(offset+ix*3,v,color=color,width=.7,label=label if ix==0 else None)
    for xx,v in zip(np.array([0,1])+ix*3,vals):ax[1,0].text(xx,v+.15,f'{v:.2f}',ha='center',fontsize=7)
ax[1,0].set(xticks=[.5,3.5],xticklabels=['XML limits','Limit disabled\n(diagnostic only)'],ylabel='Full-span position RMSE (mrad)',title='(b) Limit removal alone is insufficient',ylim=(0,11))
ax[1,0].legend(fontsize=6,loc='upper right')
sel=(raw['st']>=lo)&(raw['st']<=hi)
peak=raw['st'][np.flatnonzero(sel)[np.argmax(np.abs(raw['q'][sel,11]-np.median(raw['q'][sel,11])))]]
win=(raw['st']>=peak-.4)&(raw['st']<=peak+.6)
for idx,col,label in [(10,BLUE,'Pitch'),(11,ORANGE,'Roll')]:
    ax[0,1].plot(raw['st'][win]-peak,raw['q'][win,idx],color=col,label=label,lw=.8)
ax[0,1].set(title='(c) R ankle: largest roll excursion',xlabel='Time relative to maximum roll excursion (s)',ylabel='Measured position (rad)');ax[0,1].legend()
win2=(raw['ct']>=peak-.4)&(raw['ct']<=peak+.6)
ax[0,1].plot(raw['ct'][win2]-peak,raw['cq'][win2,11],ls=':',color=BLACK,label='Roll command')
ax[0,1].legend(fontsize=6)
for idx,col,label in [(10,BLUE,'Pitch'),(11,ORANGE,'Roll')]:ax[1,1].plot(raw['st'][win]-peak,raw['tau'][win,idx],color=col,label=label,lw=.8)
ax[1,1].set(title='(d) Joint-side feedback torque',xlabel='Time relative to maximum roll excursion (s)',ylabel='Torque (N m)')
ax[1,1].legend(fontsize=6,loc='upper right')
for a in ax.flat:grid(a)
save(fig,'fig04_failure_modes')

# Fig 5: explicitly historical runtime snapshot, with right-ankle candidate overlay.
summary=js(A/'joint_parameter_summary_20261120/summary.json')
rows=summary['rows']; labels=[r['joint'].replace('left_','L ').replace('right_','R ').replace('_',' ') for r in rows]
fig,axes=plt.subplots(5,1,figsize=(7.2,7.5),sharex=True,layout='constrained')
keys=['armature','damping','frictionloss','load_offset','command_delay_sec']
units=['Armature\n(kg m²)','Damping\n(N m s/rad)','Friction\n(N m)','Load bias L\n(N m)','Command delay\n(ms)']
candidate=[.1249,0,.7259,-.1324,0]
ri=next(i for i,r in enumerate(rows) if r['joint']=='right_ankle_pitch')
for ax,key,unit,val in zip(axes,keys,units,candidate):
    factor=1000 if key=='command_delay_sec' else 1
    yy=[r['runtime'][key]*factor for r in rows]
    ax.bar(np.arange(len(rows)),yy,color=GREY,width=.55,label='Runtime snapshot (20 Nov 2026)')
    ax.scatter([ri],[val*factor],color=BLUE,marker='D',s=22,zorder=3,label='R ankle pitch candidate (not deployed)')
    ax.axhline(0,color=BLACK,lw=.5);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.15)
axes[0].legend(loc='upper right',fontsize=6)
axes[-1].set(xticks=np.arange(len(rows)),xticklabels=labels)
plt.setp(axes[-1].get_xticklabels(),rotation=50,ha='right')
fig.suptitle('Parameter inventory: deployment is not independent validation',fontsize=10)
save(fig,'fig05_parameter_inventory')

export=[]
for r in records:
    for name,(_,metrics) in zip(['original_model',r['candidate']],r['models']):
        export.append({'joint':r['label'],'run_scale_period':r['run'],'parameter_set':name,'source_directory':r['directory'],**metrics})
with (OUT/'figure_metrics.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(export[0]));w.writeheader();w.writerows(export)
(OUT/'timing_statistics.json').write_text(json.dumps(timing,indent=2)+'\n')
(OUT/'source_hashes.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
(OUT/'checks.json').write_text(json.dumps({'rmse_recomputed_and_matched_existing_reports':True,'comparisons':len(records),'metrics_per_parameter_set':3,'max_step_seconds':.0005,'scoring_grid_seconds':.002,'all_transients_retained_in_metrics':True,'synthetic_or_refitted_results':False,'trajectory_window_rule':'first three scored periods','failure_window_rule':'largest absolute roll deviation from full replay-span median','actual_execution_timestamps_available':False},indent=2)+'\n')
print(json.dumps({'output':str(OUT),'figures':5,'exports':15,'source_files':len(SOURCES),'timing':timing},indent=2))
