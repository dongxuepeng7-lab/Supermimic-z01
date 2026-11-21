import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

root = Path(__file__).resolve().parents[1]
out = root / "RA-L_Actuator_World_Model" / "figures"
out.mkdir(exist_ok=True)
source = next((root / "paper_figures_20261121").rglob("fig03_heldout_trajectories.svg"))
mpl.rcParams.update({"font.family":"DejaVu Sans", "svg.fonttype":"none", "pdf.fonttype":42,
                     "path.simplify":False, "axes.unicode_minus":True, "font.size":8,
                     "axes.linewidth":0.6, "svg.hashsalt":"joint-figures"})

def export(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", facecolor="white", metadata={"Creator":"Matplotlib"})
    plt.close(fig)

def flow():
    w, h = 530, 157
    f = plt.figure(figsize=(w/72,h/72))
    a = f.add_axes([0,0,1,1], xlim=(0,w), ylim=(h,0))
    a.axis("off")
    ink, teal, gray = "#172d3a", "#00685e", "#425963"
    def label(x,y,s,size=8,weight="normal",color=ink):
        return a.text(x,y,s,fontsize=size,fontweight=weight,color=color,va="center")
    def box(x,y,ww,hh,title,lines,fill,edge):
        a.add_patch(Rectangle((x,y),ww,hh,facecolor=fill,edgecolor=edge,lw=.75))
        label(x+6,y+10,title,8.8,"bold",ink)
        for i,s in enumerate(lines):
            label(x+6,y+23+i*10.5,s,7.8)
    def arrow(x,y,x2,y2):
        a.annotate("",(x2,y2),(x,y),arrowprops=dict(arrowstyle="-|>",color=gray,lw=.95,shrinkA=0,shrinkB=0,mutation_scale=6))
    label(3,7,"OFFLINE IDENTIFICATION",7.1,"bold",teal)
    label(137,7,"SIMULATION AND POLICY TRAINING",7.1,"bold",teal)
    for x,ww,t,lines in [
        (3,119,"Joint identification",["Recorded joint commands","and feedback events","Fit actuator parameters"]),
        (137,117,"Physics simulator",["Nominal or identified","actuators; collect states","and paired perturbations"]),
        (269,118,"World-model fit",["Actuator + body dynamics","Trajectory loss; optional","multi-step response loss"]),
        (402,125,"Policy update",["Roll out the world model","Backpropagate return","to the actor"])
    ]:
        box(x,17,ww,54,t,lines,"#eaf4f1" if x==3 else "#f2f6f8",teal if x==3 else "#73909b")
    for start,end in [(122,137),(254,269),(387,402)]:
        arrow(start,42,end-1,42)
    a.plot([434,434,192],[71,86,86],color=gray,lw=.95)
    arrow(192,86,192,72)
    label(219,80,"Updated actor collects new transitions",7.4,color=gray)
    arrow(47,71,47,111); arrow(500,71,500,111)
    label(53,97,"Held-out sessions",7.4,color=gray)
    label(439,97,"Frozen actor",7.4,color=gray)
    box(3,112,250,40,"Offline replay on hardware records",
        ["Replay identical commands from the recorded initial state.","Score joint position, velocity and feedback torque."],"#ffffff","#83969e")
    box(269,112,258,40,"Closed-loop evaluation in physics simulation",
        ["Use shared reference and held-out actuator settings.","Score task completion, tracking and early termination."],"#ffffff","#83969e")
    export(f,"training_pipeline")

def trajectories():
    # Use the original vector vertices and tick transforms. These paths are
    # for display only. Full-trajectory scores come from the archived CSV.
    xml = ET.parse(source)
    g = xml.find(".//*[@id='axes_2']")
    assert "L hip roll" in "".join(g.itertext())
    def tick_transform(axis):
        ticks=[]
        for t in g.iter():
            if t.get("id","").startswith(axis+"tick_"):
                loc=t.find(".//{*}use")
                val=t.find(".//{*}text")
                ticks.append((float(loc.get(axis)),float(val.text.replace("−","-"))))
        px,data=np.array(ticks).T
        line=np.polyfit(px,data,1)
        assert np.allclose(np.polyval(line,px),data,atol=1e-6)
        return line
    tx,ty=tick_transform("x"),tick_transform("y")
    traces=[]
    for line in g:
        if line.get("id","").startswith("line2d_"):
            d=line[0].attrib["d"]
            assert set(re.findall("[A-Za-z]",d)) <= {"M","L"}
            p=np.array(re.findall(r"[ML]\s+([-0-9.]+)\s+([-0-9.]+)",d),float)
            x,y=np.polyval(tx,p[:,0]),np.polyval(ty,p[:,1])
            assert abs(x[0])<1e-5 and abs(x[-1]-9)<.01 and len(x)>20
            traces.append((x,y))
    assert len(traces)==3
    with source.with_name("figure_metrics.csv").open() as file:
        scores={r["parameter_set"]:float(r["q"])*1000 for r in csv.DictReader(file) if r["joint"]=="L hip roll"}
    f,axs=plt.subplots(1,2,sharey=True,figsize=(3.48,1.78))
    f.subplots_adjust(left=.164,right=.985,top=.78,bottom=.30,wspace=.12)
    colors=["#9b4d1a","#006e65"]
    for i,(a,key,title) in enumerate(zip(axs,["original_model","current_rounded"],["Original parameters","Identified parameters"])):
        a.plot(*traces[0],lw=1.35,color="#18252e")
        a.plot(*traces[1+i],lw=.95,color=colors[i],ls=(0,(3,1.35)))
        a.set_title(title,fontsize=7.9,fontweight="bold",pad=4)
        ymin=min(y.min() for _,y in traces)
        a.set(xlim=(-.1,9.1),ylim=(ymin-.003,-.013),xticks=[0,3,6,9],yticks=[-.08,-.06,-.04,-.02])
        a.grid(axis="y",color="#d1d9da",lw=.45)
        a.tick_params(labelsize=7.0,length=2.1,pad=1.7)
        a.spines[["top","right"]].set_visible(False)
        a.text(.5,-.47,f"RMSE  {scores[key]:.2f} mrad",ha="center",va="center",fontsize=7.0,transform=a.transAxes)
    axs[0].set_ylabel("Position (rad)",fontsize=7.6,labelpad=1)
    f.text(.57,.146,"Time from first scored sample (s)",ha="center",fontsize=7.5)
    keys=[Line2D([0],[0],color=c,lw=1.3,ls=ls) for c,ls in [("#18252e","-"),(colors[0],"--"),(colors[1],"--")]]
    f.legend(keys,["Measured","Original","Identified"],ncol=3,loc="upper center",bbox_to_anchor=(.54,1.035),
             fontsize=7.1,frameon=False,columnspacing=1.0,handlelength=1.65,handletextpad=.35)
    export(f,"joint_replay_before_after")

flow()
trajectories()
