import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const root=process.cwd();
const build=path.join(root,'_ppt_build');
const out=path.join(root,'output');
const skill='C:/Users/dongx/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const python='C:/Users/dongx/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/dongx/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const p=Presentation.create({slideSize:{width:1280,height:720}});
p.theme.defaultFont='Arial';
const C={navy:'#10283F',ink:'#183348',muted:'#536878',teal:'#007F88',amber:'#B76A19',light:'#DDE8ED',white:'#FFFFFF',pale:'#F3F7F9',gray:'#9BAEBB',blue:'#4B708E'};
p.theme.colorScheme={name:'SG-AMP Academic',themeColors:{accent1:C.teal,accent2:C.blue,accent3:C.amber,accent4:'#99536B',accent5:'#6C6997',accent6:C.gray,bg1:C.white,bg2:C.pale,tx1:C.ink,tx2:C.muted,dk1:'#000000',dk2:C.navy,lt1:C.white,lt2:C.light,hlink:C.teal,folHlink:'#6C6997'}};
let count=0;const scripts=[];
function text(s,t,x,y,w,h,size=28,color=C.ink,bold=false,align='left'){
 const sh=s.shapes.add({geometry:'textbox',name:t.slice(0,45),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=t;sh.text.style={fontSize:size,typeface:'Arial',color,bold,alignment:align,verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0};return sh;
}
function slide(title,{dark=false,backup=false}={}){
 const s=p.slides.add();s.background.fill=dark?C.navy:C.white;count++;
 if(title) text(s,title,64,48,1152,105,44,dark?C.white:C.navy,true);
 text(s,backup?'SG-AMP   /   BACKUP':'SG-AMP',64,677,1000,22,15,dark?'#AFCAD8':C.muted);
 text(s,backup?`A${count-12}`:String(count).padStart(2,'0'),1160,675,56,26,17,dark?'#AFCAD8':C.muted,false,'right');
 return s;
}
async function img(s,file,x,y,w,h){const b=await fs.readFile(path.join(root,file));return s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:file.endsWith('.png')?'image/png':'image/jpeg',alt:`Paper figure: ${file}`,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function note(s,title,seconds,script,source,extra=''){
 const n=`SUGGESTED TIME: ${seconds} seconds\n\nSPEAKING SCRIPT\n${script}\n\nSOURCE\nSGAMP.pdf, ${source}. Local paper supplied by the user.\n${extra}`;
 s.speakerNotes.text=n;scripts.push({title,seconds,script,extra});
}
function caption(s,t){text(s,t,64,636,1152,31,17,C.muted);}
function chart(s,{x,y,w,h,categories,series,max=1,format='0.0%',legend=true,unit=.2}){
 const ch=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories,series,hasLegend:legend,legend:{position:'bottom',textStyle:{fontSize:21,fill:C.ink}},barOptions:{direction:'column',grouping:'clustered',gapWidth:110},chartFill:C.white,plotAreaFill:C.white,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},xAxis:{textStyle:{fontSize:23,fill:C.ink},line:{fill:C.gray,width:1},majorGridlines:null},yAxis:{min:0,max,majorUnit:unit,numberFormatCode:format,textStyle:{fontSize:18,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.light,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:23,fill:C.ink,bold:true}}});
 applyPresentationChartFont(ch,{fontFamily:'Arial'});return ch;
}
function table(s,values,x,y,width,height,colWidths,font=25){
 const tb=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:colWidths,values});
 tb.borders.assign({fill:C.light,width:.7});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=tb.getCell(r,c);cell.fill=r===0?C.navy:(r===values.length-1?'#E6F2F2':C.white);
  cell.text.style={fontSize:font,typeface:'Arial',color:r===0?C.white:C.ink,bold:r===0||r===values.length-1,alignment:c===0?'left':'center',verticalAlignment:'middle',insets:12};
 }
 return tb;
}

// 1. Minimal cover; exact paper title and author order.
{
 const s=slide('',{dark:true});
 text(s,'SG-AMP',64,92,1152,110,88,'#69CCD0',true);
 text(s,'Stability-Gated Adversarial Motion Priors\nwith Phase-Conditioned Discriminator\nfor Natural and Robust Bipedal Locomotion',64,238,1152,200,42,C.white,true);
 text(s,'Jiahong Li   ·   Xuepeng Dong   ·   Shengli Xu   ·   Beiyan Jiang*',64,510,1152,38,25,C.white);
 text(s,'Beijing Union University\nBeijing Zoyi Technology Co., Ltd.',64,562,1152,64,23,'#AFCAD8');
 note(s,'SG-AMP',30,'Good morning, everyone. Today I will present SG-AMP, a method for learning natural and robust bipedal locomotion. This is joint work by Jiahong Li, Xuepeng Dong, Shengli Xu, and Beiyan Jiang. Our main idea is to regulate motion imitation using a physical stability signal. I will first explain the motivation, then introduce the method, and finally discuss our simulation results.','title and abstract');
}
// 2. Source locomotion image anchors the research question.
{
 const s=slide('Natural gait and stable contact');
 text(s,'AMP learns motion style from reference trajectories.\nA high style reward alone does not ensure stable foot contact.',64,150,1152,88,30);
 await img(s,'fig7.jpg',64,260,850,376);
 text(s,'Physics-blind\nstyle reward',950,278,260,76,27,C.teal,true);
 text(s,'Within-cycle\nreward fluctuation',950,396,270,76,27,C.teal,true);
 text(s,'Fixed task–style\nweight',950,514,270,76,27,C.teal,true);
 note(s,'Natural gait and stable contact',45,'Reinforcement learning can produce robust walking, but the resulting gait may be stiff or excessively bent at the knees. Adversarial Motion Priors, or AMP, address naturalness by rewarding transitions that resemble reference motion. However, this introduces three difficulties. First, the style score does not directly describe contact stability. Second, a discriminator without gait phase can provide inconsistent feedback within a walking cycle. Third, a fixed style weight creates a difficult tradeoff when balance becomes challenging. The snapshots here show the SG-AMP gait. Our research question is how to preserve useful motion style while adapting its influence to the robot’s current stability.','Introduction and Fig. 4; source image fig7.jpg');
}
// 3. Flat, editable overview with mathematically defined integration.
{
 const s=slide('SG-AMP: three changes to AMP');
 const rows=[['G1','CoP-gated style reward','Scale imitation using the current stability margin.'],['G2','Phase-conditioned discriminator','Evaluate motion with its position in the gait cycle.'],['G3','Adaptive style weight','Increase imitation as average stability improves.']];
 rows.forEach((r,i)=>{const yy=170+i*117;text(s,r[0],64,yy,90,48,34,C.teal,true);text(s,r[1],180,yy,1030,48,32,C.navy,true);text(s,r[2],180,yy+48,1030,44,26,C.muted);});
 text(s,'rₜ = r_task + λₜ · g(mₜ) · r_AMP',64,563,1152,64,40,C.teal,true);
 caption(s,'PPO updates the policy. Reference transitions and policy rollouts train the discriminator.');
 note(s,'SG-AMP: three changes to AMP',50,'SG-AMP keeps the standard AMP and PPO training structure, and changes three specific parts. G1 multiplies the style reward by a gate computed from the Center of Pressure, or CoP. G2 adds a gait phase encoding to the discriminator. G3 adapts the style weight using a moving average of the gate. In the objective at the bottom, the task reward still encourages locomotion. The AMP style reward is scaled by both the instantaneous gate and the adaptive weight. These two factors operate at different time scales. The design uses contact forces and foot geometry already available in the physics backend, and each component can be removed independently for ablation.','Section III, Eqs. (5)–(8) and Algorithm 1');
}
// 4. Original physical construction, preserved without redrawing.
{
 const s=slide('G1: CoP-gated style reward');
 await img(s,'fig2.jpg',64,161,900,408);
 text(s,'CoP',1000,185,210,42,31,C.teal,true);
 text(s,'Force-weighted\ncontact position',1000,235,210,86,26);
 text(s,'Margin mₜ',1000,365,210,45,31,C.teal,true);
 text(s,'Minimum signed\ndistance to the\nsupport boundary',1000,418,210,120,25);
 text(s,'g(mₜ) = σ((mₜ − m₀) / κ)',64,575,1152,58,38,C.teal,true);
 caption(s,'m₀ = 0.025 m     κ = 0.006 m     Smooth attenuation near the support boundary');
 note(s,'G1: CoP-gated style reward',65,'The first module connects style learning to contact geometry. We compute the CoP as a vertical-force-weighted average of the contact positions. The support polygon is constructed from the active foot geometry, and the margin is the minimum signed distance from the CoP to its boundary. We then map that margin to a continuous gate using a sigmoid. The threshold is 2.5 centimeters, and the smoothing scale is 6 millimeters. When the margin is large, the gate approaches one and preserves the style signal. Near the support boundary, it reduces the style contribution, allowing the task objective to have more influence. This is a multiplication on the scalar reward and does not add a new gradient path. We use the margin as a stability-related training signal. It should not be interpreted as a formal guarantee against falling.','Section III-B, Eqs. (3)–(5), Fig. 2; source image fig2.jpg');
}
// 5. Original phase diagram, with conference-sized explanation.
{
 const s=slide('G2: a phase-conditioned discriminator');
 text(s,'Contact events align the gait clock.\nReference motion and policy transitions use the same phase definition.',64,151,1152,80,28);
 await img(s,'fig3.jpg',64,245,1152,400);
 note(s,'G2: a phase-conditioned discriminator',55,'The second module addresses consistency within a gait cycle. The same joint configuration can have a different meaning during support and swing. A discriminator that ignores this context may give fluctuating style feedback. We maintain a phase clock using the commanded step frequency and align or reset it at heel-strike events. The phase enters the discriminator as cosine and sine, together with the state transition. This encoding represents a periodic variable continuously across the cycle boundary. Reference trajectories receive the same phase labels offline, so the discriminator compares motion under a consistent phase convention. The intended effect is smoother style supervision and less high-frequency joint motion. The ablation results later show how much spectral energy increases when this phase input is removed.','Section III-C, Eq. (6), Fig. 3; source image fig3.jpg');
}
// 6. Two time scales, using readable native mathematical text.
{
 const s=slide('G3: stability-driven style curriculum');
 text(s,'ḡₜ = α · g(mₜ) + (1 − α) · ḡₜ₋₁',64,169,1152,66,44,C.teal,true);
 text(s,'λₜ = λ_max · ḡₜ',64,252,1152,66,44,C.teal,true);
 text(s,'Fast response',64,385,480,44,32,C.navy,true);
 text(s,'g(mₜ) attenuates style\nat the current step.',64,443,490,94,30);
 text(s,'Slow adaptation',680,385,520,44,32,C.navy,true);
 text(s,'λₜ follows recent stability\nand adjusts imitation intensity.',680,443,536,94,30);
 text(s,'α = 0.01        λ_max = 0.5        ḡ₀ = 0',64,593,1152,40,27,C.muted);
 note(s,'G3: stability-driven style curriculum',50,'An instantaneous gate can respond to a difficult contact, but it does not by itself determine the appropriate overall intensity of imitation. The third module therefore takes an exponential moving average of the gate and uses it to set the style weight. Alpha is 0.01, the maximum weight is 0.5, and the average starts at zero. This creates a self-paced increase in imitation as the policy becomes more stable. The distinction between the two time scales is important. The fast gate reacts to the current step. The slower weight reflects recent stability. Together, they reduce the influence of style when stability is poor and allow stronger imitation when the policy can support it.','Section III-D, Eqs. (7)–(8), Algorithm 1');
}
// 7. Experimental scope and fair interpretation.
{
 const s=slide('Simulation protocol');
 text(s,'1.75 m',64,163,360,82,64,C.teal,true);
 text(s,'height',64,252,360,40,28,C.muted);
 text(s,'60.9 kg',470,163,360,82,64,C.teal,true);
 text(s,'mass',470,252,360,40,28,C.muted);
 text(s,'12 joints',873,163,343,82,64,C.teal,true);
 text(s,'6 actuated joints per leg',873,252,343,40,25,C.muted);
 text(s,'Training',64,354,245,48,32,C.navy,true);
 text(s,'Isaac Sim / PhysX, PPO, approximately 4,096 environments\nFlat terrain with randomly sampled velocity commands',340,350,876,90,28);
 text(s,'Evaluation',64,470,245,48,32,C.navy,true);
 text(s,'Zero-shot transfer to MuJoCo, 5 random seeds\nFlat walking, external pushes ×3, unseen rough terrain',340,466,876,90,28);
 caption(s,'Nominal flat-walking survival is 100% for every method. All reported results are in simulation.');
 note(s,'Simulation protocol',45,'We evaluate a self-developed biped that is 1.75 meters tall, weighs 60.9 kilograms, and has twelve actuated leg joints. Training uses PPO in Isaac Sim with the PhysX backend and approximately four thousand parallel environments. The policy trains on flat terrain with randomly sampled velocity commands. Evaluation transfers the policy to MuJoCo without additional training and uses five random seeds. We compare Pure RL, standard AMP, and SG-AMP under flat walking, stronger external pushes, and unseen rough terrain. Every method survives nominal flat walking, so that condition does not distinguish robustness. The more informative comparisons are the two stress tests and the contact and motion-quality metrics.','Section IV-A–C and Table I');
}
// 8. Editable survival chart from Table I; recalculated percentage points.
{
 const s=slide('Robustness improves in both stress tests');
 chart(s,{x:58,y:160,w:865,h:450,categories:['External pushes ×3','Rough terrain'],series:[{name:'Pure RL',values:[.435,.245],fill:C.gray},{name:'Standard AMP',values:[.462,.218],fill:C.blue},{name:'SG-AMP',values:[.736,.386],fill:C.teal}]});
 text(s,'+27.4 pp',950,209,266,70,44,C.teal,true);
 text(s,'Push survival\nvs. standard AMP',950,284,266,87,26);
 text(s,'+16.8 pp',950,420,266,70,44,C.teal,true);
 text(s,'Rough-terrain survival\nvs. standard AMP',950,495,266,87,26);
 caption(s,'Table I means, 5 seeds. Full mean ± standard deviation in backup A1. pp = percentage points.');
 note(s,'Robustness improves in both stress tests',65,'This slide shows the central robustness result. The bars report the mean survival rates from Table One. Under stronger external pushes, standard AMP achieves 46.2 percent survival, while SG-AMP reaches 73.6 percent. That is an increase of 27.4 percentage points, or approximately 59.3 percent relative to AMP. On unseen rough terrain, survival increases from 21.8 percent to 38.6 percent. That is 16.8 percentage points, or about 77.1 percent relative improvement. Pure RL is also below SG-AMP in both protocols. The full standard deviations are provided in the backup table. These results support improved robustness in the tested simulation conditions. They do not imply that rough-terrain locomotion is solved: survival remains below forty percent in this challenging protocol.','Table I, baseline comparisons','Data reconciliation: Table I reports rough survival 0.386 ± 0.035. The abstract reports 0.486 and +122.9%, which conflict with the table. This deck consistently uses Table I. Charts show means only; backup A1 preserves the reported standard deviations.');
}
// 9. Motion metric and original contact evidence.
{
 const s=slide('Lower jitter and more planar contact');
 text(s,'Spectral energy (×10⁻³), lower is better',64,151,610,39,25,C.muted);
 chart(s,{x:55,y:209,w:603,h:319,categories:['Pure RL','AMP','SG-AMP'],series:[{name:'Spectral energy',values:[4.52,2.48,1.65],fill:C.teal,points:[{idx:0,fill:C.gray},{idx:1,fill:C.blue},{idx:2,fill:C.teal}]}],max:5.5,format:'0.00',legend:false,unit:1});
 text(s,'33.5% lower than AMP',64,557,617,57,32,C.teal,true);
 await img(s,'fig8.png',748,153,450,402);
 text(s,'Planar contact: 46.8% to 85.4%',720,570,496,48,29,C.teal,true);
 caption(s,'Paper Fig. 5: (a) contact model, (b) AMP, (c) PPO, (d) SG-AMP. Table I, 5 seeds.');
 note(s,'Lower jitter and more planar contact',55,'The improvements also appear in motion and contact quality. On the left, spectral energy decreases from 2.48 to 1.65 times ten to the minus three compared with standard AMP, a reduction of approximately 33.5 percent. Compared with Pure RL, the reduction is 63.5 percent. Keeping these comparison baselines explicit is important. On the right, the paper’s contact analysis shows the contact model, standard AMP, vanilla PPO, and SG-AMP. The reported planar-contact ratio increases from 46.8 percent for AMP to 85.4 percent for SG-AMP. This metric describes the portion of stance without ankle inversion or eversion tilting. The qualitative images and quantitative metrics together indicate smoother motion and more consistent flat-foot contact in these tests.','Table I and Fig. 5; source image fig8.png','Spectral-energy reduction of 63.5% uses Pure RL as the baseline. The reduction relative to standard AMP is 33.5%. Planar-contact ratio retains the paper’s stance-based definition.');
}
// 10. Editable ablation summary, with full values retained in backup.
{
 const s=slide('Each module contributes to performance');
 table(s,[['Configuration','Push survival','Rough survival','Spectral energy\n(×10⁻³)'],['Without CoP gate', '51.2%','25.8%','1.92'],['Without phase input','64.8%','33.4%','3.08'],['Fixed style weight','61.5%','30.5%','1.85'],['Full SG-AMP','73.6%','38.6%','1.65']],64,167,1152,330,[387,242,242,281],25);
 text(s,'Removing G1: planar contact falls from 85.4% to 49.6%.',64,536,1152,43,27,C.teal,true);
 text(s,'Removing G2: spectral energy increases by 86.7%.',64,583,1152,43,27,C.teal,true);
 caption(s,'Table I means, 5 seeds. Higher survival and lower spectral energy indicate better performance.');
 note(s,'Each module contributes to performance',65,'The ablation study removes one component at a time. Without the CoP gate, push survival drops from 73.6 to 51.2 percent. Planar contact also falls sharply, from 85.4 to 49.6 percent. This links the gate to contact quality as well as robustness. Without phase conditioning, spectral energy rises from 1.65 to 3.08 times ten to the minus three, an increase of 86.7 percent. That is consistent with phase information reducing high-frequency fluctuations. Finally, fixing the style weight reduces push survival to 61.5 percent and rough-terrain survival to 30.5 percent. This supports the benefit of adapting imitation intensity. The full method achieves the best mean values in this table. These ablations support complementary roles, although they do not isolate every possible interaction among the components.','Table I, ablation configurations','The narrative text reports 0.352 rough survival without G3, while Table I gives 0.305 ± 0.039. This deck uses the table value.');
}
// 11. Reported gate usage, defined precisely.
{
 const s=slide('The gate attenuates style more under stress');
 chart(s,{x:56,y:188,w:800,h:409,categories:['Flat walking','Pushes','Rough terrain'],series:[{name:'Stance with g < 0.8',values:[.084,.356,.442],fill:C.teal}],max:.6,format:'0.0%',legend:false,unit:.2});
 text(s,'Attenuation criterion',900,210,316,50,29,C.teal,true);
 text(s,'Percentage of stance\nwith g(mₜ) < 0.8',900,275,316,97,29);
 text(s,'Higher exposure\nto difficult contacts\nreduces style influence.',900,426,316,128,28);
 caption(s,'Reported gate behavior in Section IV-C. These are attenuation fractions, not failure rates.');
 note(s,'The gate attenuates style more under stress',40,'We can also inspect when the gate reduces the style signal. The paper reports the proportion of stance for which the gate is below 0.8. That fraction is 8.4 percent during flat walking, 35.6 percent under pushes, and 44.2 percent on rough terrain. These are attenuation fractions, not failure rates. The pattern is consistent with the intended mechanism: difficult contacts lead to more frequent reductions in imitation strength. This provides a useful behavioral explanation for the robustness results, beyond the final survival numbers alone.','Section IV-C, Gate behavior under stress');
}
// 12. Closing remains visible for discussion; backup slides are hidden.
{
 const s=slide('Conclusion',{dark:true});
 text(s,'Stability regulates the influence\nof motion imitation',64,174,1152,151,54,C.white,true);
 text(s,'CoP gating improves contact quality.\nPhase conditioning reduces jitter.\nAn adaptive style weight improves the robustness tradeoff.',64,369,1152,142,31,'#BFD8E1');
 text(s,'Next: hardware transfer and perception for varied terrain',64,556,1152,46,28,'#69CCD0');
 text(s,'Thank you. Questions?',64,617,1152,43,30,C.white,true);
 note(s,'Conclusion',35,'To conclude, SG-AMP makes motion imitation responsive to stability. The CoP gate improves contact behavior, phase conditioning reduces jitter, and the adaptive style weight helps balance imitation with locomotion. Together, these changes improve robustness in our simulation benchmarks while preserving useful motion style. The current study is limited to simulation on one biped platform. Our next steps are hardware transfer and incorporating perception for more varied terrain. Thank you for your attention. I welcome your questions.','Conclusion and Table I');
}
// A1. Full reported benchmark, kept native and available for Q&A.
{
 const s=slide('Full benchmark and ablation results',{backup:true});s.hidden=true;
 table(s,[['Method','Push ×3\nsurvival','Rough\nsurvival','Spectral energy\n(×10⁻³)','Planar\ncontact'],['Pure RL','0.435 ± 0.038','0.245 ± 0.032','4.52 ± 0.28','54.2 ± 3.6%'],['Standard AMP','0.462 ± 0.041','0.218 ± 0.035','2.48 ± 0.16','46.8 ± 4.2%'],['SG-AMP without G1','0.512 ± 0.036','0.258 ± 0.034','1.92 ± 0.14','49.6 ± 3.9%'],['SG-AMP without G2','0.648 ± 0.035','0.334 ± 0.038','3.08 ± 0.21','81.2 ± 2.6%'],['SG-AMP without G3','0.615 ± 0.042','0.305 ± 0.039','1.85 ± 0.15','76.8 ± 3.2%'],['Full SG-AMP','0.736 ± 0.032','0.386 ± 0.035','1.65 ± 0.12','85.4 ± 2.1%']],64,163,1152,425,[350,200,200,200,202],21);
 caption(s,'Source: Table I. Mean ± standard deviation across 5 random seeds in MuJoCo.');
 s.speakerNotes.text='BACKUP A1. Not included in the ten-minute talk.\nSOURCE: SGAMP.pdf, Table I. The original table is authoritative for this presentation. Survival is a proportion. Planar contact is reported as percent of stance, with standard deviation in percentage points. No significance tests or confidence intervals are provided.\nWhen discussing uncertainty, report five-seed mean and standard deviation rather than claiming statistical significance.';
}
// A2. Technical questions, concrete limits.
{
 const s=slide('Implementation and discussion',{backup:true});s.hidden=true;
 const rows=[['Contact geometry','0.27 × 0.10 m foot soles, 3 × 2 contact elements per foot.\nSupport geometry uses active foot contacts.'],['Reference motion','Retargeted public MoCap. The paper also describes\nprivileged-expert trajectories as a fallback.'],['Scope','One self-developed biped, simulation only.\nNo hardware results or formal safety guarantee.'],['Open details','CoP reconstruction and contact-force availability\nrequire simulator-specific implementation.']];
 rows.forEach((r,i)=>{const y=165+i*116;text(s,r[0],64,y,295,50,29,C.teal,true);text(s,r[1],385,y,831,92,27);});
 s.speakerNotes.text='BACKUP A2. Not included in the ten-minute talk.\nSOURCE: SGAMP.pdf, Section III implementation notes and Section IV-A.\nSuggested Q&A:\nQ: Does CoP gating guarantee stability? A: No. It modulates a training reward using a stability-related contact proxy; the policy can still fail.\nQ: Has this transferred to real hardware? A: No, the paper reports Isaac Sim training and MuJoCo zero-shot evaluation only.\nQ: Does the reported improvement apply to other robots? A: The mechanism is designed around contact forces and sole geometry, but the experiments cover one biped.\nQ: What reference data are used? A: The manuscript describes retargeted public MoCap and a privileged-expert fallback; the exact dataset and fallback usage are not fully specified.\nQ: What implementation points should be clarified? A: The manuscript presents element-level CoP forces while its implementation notes mention per-foot net forces and reconstructed foot corners. It also pairs a 50 Hz controller with a 0.005 s step without clearly distinguishing control and simulation steps. Avoid claiming those details are fully resolved.';
}

await fs.mkdir(path.join(build,'previews'),{recursive:true});
await fs.mkdir(out,{recursive:true});
await fs.writeFile(path.join(build,'deck.proto.json'),JSON.stringify(p.toProto()));
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
 console.log(`Rendered ${i+1}/${p.slides.items.length}`);
}
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const final=path.join(out,'SG-AMP_10min_English.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:final,pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','10','--require-native-table-slide','13'],requiredNativeTableOwnerSlides:[10,13],requiredNativeChartOwnerSlides:[8,9,11],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
console.log(JSON.stringify(result));

let elapsed=0;
let guide='# SG-AMP: 10-minute English conference talk\n\n12 main slides, 600 seconds in total. Two hidden backup slides support Q&A. The English script is also embedded in PowerPoint speaker notes. Timings include short pauses and figure explanation. Rehearse once with a timer and adjust pace.\n\n';
const mmss=n=>`${Math.floor(n/60)}:${String(n%60).padStart(2,'0')}`;
guide+='## Timing\n\n| Slide | Topic | Duration | Cumulative |\n|---|---|---:|---:|\n';
scripts.forEach((v,i)=>{elapsed+=v.seconds;guide+=`| ${i+1} | ${v.title} | ${v.seconds} s | ${mmss(elapsed)} |\n`;});
scripts.forEach((v,i)=>{guide+=`\n## Slide ${i+1}: ${v.title}\n\nSuggested time: ${v.seconds} seconds.\n\n${v.script}\n`;});
guide+='\n## Before the conference: manuscript data checks\n\n以下不一致来自论文原稿。PPT 统一采用 Table I，未修改论文文件。正式报告前建议与原始实验记录核对。\n\n| Item | Conflicting manuscript text | Value used in this deck |\n|---|---|---|\n| SG-AMP rough survival | Abstract: 0.486; Table I: 0.386 ± 0.035 | 0.386 ± 0.035 |\n| Rough improvement over AMP | Abstract and results: +122.9% | +77.1% relative, or +16.8 percentage points, using 0.386 versus 0.218 |\n| G3 ablation rough survival | Results paragraph: 0.352; Table I: 0.305 ± 0.039 | 0.305 ± 0.039 |\n| Spectral-energy reduction | Abstract: 63.5%, comparison baseline not stated | 33.5% versus AMP; 63.5% versus Pure RL |\n| Controller and step | 50 Hz control and 0.005 s step appear together | Both omitted from the main slides pending clarification of simulation versus control time step |\n| CoP implementation | Element-level force equation and per-foot net-force implementation description | Main slides explain the mathematical idea; no claim that simulator reconstruction is fully documented |\n\nAll chart bars show means. Backup A1 gives the full mean ± standard deviation for five seeds. No p-values or formal safety guarantees are claimed. All experimental evidence is simulation only.\n\n## Suggested Q&A answers\n\n**Does the gate guarantee that the robot will not fall?** No. It is a reward modulation mechanism using a contact-based stability proxy.\n\n**Is this sim-to-real?** The reported transfer is sim-to-sim, from Isaac Sim / PhysX to MuJoCo. Hardware transfer is future work.\n\n**Why use both a gate and an adaptive weight?** The gate responds to the current step, while the moving average adjusts the overall imitation intensity based on recent stability.\n\n**How is phase defined?** A commanded-frequency clock advances the phase, heel-strike events align it, and both reference and policy data use the same convention.\n\n**Which data are needed?** Contact forces, foot geometry, proprioceptive state transitions, and reference motion with phase labels.\n\n**Which module has the strongest contact effect?** In Table I, removing G1 reduces planar contact from 85.4% to 49.6%. Removing G2 has the largest adverse effect on spectral energy.\n\n## Delivery tips\n\nUse PowerPoint Presenter View to see the script and timing. End the main talk at slide 12. Backup A1 and A2 are hidden so that the slide show remains on the conclusion. They can be opened from the slide navigator during questions.\n\nPronunciation: CoP, “C-O-P”; PPO, “P-P-O”; MuJoCo, “moo-JOH-koh”; sigmoid, “SIG-moyd”.\n';
await fs.writeFile(path.join(out,'SG-AMP_English_Speaker_Guide.md'),guide);
console.log('Main-script word count:',scripts.map(x=>x.script).join(' ').split(/\s+/).length);
console.log('DONE',final);
