import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';

const root=process.cwd(),tmp=path.join(root,'_ppt_build','revision2'),out=path.join(root,'output');
const skill='C:/Users/dongx/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const python='C:/Users/dongx/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/dongx/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const p=Presentation.create({slideSize:{width:1280,height:720}});p.theme.defaultFont='Arial';
const C={navy:'#10283F',ink:'#19394D',gray:'#91A5B3',blue:'#467190',teal:'#007F88',amber:'#A55E12',light:'#D9E5EA',pale:'#EDF5F6',muted:'#4F6473',white:'#FFFFFF'};
p.theme.colorScheme={name:'SG-AMP Scientific',themeColors:{accent1:C.teal,accent2:C.blue,accent3:C.amber,accent4:'#9E5D71',accent5:'#626B98',accent6:C.gray,bg1:C.white,bg2:C.pale,tx1:C.ink,tx2:C.muted,dk1:'#000000',dk2:C.navy,lt1:C.white,lt2:C.light,hlink:C.teal,folHlink:'#626B98'}};
let n=0;const spoken=[],charts=[];
const text=(s,t,x,y,w,h,size=36,color=C.ink,bold=false,align='left')=>{
 const sh=s.shapes.add({geometry:'textbox',name:t.slice(0,60),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 sh.text=t;sh.text.style={fontSize:size,typeface:'Arial',color,bold,alignment:align,verticalAlignment:'top',autoFit:'none',wrap:'square',insets:0};return sh;
};
function eq(s,latex,x,y,w,h,size=46,color=C.navy){
 const substitutions={'\\mathcal{L}':'L','\\mathcal{M}':'M','\\mathcal{S}':'S','\\mathbb{E}':'E','\\tfrac':'\\frac','\\Delta':'Δ','\\kappa':'κ','\\psi':'ψ','\\partial':'∂','\\in':'∈'};
 for(const [a,b] of Object.entries(substitutions))latex=latex.replaceAll(a,b);
 const sh=text(s,'',x,y,w,h,size,color);
 sh.text=[{runs:[{latex,displayMode:'inline'}]}];
 sh.text.style={fontSize:size,typeface:'Cambria Math',color,alignment:'left',insets:0,autoFit:'none',wrap:'none'};
 return sh;
}
function slide(title,{dark=false,backup=false}={}){
 const s=p.slides.add();n++;s.background.fill=dark?C.navy:C.white;
 if(title) text(s,title,56,40,1168,98,50,dark?C.white:C.navy,true);
 text(s,backup?`A${n-13}`:String(n).padStart(2,'0'),1154,674,70,28,22,dark?'#ABC7D5':C.muted,false,'right');
 if(backup)s.hidden=true;return s;
}
function note(s,title,seconds,script,source,additional=''){
 s.speakerNotes.text=`SUGGESTED TIME: ${seconds} seconds\n\nSPEAKING SCRIPT\n${script}\n\nSOURCE\nSGAMP.pdf, ${source}. User-supplied manuscript.\n\n${additional}`;
 if(seconds)spoken.push({title,seconds,script});
}
async function image(s,file,x,y,w,h,crop){
 const b=await fs.readFile(path.join(root,file));return s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:file.endsWith('.png')?'image/png':'image/jpeg',alt:`Original manuscript figure ${file}${crop?' (selected panel)':''}`,fit:crop?'cover':'contain',...(crop?{crop}:{}),position:{left:x,top:y,width:w,height:h}});
}
function node(s,label,x,y,w,h,color=C.teal){
 const sh=s.shapes.add({geometry:'rect',name:label.replace(/\n/g,' '),position:{left:x,top:y,width:w,height:h},fill:C.white,line:{fill:color,width:2}});
 sh.text=label;sh.text.style={fontSize:34,typeface:'Arial',color,bold:true,alignment:'center',verticalAlignment:'middle',insets:10,autoFit:'none'};return sh;
}
function connect(s,a,b,from='bottom',to='top',color=C.muted){return s.shapes.connect(a,b,{kind:'elbow',fromSide:from,toSide:to,line:{fill:color,width:2.5},tail:{type:'triangle',width:'med',length:'med'}});}
function bar(s,{x=56,y=180,w=1168,h=390,categories,series,max=1,unit=.2,format='0.0%',legend=true,title,labels=32}){
 const c=s.charts.add('bar',{position:{left:x,top:y,width:w,height:h},categories,series,hasLegend:legend,...(title?{title,titleTextStyle:{fontSize:32,fill:C.navy,bold:true}}:{}),legend:{position:'bottom',textStyle:{fontSize:30,fill:C.ink}},barOptions:{direction:'column',grouping:'clustered',gapWidth:95},chartFill:C.white,plotAreaFill:C.white,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},xAxis:{textStyle:{fontSize:32,fill:C.ink},line:{fill:C.gray,width:1},majorGridlines:null},yAxis:{min:0,max,majorUnit:unit,numberFormatCode:format,textStyle:{fontSize:28,fill:C.muted},line:{fill:'none',width:0},majorGridlines:{fill:C.light,width:1}},dataLabels:{showValue:true,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:labels,fill:C.ink,bold:true}}});
 applyPresentationChartFont(c,{fontFamily:'Arial'});charts.push(n);return c;
}
function table(s,values,y=170,height=430,widths=[280,222,222,222,222],font=32){
 const tb=s.tables.add({rows:values.length,columns:values[0].length,left:56,top:y,width:1168,height,columnWidths:widths,values});tb.borders.assign({fill:C.light,width:.8});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const ce=tb.getCell(r,c);ce.fill=r===0?C.navy:(r===values.length-1?C.pale:C.white);
  ce.text.style={fontSize:font,typeface:'Arial',color:r===0?C.white:C.ink,bold:r===0||r===values.length-1,alignment:c===0?'left':'center',verticalAlignment:'middle',insets:10};
 }
 return tb;
}

// 01. Exact official title, author order preserved.
{
 const s=slide('',{dark:true});
 text(s,'SG-AMP',56,68,1168,119,96,'#70D0D2',true);
 text(s,'Stability-Gated Adversarial Motion Priors\nwith Phase-Conditioned Discriminator\nfor Natural and Robust Bipedal Locomotion',56,222,1168,208,44,C.white,true);
 text(s,'Jiahong Li, Xuepeng Dong, Shengli Xu, Beiyan Jiang*',56,495,1168,51,32,C.white);
 text(s,'Beijing Union University\nBeijing Zoyi Technology Co., Ltd.',56,563,1168,80,30,'#B7D1DC');
 note(s,'Title',25,'Good morning. Today I will present SG-AMP, our method for natural and robust bipedal locomotion. The central idea is to regulate adversarial motion imitation using contact stability. I will explain the formulation, show how its three components interact, and then examine the quantitative evidence from simulation.','title and abstract');
}
// 02. Research problem with a precise comparison of objectives.
{
 const s=slide('Research problem: style and stability');
 text(s,'Task rewards support locomotion.\nMotion priors encourage natural gait.',56,148,1168,96,39);
 await image(s,'fig7.jpg',56,281,800,354);
 text(s,'Style score lacks\ncontact stability',900,282,324,100,35,C.teal,true);
 text(s,'Phase ambiguity\nadds jitter',900,415,324,92,35,C.teal,true);
 text(s,'A fixed weight\nlimits adaptation',900,541,324,92,35,C.teal,true);
 note(s,'Research problem',40,'A task reward can make a biped walk, but it does not fully capture the coordination and appearance of human motion. AMP introduces reference-motion style through an adversarial discriminator. The difficulty is that a good imitation score does not directly establish stable foot contact. Phase ambiguity can also create inconsistent rewards within a gait cycle, and a fixed style weight cannot adapt to changing stability. Our question is therefore how to retain motion imitation while regulating when, and how strongly, it should influence locomotion. The sequence shown here is the SG-AMP policy in simulation.','Introduction and Fig. 4; original source image fig7.jpg');
}
// 03. Native Office mathematical expressions, not Unicode approximations.
{
 const s=slide('AMP formulation and objective conflict');
 text(s,'Discriminator: reference transitions versus policy transitions',56,147,1168,49,34,C.teal,true);
 eq(s,'\\mathcal{L}_D=\\frac{1}{2}\\mathbb{E}_{\\mathcal{M}}[(D-1)^2]+\\frac{1}{2}\\mathbb{E}_{\\pi}[(D+1)^2]',56,211,1168,104,44);
 eq(s,'r_t^{\\mathrm{amp}}=\\max(0,1-\\tfrac{1}{4}(D(s_t,s_{t+1})-1)^2)',56,339,1168,93,44);
 eq(s,'r_t=r_t^{\\mathrm{task}}+w^{\\mathrm{style}}r_t^{\\mathrm{amp}}',56,456,1168,90,50,C.teal);
 text(s,'A fixed style weight couples imitation and task performance.',56,591,1168,62,36,C.ink,true);
 note(s,'AMP formulation',50,'Standard AMP learns a discriminator that separates reference transitions from policy transitions. The least-squares objective assigns different targets to these two groups. Its score becomes a bounded style reward, shown in the second equation. PPO then optimizes the sum of task reward and weighted style reward. The key issue is the fixed coefficient in the last equation. A strong style signal may compete with recovery during a difficult contact. A weak coefficient reduces the benefit of imitation. SG-AMP retains this adversarial framework, but changes both the discriminator’s conditioning and how its style reward enters the policy objective.','Section III-A, Eqs. (1)–(2)');
}
// 04. Native editable technical diagram, simplified from Eqs. 5–8, not the inconsistent Fig. 1 artwork.
{
 const s=slide('SG-AMP training architecture');
 const a=node(s,'Contact forces\nand sole geometry',56,158,480,101,C.muted);
 const b=node(s,'Reference / policy data\nwith contact-aligned phase',731,158,493,101,C.muted);
 const g1=node(s,'G1  CoP margin\nand style gate',56,360,335,104,C.teal);
 const g3=node(s,'G3  EMA of gate\nand style weight',472,360,335,104,C.amber);
 const g2=node(s,'G2  Discriminator\nwith gait phase',889,360,335,104,C.blue);
 connect(s,a,g1);connect(s,b,g2);connect(s,g1,g3,'right','left',C.amber);
 text(s,'All three signals enter the reward used by PPO',56,510,1168,50,35,C.teal,true);
 eq(s,'r_t=r_t^{\\mathrm{task}}+\\lambda_t\\,g(m_t)\\,r_t^{\\mathrm{amp}}',56,574,1168,90,52);
 note(s,'Training architecture',50,'This architecture shows where each modification enters training. Contact forces and sole geometry define the CoP margin and the instantaneous style gate, G1. An exponential moving average of that gate determines the adaptive style weight, G3. Separately, reference and policy transitions enter a discriminator conditioned on contact-aligned gait phase, G2. The three signals meet in the reward optimized by PPO. The important distinction is that the CoP signal directly scales the style contribution, rather than appearing only as another task-reward term. We do not replace the policy optimizer or the adversarial training loop. Each modification also has a corresponding ablation.','Section III-B–E, Eqs. (5)–(8) and Algorithm 1','The architecture follows the manuscript equations. The original Fig. 1 image contains inconsistent adaptive-weight labels, so this diagram uses the mathematical definitions.');
}
// 05. Full CoP formulation and a native analytic gate plot.
{
 const s=slide('G1: contact geometry controls style reward');
 eq(s,'p_{\\mathrm{CoP}}=\\frac{\\sum_{i,k} f_{z,i,k}\\,p_{i,k}}{\\sum_{i,k} f_{z,i,k}}',56,152,610,146,47);
 eq(s,'m_t=\\min_{e\\in\\partial\\mathcal{S}_t}d_{\\mathrm{signed}}(p_{\\mathrm{CoP}},e)',56,325,654,91,40);
 eq(s,'g(m_t)=\\sigma\\left(\\frac{m_t-m_0}{\\kappa}\\right)',56,446,654,115,47,C.teal);
 const xs=Array.from({length:21},(_,i)=>Number((i*.0025).toFixed(4))),ys=xs.map(x=>Number((1/(1+Math.exp(-(x-.025)/.006))).toFixed(7)));
 const ch=s.charts.add('scatter',{position:{left:753,top:164,width:471,height:350},categories:[],series:[{name:'Analytical sigmoid',values:ys,xValues:xs,marker:{symbol:'none'},line:{fill:C.teal,width:4}}],hasLegend:false,scatterOptions:{style:'line'},xAxis:{min:0,max:.05,majorUnit:.025,numberFormatCode:'0.000',title:{text:'Margin m (m)',textStyle:{fontSize:28,fill:C.ink}},textStyle:{fontSize:28,fill:C.ink},majorGridlines:null,line:{fill:C.gray,width:1}},yAxis:{min:0,max:1,majorUnit:.5,numberFormatCode:'0.0',title:{text:'Gate g(m)',textStyle:{fontSize:28,fill:C.ink}},textStyle:{fontSize:28,fill:C.ink},majorGridlines:{fill:C.light,width:1},line:{fill:'none',width:0}},chartFill:C.white,plotAreaFill:C.white});applyPresentationChartFont(ch,{fontFamily:'Arial'});charts.push(n);
 text(s,'m₀ = 0.025 m\nκ = 0.006 m',783,536,441,95,35,C.teal,true);
 text(s,'Support polygon: active-foot convex hull',56,598,684,48,33);
 note(s,'CoP gate',75,'The CoP is the vertical-force-weighted average of the contact positions. We construct a support polygon from active-foot geometry, then compute the minimum signed distance from the CoP to its boundary. This margin controls a sigmoid gate. The curve is an analytical evaluation of the equation, not an experimental trajectory. At the threshold of 2.5 centimeters, the gate is one half. Larger margins preserve more of the style reward, while smaller margins attenuate it continuously. The smoothing scale is 6 millimeters. This multiplication regulates the imitation signal at each step without adding a new gradient path through the physics. We treat the contact margin as a training proxy for stability, rather than a formal guarantee. Accurate contact reconstruction is therefore an important implementation requirement.','Section III-B, Eqs. (3)–(5)','Analytic plot computed directly from Eq. (5) with m0=0.025 m and kappa=0.006 m. Active contact threshold in the paper: vertical force >1 N. The manuscript’s CoP reconstruction details need clarification before hardware implementation.');
}
// 06. Clear conditioning equations replace the tiny labels of the source diagram.
{
 const s=slide('G2: gait phase conditions the discriminator');
 text(s,'Commanded frequency advances the clock.\nHeel strikes align the phase with actual contact.',56,148,1168,99,37);
 eq(s,'\\phi_{t+\\Delta t}=\\phi_t+2\\pi f_{\\mathrm{cmd}}\\Delta t',56,284,1168,89,49);
 eq(s,'D_\\psi(s_t,s_{t+1},\\cos\\phi_t,\\sin\\phi_t)',56,414,1168,93,52,C.teal);
 text(s,'Reference and policy data use the same phase convention.',56,563,1168,54,36,C.ink,true);
 note(s,'Phase conditioning',50,'Gait phase gives the discriminator context for interpreting a transition. The phase clock advances at the commanded step frequency, and heel-strike events align it with actual contacts. Cosine and sine represent the cyclic variable continuously across the phase boundary. We concatenate this encoding with the state transition. The same convention labels the reference data offline, so the discriminator can compare motion at corresponding phases. This targets within-cycle inconsistency rather than changing the task reward. In the ablation, removing phase conditioning almost doubles spectral energy, which is the most direct quantitative evidence for its contribution.','Section III-C, Eq. (6), Table I');
}
// 07. Full two-timescale objective, with native math.
{
 const s=slide('G3: imitation adapts at two time scales');
 eq(s,'\\bar g_t=\\alpha g(m_t)+(1-\\alpha)\\bar g_{t-1}',56,159,1168,96,51);
 eq(s,'\\lambda_t=\\lambda_{\\max}\\bar g_t',56,286,1168,88,51,C.teal);
 eq(s,'r_t=r_t^{\\mathrm{task}}+\\lambda_{\\max}\\bar g_t\\,g(m_t)\\,r_t^{\\mathrm{amp}}',56,407,1168,103,48);
 text(s,'Fast gate: current contact\nSlow weight: recent stability',56,557,745,99,36,C.ink,true);
 eq(s,'\\alpha=0.01',904,552,320,47,37,C.teal);
 eq(s,'\\lambda_{\\max}=0.5',904,608,320,48,37,C.teal);
 note(s,'Adaptive curriculum',50,'The curriculum adds a slower adaptation to the instantaneous gate. We average the gate using an exponential moving average, then multiply it by the maximum style weight. The average starts at zero, with alpha equal to 0.01 and a maximum weight of 0.5. The full reward therefore contains two stability factors. The current gate responds immediately to contact conditions. Its average changes imitation intensity according to recent stability. This allows imitation to increase as the policy stabilizes, while retaining a fast response to difficult contacts. The ablation with a fixed style weight tests whether this slower adaptation contributes beyond the gate alone.','Section III-D, Eqs. (7)–(8) and Algorithm 1');
}
// 08. Experimental design states what the comparisons actually test.
{
 const s=slide('Experimental design and evaluation');
 text(s,'Platform',56,155,265,50,38,C.teal,true);
 text(s,'1.75 m, 60.9 kg, 12 actuated joints\nProprioceptive policy, joint PD targets',352,153,872,104,36);
 text(s,'Training',56,297,265,50,38,C.teal,true);
 text(s,'PPO in Isaac Sim / PhysX, ≈4,096 environments\nFlat terrain with random velocity commands',352,295,872,104,35);
 text(s,'Evaluation',56,439,265,50,38,C.teal,true);
 text(s,'Zero-shot MuJoCo transfer, 5 random seeds\nPushes ×3 and unseen rough terrain',352,437,872,104,36);
 text(s,'All methods reach 100% survival in nominal flat walking.',56,600,1168,53,35,C.ink,true);
 note(s,'Experimental design',40,'Our platform is a 1.75-meter, 60.9-kilogram biped with twelve actuated joints. The proprioceptive policy outputs joint-level PD targets. We train with PPO in approximately four thousand parallel Isaac Sim environments, using flat terrain and random velocity commands. Evaluation transfers the policy to MuJoCo without further training and uses five random seeds. The baselines are task-only reinforcement learning and standard AMP with a fixed style weight. All methods survive nominal flat walking. The stronger tests are external pushes and unseen rough terrain, together with motion and contact quality. These results establish simulation evidence on this platform.','Section III-A, implementation notes, Section IV-A–C');
}
// 09. Full mean ± SD in native data labels, avoiding misleading inferred error bars.
{
 const s=slide('SG-AMP improves survival under stress');
 const vals=[[.435,.245],[.462,.218],[.736,.386]],sd=[[3.8,3.2],[4.1,3.5],[3.2,3.5]],names=['Pure RL','AMP','SG-AMP'],colors=[C.gray,C.blue,C.teal];
 const series=vals.map((v,i)=>({name:names[i],values:v,fill:colors[i],dataLabelOverrides:v.map((vv,j)=>({idx:j,text:`${(vv*100).toFixed(1)}%\n±${sd[i][j].toFixed(1)} pp`,position:'outEnd',showValue:false,textStyle:{fontSize:31,fill:C.ink,bold:true}}))}));
 bar(s,{y:156,h:430,categories:['Pushes ×3','Rough terrain'],series,max:1.05,unit:.25,format:'0%',labels:32});
 text(s,'Mean ± SD over 5 random seeds',56,143,1168,49,32,C.muted);
 text(s,'vs. AMP: +27.4 pp on pushes, +16.8 pp on rough terrain',56,604,1168,56,35,C.teal,true);
 note(s,'Robustness results',60,'The labels show the mean survival and its standard deviation over five seeds. Under external pushes, AMP reaches 46.2 percent survival and SG-AMP reaches 73.6 percent, an increase of 27.4 percentage points. On unseen rough terrain, the corresponding values are 21.8 and 38.6 percent, a gain of 16.8 points. In relative terms, these improvements are 59.3 and 77.1 percent. Pure RL also performs below SG-AMP in both tests. The variation across seeds is much smaller than the observed mean differences, but the paper does not report a significance test, so we do not claim one. Rough-terrain survival remains below forty percent, which also sets a clear limit on the result.','Table I baseline comparisons','Labels give mean ± standard deviation. Standard deviation is expressed in percentage points. Table I is used consistently: rough survival 0.386, not the abstract’s 0.486. Relative rough improvement recalculated as (0.386/0.218−1)×100=77.1%.');
}
// 10. Larger selected source panels and measurable contact quality.
{
 const s=slide('Contact quality and motion smoothness');
 text(s,'Planar stance ratio',56,150,650,51,36,C.teal,true);
 bar(s,{x:48,y:220,w:655,h:328,categories:['RL','AMP','SG-AMP'],series:[{name:'Planar stance',values:[.542,.468,.854],fill:C.teal,points:[{idx:0,fill:C.gray},{idx:1,fill:C.blue},{idx:2,fill:C.teal}]}],legend:false,max:1.05,unit:.25,format:'0.0%',labels:33});
 text(s,'AMP',751,151,215,45,34,C.navy,true,'center');
 text(s,'SG-AMP',1002,151,222,45,34,C.teal,true,'center');
 await image(s,'fig8.png',751,215,215,188,{left:290/525,top:25/469,right:35/525,bottom:270/469});
 await image(s,'fig8.png',1002,215,222,188,{left:290/525,top:240/469,right:35/525,bottom:55/469});
 text(s,'Spectral energy',751,447,473,49,35,C.teal,true);
 text(s,'2.48 to 1.65 ×10⁻³\n33.5% lower than AMP',751,512,473,104,35,C.ink,true);
 text(s,'Planar stance excludes ankle inversion / eversion.',56,603,650,59,31,C.muted);
 note(s,'Contact and motion quality',45,'The contact metric measures the proportion of stance without ankle inversion or eversion. SG-AMP increases this ratio from 46.8 percent for AMP to 85.4 percent. The enlarged source panels compare AMP and SG-AMP at contact. They illustrate the difference qualitatively, while the ratio provides the measured evidence. Motion smoothness also improves: spectral energy falls from 2.48 to 1.65 times ten to the minus three, or 33.5 percent relative to AMP. These metrics support better contact behavior and lower jitter. They do not establish that every edge landing or oscillation has been eliminated.','Table I and Fig. 5, panels (b) and (d), from fig8.png','Two selected panels from the same original figure are shown as a direct comparison. Full numerical uncertainties appear in backup A1.');
}
// 11. Scientific multipanel ablation; each panel tests the most relevant metric.
{
 const s=slide('Ablations link each module to its effect');
 const spec=[{x:48,title:'G1: planar stance',cat:['Full','w/o G1'],v:[.854,.496],max:1,unit:.5,fmt:'0.0%'},{x:451,title:'G2: energy (×10⁻³)',cat:['Full','w/o G2'],v:[1.65,3.08],max:4,unit:2,fmt:'0.00'},{x:854,title:'G3: rough survival',cat:['Full','w/o G3'],v:[.386,.305],max:.5,unit:.25,fmt:'0.0%'}];
 spec.forEach(sp=>{bar(s,{x:sp.x,y:166,w:378,h:390,title:sp.title,categories:sp.cat,series:[{name:sp.title,values:sp.v,fill:C.teal,points:[{idx:0,fill:C.teal},{idx:1,fill:C.gray}]}],legend:false,max:sp.max,unit:sp.unit,format:sp.fmt,labels:32});});
 text(s,'−35.8 pp',56,588,370,51,39,C.teal,true,'center');
 text(s,'+86.7%',459,588,370,51,39,C.teal,true,'center');
 text(s,'−8.1 pp',862,588,362,51,39,C.teal,true,'center');
 note(s,'Module ablations',65,'The three ablation panels focus on the metric most directly related to each component. On the left, removing CoP gating reduces planar stance from 85.4 to 49.6 percent, a loss of 35.8 percentage points. In the middle, removing phase conditioning increases spectral energy from 1.65 to 3.08 times ten to the minus three, an increase of 86.7 percent. On the right, fixing the style weight reduces rough-terrain survival from 38.6 to 30.5 percent, a loss of 8.1 points. The full method also has the best push survival in the complete table. This gives each design choice a measurable role. The ablations test removal of individual modules, but they do not fully characterize all possible interactions or hyperparameter sensitivities.','Table I ablation configurations','Spectral energy units in the middle panel: ×10^-3. The G3 rough survival is taken from Table I (0.305), not the conflicting results paragraph (0.352). Changes compare each ablation with full SG-AMP.');
}
// 12. Observed mechanism with precisely defined denominator.
{
 const s=slide('Gate activity increases in difficult conditions');
 bar(s,{x:48,y:181,w:796,h:382,categories:['Flat','Pushes','Rough'],series:[{name:'Attenuation',values:[.084,.356,.442],fill:C.teal}],max:.6,unit:.2,format:'0.0%',legend:false,labels:36});
 text(s,'Stance fraction\nwith g(mₜ) < 0.8',904,229,320,115,37,C.teal,true);
 text(s,'More frequent\nstyle attenuation\nunder stress',904,416,320,154,36);
 text(s,'This pattern supports the intended reward-regulation mechanism.',56,610,1168,57,34,C.ink,true);
 note(s,'Gate behavior',25,'The gate’s observed behavior is consistent with the proposed mechanism. The fraction of stance with a gate below 0.8 rises from 8.4 percent on flat ground to 35.6 percent under pushes and 44.2 percent on rough terrain. Thus, difficult conditions trigger more frequent style attenuation. These values describe gate activity, not failure rates.','Section IV-C, Gate behavior under stress');
}
// 13. Quantitative closing and limitations, with ample projection size.
{
 const s=slide('Conclusions and remaining questions',{dark:true});
 text(s,'Contact-aware reward modulation\nimproves the tested AMP controller',56,169,1168,146,50,C.white,true);
 text(s,'73.6% push survival\n85.4% planar stance\n33.5% lower spectral energy than AMP',56,367,1168,154,38,'#BBD5E0');
 text(s,'Evidence: one biped, simulation only\nNext: hardware transfer and terrain perception',56,565,1168,92,34,'#70D0D2');
 note(s,'Conclusions',25,'SG-AMP regulates motion imitation using contact stability and gait phase. The experiments support improved robustness, flatter contact, and reduced jitter on the tested biped. The evidence remains simulation-only, and hardware transfer and terrain perception are the next steps. Thank you for your attention. I welcome your questions.','Conclusion and Table I');
}

// Four readable backup pages; no tiny full-paper table.
{
 const s=slide('A1  Baselines: five-seed mean ± SD',{backup:true});
 table(s,[['Method','Push\nsurvival','Rough\nsurvival','Spectral\n×10⁻³','Planar\nstance'],['Pure RL','0.435\n±0.038','0.245\n±0.032','4.52\n±0.28','54.2%\n±3.6 pp'],['AMP','0.462\n±0.041','0.218\n±0.035','2.48\n±0.16','46.8%\n±4.2 pp'],['SG-AMP','0.736\n±0.032','0.386\n±0.035','1.65\n±0.12','85.4%\n±2.1 pp']],163,436,[280,222,222,222,222],32);
 text(s,'MuJoCo zero-shot evaluation. Standard deviation across 5 seeds.',56,626,1168,44,30,C.muted);
 note(s,'Baseline table',0,'Backup for questions. Survival values are proportions. Planar stance standard deviations are in percentage points.','Table I');
}
{
 const s=slide('A2  Ablations: five-seed mean ± SD',{backup:true});
 table(s,[['Configuration','Push\nsurvival','Rough\nsurvival','Spectral\n×10⁻³','Planar\nstance'],['Without G1','0.512\n±0.036','0.258\n±0.034','1.92\n±0.14','49.6%\n±3.9 pp'],['Without G2','0.648\n±0.035','0.334\n±0.038','3.08\n±0.21','81.2%\n±2.6 pp'],['Without G3','0.615\n±0.042','0.305\n±0.039','1.85\n±0.15','76.8%\n±3.2 pp'],['Full SG-AMP','0.736\n±0.032','0.386\n±0.035','1.65\n±0.12','85.4%\n±2.1 pp']],156,482,[280,222,222,222,222],31);
 note(s,'Ablation table',0,'Backup for questions. G1 sets g to one, G2 removes phase conditioning, and G3 fixes lambda to its maximum.','Table I and Section IV-B');
}
{
 const s=slide('A3  Training loop and signal definitions',{backup:true});
 const a=[['1','Collect policy rollouts and contact signals.'],['2','Update phase, CoP margin, gate, and EMA.'],['3','Compute task reward and gated AMP reward.'],['4','Train D on reference and policy transitions.'],['5','Update the policy with PPO.']];
 a.forEach((v,i)=>{const y=160+i*92;text(s,v[0],56,y,80,54,42,C.teal,true);text(s,v[1],153,y+2,1071,70,36);});
 note(s,'Training loop',0,'Backup for questions. Observation: base orientation and angular velocity, joint position and velocity, and previous action. Action: joint PD targets. The gait clock uses commanded frequency with heel-strike alignment. Contact threshold: 1 N. Each sole is 0.27 by 0.10 m with six contact elements.','Section III-A–E and Algorithm 1');
}
{
 const s=slide('A4  Scope, assumptions, and discussion',{backup:true});
 const rows=[['Generalization','One biped and two physics backends.\nNo hardware evidence in this manuscript.'],['Stability signal','CoP margin modulates training reward.\nIt provides no formal fall-prevention guarantee.'],['Reproducibility','Exact reference-data usage and CoP\nreconstruction need further specification.']];
 rows.forEach((v,i)=>{const y=169+i*153;text(s,v[0],56,y,310,60,37,C.teal,true);text(s,v[1],407,y,817,111,35);});
 note(s,'Scope and assumptions',0,'Backup for questions. The paper gives a contact-element CoP equation but implementation notes mention per-foot net forces and reconstructed foot corners. Clarify simulator-specific CoP reconstruction. The manuscript also states a 50 Hz controller with a 0.005 s step without distinguishing simulation and control steps. Reference data are described as retargeted MoCap with privileged-expert trajectories as a fallback, but the exact dataset and fallback use are not fully specified. The CoP is a force-weighted convex combination of contact positions; if it and the support hull use identical positive-force contact points, a negative margin needs a more precise geometric convention. Do not claim those details are experimentally resolved.','Implementation notes, Section IV-A and Conclusion');
}

await fs.mkdir(path.join(tmp,'previews'),{recursive:true});await fs.mkdir(out,{recursive:true});
await fs.writeFile(path.join(tmp,'deck.proto.json'),JSON.stringify(p.toProto()));
await fs.writeFile(path.join(tmp,'spoken.json'),JSON.stringify(spoken,null,2));
await (await PresentationFile.exportPptx(p)).save(path.join(tmp,'candidate.pptx'));
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(tmp,'previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
console.log('DRAFT READY:',n,'slides; main talk:',spoken.reduce((a,b)=>a+b.seconds,0),'seconds;',spoken.map(x=>x.script).join(' ').split(/\s+/).length,'words');
const final=path.join(out,'SG-AMP_10min_English_Academic_v2.pptx');
if(process.argv.includes('--draft'))process.exit(0);
const report=await finalizePresentation({workspaceDir:root,candidatePath:path.join(tmp,'candidate.pptx'),finalPath:final,pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--validate-bullet-geometry','--require-native-table-slide','14','--require-native-table-slide','15'],requiredNativeTableOwnerSlides:[14,15],requiredNativeChartOwnerSlides:[5,9,10,11,12],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Arial','Cambria Math']},verifyArtifactToolImport:true,receiptPath:path.join(tmp,'validation.json')});
console.log('FINAL',report.finalPath,'layout findings:',report.presentationLayout.findingCount);
let elapsed=0;const mmss=s=>`${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
let guide='# SG-AMP: Academic conference presentation, revision 2\n\n13 main slides, 600 seconds. Four hidden backup slides support questions. English scripts and sources appear in the PowerPoint speaker notes. Main text uses larger projection sizes, and the equations are native Office mathematics.\n\n';
guide+='## Timing\n\n| Slide | Topic | Seconds | End time |\n|---|---|---:|---:|\n';
spoken.forEach((v,i)=>{elapsed+=v.seconds;guide+=`| ${i+1} | ${v.title} | ${v.seconds} | ${mmss(elapsed)} |\n`;});
spoken.forEach((v,i)=>{guide+=`\n## Slide ${i+1}: ${v.title}\n\n${v.script}\n`;});
guide+='\n## Presenter checks\n\n本版继续统一采用 Table I。报告前请核对以下原稿矛盾，不应将它们解释为新的实验结果。\n\n| Item | Manuscript discrepancy | Value used |\n|---|---|---|\n| Rough survival | Abstract 0.486, Table I 0.386 | 0.386 ± 0.035 |\n| Rough relative improvement | Text +122.9% | +77.1% versus AMP, or +16.8 pp |\n| G3 ablation rough survival | Results text 0.352, Table I 0.305 | 0.305 ± 0.039 |\n| Spectral improvement | Abstract 63.5% without explicit baseline | 33.5% versus AMP, 63.5% versus Pure RL |\n| Timing | 50 Hz control and 0.005 s step | Omitted pending control/simulation-step clarification |\n| Original framework image | Adaptive-weight labels disagree with equations | Revised architecture follows Eqs. (5)–(8) |\n\nTheoretical and implementation points for discussion:\n\n- CoP is a positive-force-weighted combination of contact positions. If those positions also define the support convex hull, a negative CoP margin requires a more precise definition or reconstruction convention. The deck avoids presenting negative margins as established mechanistic proof.\n- The manuscript describes element-level CoP forces, while implementation notes mention per-foot net forces and reconstructed corners. Specify the actual estimator before making reproducibility claims.\n- Survival comparisons use five-seed means and standard deviations. The paper provides no significance tests.\n- The sigmoid plot is an analytical plot of the stated gate equation, not measured training data.\n\n## How to rehearse\n\nUse Presenter View, and speak the formulas conceptually instead of reading every symbol. On slide 5, define CoP and margin before explaining the gate curve. On slide 9, say “percentage points” for absolute gains and “percent relative improvement” only when giving 59.3% or 77.1%. On slide 11, explain one metric per ablation panel. Stop the main talk at slide 13. Slides A1–A4 are hidden and available through the slide navigator.\n\n## Backup questions\n\n**Why is the method distinct from a CoP task reward?** CoP stability scales the AMP style contribution directly.\n\n**Why keep both the gate and its average?** The gate responds to the current step; the average sets slower imitation intensity.\n\n**Is naturalness directly established?** The paper provides motion snapshots, spectral energy, and contact metrics. It does not report a human preference study or a full perceptual-naturalness evaluation.\n\n**What does the benchmark prove?** Better means in the reported simulation protocols on one robot. Hardware transfer and wider morphology generalization remain open.\n';
await fs.writeFile(path.join(out,'SG-AMP_Academic_v2_Speaker_Guide.md'),guide);

// Render the exact finalized package. Finalizer may add workbook snapshots.
const imported=await PresentationFile.importPptx(await FileBlob.load(final));await fs.mkdir(path.join(tmp,'final_previews'),{recursive:true});
for(let i=0;i<imported.slides.items.length;i++){
 const b=await imported.export({slide:imported.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(tmp,'final_previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
console.log('Rendered exact final deck.');

