import fs from 'node:fs/promises';
import * as art from '@oai/artifact-tool';
const p=art.Presentation.create({slideSize:{width:1280,height:720}});
console.log('MATH EXPORTS',Object.keys(art).filter(k=>/math/i.test(k)));
console.log(p.help('*',{search:'math equation inline',include:['index','notes','examples'],maxChars:16000}));
try { console.log('AST',JSON.stringify(art.mathFromLatex({latex:'r_t=r_t^{\\mathrm{task}}+\\lambda_t g(m_t)r_t^{\\mathrm{amp}}'}))); } catch(e){console.log('ERROR',e.message);}
const s=p.slides.add();const sh=s.shapes.add({geometry:'textbox',position:{left:50,top:100,width:1180,height:180}});
console.log('TEXT PROTO',Object.getOwnPropertyNames(Object.getPrototypeOf(sh.text)));
try{
 sh.text=[{runs:[{latex:'r_t=r_t^{\\mathrm{task}}+\\lambda_t g(m_t)r_t^{\\mathrm{amp}}',displayMode:'inline'}]}];
 sh.text.style={fontSize:48,typeface:'Cambria Math',color:'#10283F',insets:0};
 const bb=await p.export({slide:s,format:'png',scale:1});await fs.writeFile('_ppt_build/math-probe.png',new Uint8Array(await bb.arrayBuffer()));
 await (await art.PresentationFile.exportPptx(p)).save('_ppt_build/math-probe.pptx');
 console.log('OK',JSON.stringify(p.toProto().slides[0].elements[0].paragraphs));
}catch(e){console.log('ERROR2',e.message);}
