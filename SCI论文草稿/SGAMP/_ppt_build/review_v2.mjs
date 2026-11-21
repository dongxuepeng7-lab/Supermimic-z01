import fs from 'node:fs/promises';
import path from 'node:path';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/dongx/Desktop/SG-AMP_10min_English_Academic_v2.pptx'));
const out=path.join(process.cwd(),'_ppt_review','renders');await fs.mkdir(out,{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(out,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
 console.log('Rendered',i+1);
}
