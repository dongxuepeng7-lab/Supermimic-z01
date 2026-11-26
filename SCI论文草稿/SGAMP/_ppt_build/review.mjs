import fs from 'node:fs/promises';
import path from 'node:path';
import { PresentationFile, FileBlob } from '@oai/artifact-tool';
const root=process.cwd();
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(root,'output/SG-AMP_10min_English.pptx')));
await fs.mkdir(path.join(root,'_ppt_build/final_previews'),{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(root,'_ppt_build/final_previews',`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
console.log('Final PPTX rendered:',p.slides.items.length,'slides');
