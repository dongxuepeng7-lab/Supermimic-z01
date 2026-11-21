import zipfile,xml.etree.ElementTree as ET,json,pathlib,hashlib
p=pathlib.Path(r'C:\Users\dongx\Desktop\SG-AMP_10min_English_Academic_v2.pptx')
out=pathlib.Path(__file__).parent
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
z=zipfile.ZipFile(p)
slides=[]
for fn in sorted([n for n in z.namelist() if __import__('re').match(r'ppt/slides/slide\d+\.xml$',n)],key=lambda x:int(__import__('re').search(r'slide(\d+)',x).group(1))):
 root=ET.fromstring(z.read(fn)); n=int(__import__('re').search(r'slide(\d+)',fn).group(1))
 texts=[''.join(el.itertext()) for el in root.findall('.//a:t',ns)]
 notes=f'ppt/notesSlides/notesSlide{n}.xml'
 nt='\n'.join(el.text or '' for el in ET.fromstring(z.read(notes)).findall('.//a:t',ns)) if notes in z.namelist() else ''
 math=[''.join(el.itertext()) for el in root.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath')]
 slides.append({'slide':n,'hidden':root.get('show')=='0','texts':texts,'math':math,'notes':nt})
print(json.dumps(slides,ensure_ascii=False,indent=2))
(out/'slides.json').write_text(json.dumps(slides,ensure_ascii=False,indent=2),encoding='utf-8')
for name in z.namelist():
 if name.startswith('ppt/charts/') and name.endswith('.xml') and '/_rels/' not in name:
  print(name,z.read(name).decode())
print('HASH',hashlib.sha256(p.read_bytes()).hexdigest())
