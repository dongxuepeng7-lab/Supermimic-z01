from PIL import Image, ImageOps, ImageDraw
from pathlib import Path
root = Path(__file__).resolve().parent
for name in ['english_pages','chinese_pages']:
    files=sorted((root/name).glob('page-*.png'), key=lambda p:int(p.stem.split('-')[-1]))
    for k in range(0,len(files),4):
        imgs=[]
        for p in files[k:k+4]:
            im=Image.open(p).convert('RGB')
            im.thumbnail((620,880))
            tile=Image.new('RGB',(640,910),'#eeeeee')
            tile.paste(im,((640-im.width)//2,26))
            ImageDraw.Draw(tile).text((14,7),p.stem,fill='black')
            imgs.append(tile)
        sheet=Image.new('RGB',(1280,1820),'white')
        for j,im in enumerate(imgs): sheet.paste(im,((j%2)*640,(j//2)*910))
        sheet.save(root/f'{name}_sheet_{k//4+1}.jpg',quality=90)
