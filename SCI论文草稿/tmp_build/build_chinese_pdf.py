from pathlib import Path
import re
from html import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

ROOT = Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(TTFont('Song', 'C:/Windows/Fonts/simsun.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
body = ParagraphStyle('Body', fontName='Song', fontSize=10.5, leading=16.2,
                      spaceAfter=7, wordWrap='CJK', textColor=colors.black)
heading = ParagraphStyle('Heading', parent=body, fontSize=14, leading=20,
                         spaceBefore=14, spaceAfter=8, keepWithNext=True)
sub = ParagraphStyle('Sub', parent=body, fontSize=11.2, leading=17,
                     spaceBefore=6, spaceAfter=6, keepWithNext=True)
title = ParagraphStyle('Title', parent=body, fontSize=20, leading=28, spaceAfter=14)
small = ParagraphStyle('Small', parent=body, fontSize=9, leading=13)
url = ParagraphStyle('URL', fontName='Arial', fontSize=8.3, leading=12,
                     spaceAfter=8, splitLongWords=True, wordWrap='CJK')

source = ROOT/'中文修改说明与投稿思路_20261104.txt'
text = source.read_text(encoding='utf-8')
blocks = re.split(r'\n\s*\n', text.strip())
story = []
for i, block in enumerate(blocks):
    lines = block.splitlines()
    if i == 0:
        story.append(Paragraph(escape(lines[0]), title))
        story.append(Paragraph(escape(lines[1]), small))
        story.append(Spacer(1, 8))
        continue
    if re.match(r'^[一二三四五六七八九十]+ ', block):
        story.append(Paragraph(escape(block), heading))
        continue
    for line in lines:
        if line.startswith('https://'):
            safe = escape(line, quote=True)
            story.append(Paragraph(f'<link href="{safe}">{safe}</link>', url))
        elif re.match(r'^\d+ (CompliantBot|Contact-UAN|DAMP|T-GMP|Sim-to-Real|Bridging|Stability-Aware|Anchor|SuperMimic|HuMiT|ExtremControl|Delay)', line):
            story.append(Paragraph(escape(line), sub))
        else:
            story.append(Paragraph(escape(line), body))

def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Arial', 8)
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(48, 26, 'Research plan and revision notes | 2026-11-04')
    canvas.drawRightString(A4[0]-48, 26, str(doc.page))
    canvas.restoreState()

output = ROOT/'中文修改说明与投稿思路_20261104.pdf'
doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=48, rightMargin=48,
                       topMargin=45, bottomMargin=43, title='人形机器人执行器余量门控论文修改说明与投稿思路',
                       author='')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(output)
