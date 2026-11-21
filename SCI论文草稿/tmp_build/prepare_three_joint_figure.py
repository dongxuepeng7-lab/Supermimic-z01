from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import RectangleObject
from reportlab.pdfgen.canvas import Canvas

root = Path(__file__).resolve().parents[1]
source = next((root / "paper_figures_20261121").rglob("fig03_heldout_trajectories.svg"))
dest = root / "RA-L_Actuator_World_Model" / "figures" / "three_joint_replay"
svg = "http://www.w3.org/2000/svg"
ET.register_namespace("", svg)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
tree = ET.parse(source)
document = tree.getroot()
fig = document.find(".//*[@id='figure_1']")
for i in (7, 8, 9):
    fig.remove(fig.find(f"./*[@id='axes_{i}']"))
w = float(document.get("width").removesuffix("pt"))
h = 300
label = "Time from first scored sample (s)"
document.set("height", f"{h}pt")
document.set("viewBox", f"0 0 {w} {h}")
ET.SubElement(fig, f"{{{svg}}}text", {"x":str(w/2),"y":"296","text-anchor":"middle",
              "font-family":"DejaVu Sans, sans-serif","font-size":"8"}).text = label
tree.write(dest.with_suffix(".svg"), encoding="utf-8", xml_declaration=True)

original = PdfReader(source.with_suffix(".pdf")).pages[0]
top = float(original.mediabox.top)
write = PdfWriter()
p = write.add_blank_page(w,h)
original.cropbox = RectangleObject((0,top-287,w,top))
p.merge_transformed_page(original, Transformation().translate(0,h-top))
buf = BytesIO()
overlay = Canvas(buf, pagesize=(w,h))
overlay.setFont("Helvetica",8)
overlay.drawCentredString(w/2,h-296,label)
overlay.save()
p.merge_page(PdfReader(buf).pages[0])
with dest.with_suffix(".pdf").open("wb") as f:
    write.write(f)

paper = root / "RA-L_Actuator_World_Model"
tex = paper / "manuscript.tex"
text = tex.read_text("utf-8")
start = text.index(r"\begin{figure}[!ht]" + "\n" + r"\centering" + "\n" + r"\includegraphics[width=\columnwidth]{figures/joint_replay_before_after.pdf}")
end = text.index(r"\end{figure}",start) + len(r"\end{figure}")
text = text[:start] + text[end:]
figure = r"""\begin{figure*}[!t]
\centering
\includegraphics[width=0.90\textwidth]{figures/three_joint_replay.pdf}
\caption{Held-out joint replay over the first three scored periods. Top: joint position. Bottom: joint-side feedback torque and predicted torque. Black shows the measured signals, dashed gray the original-parameter replay, and blue the identified-parameter replay. Column headings give the command scale and period. The right ankle uses a frozen candidate parameter set.}
\label{fig:jointreplay}
\end{figure*}

"""
text = text.replace(r"\subsection{Measurements and Analysis}",figure+r"\subsection{Measurements and Analysis}")
text = text.replace("illustrates the change in left hip roll over three periods.", "shows the first three scored periods for left knee, left hip roll and right ankle pitch. The left hip roll replay captures the measured position plateau and the torque level during that plateau more closely with the identified parameters.")
text = text.replace("shows a continuous joint replay with the archived parameter configurations.", "shows continuous joint replays with the archived parameter configurations.")
tex.write_text(text,"utf-8")
notes = paper / "中文修改说明.txt"
text = notes.read_text("utf-8").replace("正文图 2 取原图包图 3 的左髋侧摆轨迹，以同一实测参照、同一时间窗和坐标范围并排展示原始参数与辨识参数；图下 RMSE 源于完整评分记录。","正文图 2 采用原图包图 3 的前两行。三列依次为左膝、左髋侧摆和右踝俯仰，每列上方为位置、下方为关节侧力矩，均叠加实测、原始参数回放及辨识参数回放。显示窗口为前三个评分周期，正文的 RMSE 取自完整评分记录。")
notes.write_text(text,"utf-8")
