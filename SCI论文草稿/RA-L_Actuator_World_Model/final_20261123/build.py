"""Build both manuscripts. No training, dataset access, or model execution."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
for stem in ("manuscript_en", "manuscript_zh"):
    if shutil.which("tectonic"):
        commands = [["tectonic", "--keep-logs", "--keep-intermediates", f"{stem}.tex"]]
    elif shutil.which("xelatex") and shutil.which("bibtex"):
        tex = ["xelatex", "-halt-on-error", "-interaction=nonstopmode", f"{stem}.tex"]
        commands = [tex, ["bibtex", stem], tex, tex]
    else:
        raise SystemExit("Install Tectonic, or XeLaTeX + BibTeX. See README.txt for fonts.")
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    print(ROOT / f"{stem}.pdf")
