from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import io
import urllib.request
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent.parent / '近期运控文献_20261104'
PAPERS = [
    ('CompliantBot', '2611.01748v1'),
    ('Contact-UAN', '2610.18991v1'),
    ('DelayRandomization', '2610.32277v1'),
    ('DAMP', '2610.11505v2'),
    ('T-GMP', '2606.06944v2'),
    ('TorqueSpacePerturbation', '2504.06585v2'),
    ('StabilityAwareRetargeting', '2510.04353v1'),
]

def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for short, version in PAPERS:
        filename = f'{short}_{version}.pdf'
        url = f'https://arxiv.org/pdf/{version}'
        row = dict(filename=filename, arxiv_version=version, url=url, bytes='', sha256='', pages='', status='failed', error='', retrieved_utc=datetime.now(timezone.utc).isoformat())
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'AcademicManuscriptResearch/1.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                body = response.read()
                row['final_url'] = response.url
                row['content_type'] = response.headers.get('Content-Type', '')
            if not body.startswith(b'%PDF-'):
                raise ValueError('Response is not a PDF (missing PDF header)')
            reader = PdfReader(io.BytesIO(body))
            pages = len(reader.pages)
            if pages < 1:
                raise ValueError('PDF contains no pages')
            (OUT / filename).write_bytes(body)
            row.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), pages=pages, status='downloaded_and_pdf_validated')
            print(f'{filename}: {len(body)} bytes; {pages} pages', flush=True)
        except Exception as error:
            row['error'] = str(error)
            print(f'{filename}: FAILED {error}', flush=True)
        rows.append(row)
        keys = ['filename','arxiv_version','url','final_url','content_type','bytes','sha256','pages','status','error','retrieved_utc']
        with (OUT / 'download_manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

if __name__ == '__main__':
    main()
