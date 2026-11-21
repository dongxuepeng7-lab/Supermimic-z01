"""Fetch selected public author/proceedings PDFs, validating before saving."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
import csv

from pypdf import PdfReader

DEST = Path(__file__).resolve().parents[1] / '参考文献'
PAPERS = [
    ('RSS2026_Perceptive_Humanoid_Parkour.pdf', 'https://roboticsproceedings.org/rss22/p020.pdf'),
    ('RSS2026_Now_You_See_That.pdf', 'https://roboticsproceedings.org/rss22/p027.pdf'),
    ('RSS2026_Mind_Your_Steps.pdf', 'https://roboticsproceedings.org/rss22/p028.pdf'),
    ('RAL2026_CReF_arXiv.pdf', 'https://arxiv.org/pdf/2603.29452v3'),
    ('RAL2026_DPL_arXiv.pdf', 'https://arxiv.org/pdf/2510.07152v3'),
    ('RAL2026_GLAD_arXiv.pdf', 'https://arxiv.org/pdf/2606.00637v3'),
    ('RAL2026_Gait_Adaptive_Perceptive_Locomotion_arXiv.pdf', 'https://arxiv.org/pdf/2512.07464'),
    ('RAL2026_Motion_Generation_Tracking_arXiv.pdf', 'https://arxiv.org/pdf/2604.17335v2'),
    ('TRO2026_AME2_arXiv.pdf', 'https://arxiv.org/pdf/2601.08485v3'),
    ('ScienceRobotics2026_BeyondMimic_arXiv.pdf', 'https://arxiv.org/pdf/2508.08241'),
    ('ScienceRobotics2026_SONIC_arXiv.pdf', 'https://arxiv.org/pdf/2511.07820v4'),
    ('CoRL2026_CAP_arXiv.pdf', 'https://arxiv.org/pdf/2609.11553v1'),
    ('CoRL2026_Contact_UAN_arXiv.pdf', 'https://arxiv.org/pdf/2610.18991v1'),
    ('CoRL2026_Hiking_in_the_Wild_arXiv.pdf', 'https://arxiv.org/pdf/2601.07718'),
    ('ICRA2026_RL_Augmented_Adaptive_MPC_arXiv.pdf', 'https://arxiv.org/pdf/2509.18466'),
    ('ICRA2026_PolygMap_arXiv.pdf', 'https://arxiv.org/pdf/2510.12346'),
    ('IROS2026_DAMP_arXiv.pdf', 'https://arxiv.org/pdf/2610.11505v2'),
    ('Preprint2026_WM_LOCO_arXiv.pdf', 'https://arxiv.org/pdf/2609.02542v2'),
    ('Preprint2026_STEP_arXiv.pdf', 'https://arxiv.org/pdf/2611.08097v1'),
]


def fetch(paper):
    name, url = paper
    path = DEST / name
    try:
        if path.exists():
            data = path.read_bytes()
            status = 'already_present'
        else:
            request = Request(url, headers={'User-Agent': 'AcademicReferenceCollection/1.0'})
            with urlopen(request, timeout=14) as response:
                data = response.read(65 * 1024 * 1024 + 1)
            status = 'downloaded'
        if not data.startswith(b'%PDF-') or len(data) > 65 * 1024 * 1024:
            raise ValueError('response does not meet PDF file format/size checks')
        pages = len(PdfReader(BytesIO(data)).pages)
        if pages < 1:
            raise ValueError('PDF has no pages')
        if status == 'downloaded':
            with path.open('xb') as handle:
                handle.write(data)
        return (name, url, status, pages, len(data), sha256(data).hexdigest(), '')
    except Exception as error:
        return (name, url, 'not_saved', '', '', '', str(error))


def main():
    with ThreadPoolExecutor(max_workers=5) as pool:
        rows = list(pool.map(fetch, PAPERS))
    manifest = DEST / '公开全文下载记录_20261117.csv'
    with manifest.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['filename', 'source_url', 'status', 'pages', 'bytes', 'sha256', 'error'])
        writer.writerows(rows)
    for row in rows:
        print(row[2], row[0], row[-1], flush=True)


if __name__ == '__main__':
    main()
