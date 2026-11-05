"""REPORT_2.md -> REPORT_2.html -> REPORT_2.pdf (headless Microsoft Edge, A4).

Usage:  python scripts/make_pdf.py
"""

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
EDGE = [Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")]

CSS = """
@page { size: A4; margin: 15mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #0b0b0b; }
h1 { font-size: 17pt; margin: 0 0 4pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; border-bottom: 1px solid #c3c2b7; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 10pt 0 3pt; page-break-after: avoid; }
p, li { margin: 3pt 0; }
table { border-collapse: collapse; margin: 5pt 0; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #e1e0d9; padding: 2.5pt 5pt; text-align: left; vertical-align: top; }
th { background: #f3f2ee; }
img { max-width: 100%; max-height: 74mm; display: block; margin: 5pt auto; page-break-inside: avoid; }
h2 + p img, .appendix img { max-height: 120mm; }
td img { margin: 0; }
blockquote { margin: 5pt 0; padding: 3pt 8pt; border-left: 3px solid #2a78d6; background: #f5f8fd; font-size: 8.6pt; }
code { font-family: Consolas, monospace; font-size: 8.2pt; background: #f3f2ee; padding: 0 2pt; }
pre { background: #f3f2ee; padding: 5pt; font-size: 7.8pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #e1e0d9; margin: 8pt 0; }
a { color: #1c5cab; text-decoration: none; }
"""


def main(src="REPORT_2.md"):
    md_path = ROOT / src
    html_body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    html = f'<!doctype html><html><head><meta charset="utf-8"><title>{md_path.stem}</title>' \
           f'<style>{CSS}</style></head><body>{html_body}</body></html>'
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")

    edge = next((e for e in EDGE if e.exists()), None) or shutil.which("msedge")
    if edge is None:
        sys.exit(f"Edge not found; open {html_path.name} in a browser and print to PDF")
    pdf_path = md_path.with_suffix(".pdf")
    # a separate profile keeps a running Edge from taking over the call; Edge writes to an
    # ASCII temp path (the project path is Cyrillic), then the file is moved into place
    tmp = Path(tempfile.mkdtemp())
    out = tmp / "report.pdf"
    subprocess.run([str(edge), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--user-data-dir={tmp / 'profile'}", f"--print-to-pdf={out}", html_path.as_uri()],
                   check=True, timeout=180)
    for _ in range(120):  # Edge may return before the file is written
        if out.exists() and out.stat().st_size > 0:
            time.sleep(1)
            break
        time.sleep(0.5)
    shutil.move(out, pdf_path)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{pdf_path.name}: {pdf_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
