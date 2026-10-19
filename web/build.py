"""Inline web/dna_align.js into web/template.html -> web/index.html (one self-contained page).

Run:  python web/build.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
html = (HERE / "template.html").read_text(encoding="utf-8")
js = (HERE / "dna_align.js").read_text(encoding="utf-8")
out = html.replace("/*__DNA_ALIGN_JS__*/", js)
(HERE / "index.html").write_text(out, encoding="utf-8")
print(f"wrote {HERE / 'index.html'} ({len(out):,} bytes)")
