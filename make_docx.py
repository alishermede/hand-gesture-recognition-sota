"""Convert REPORT.md into REPORT.docx, embedding the key figures."""
import re
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

SRC, OUT = "REPORT.md", "REPORT.docx"

# figure -> (heading line it follows, width cm, caption)
FIGS = {
    "1.3 Diversity, and its limits": [("figures/01_samples.png", 12.0, "Figure 1. Two frames from every clip of every class.")],
    "How much does this choice change the answer?": [("figures/08_leakage.png", 8.5, "Figure 2. The same SVM evaluated with a random frame split vs. held-out clips.")],
    "Confusion structure": [("figures/06_confusion_matrices.png", 13.0, "Figure 3. Confusion matrices on the held-out clips.")],
    "Training behaviour": [("figures/04_training_curves.png", 7.5, "Figure 4. Training curves: small CNN vs. ResNet-18 (dashed line = unfreezing).")],
}

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(1.8))
sec.top_margin = sec.bottom_margin = Cm(1.5)

st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
for name, size in (("Heading 1", 15), ("Heading 2", 12.5), ("Heading 3", 11)):
    h = doc.styles[name]
    h.font.name, h.font.size, h.font.bold = "Calibri", Pt(size), True
    h.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    h.paragraph_format.space_before, h.paragraph_format.space_after = Pt(8), Pt(3)

INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)")

def add_inline(p, text):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("`"):
            r = p.add_run(part[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9.5)
        elif part.startswith("*") and len(part) > 2:
            r = p.add_run(part[1:-1]); r.italic = True
        else:
            p.add_run(part)

def shade(cell, color):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color)
    tcPr.append(sh)

def add_table(rows):
    rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    t = doc.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            cell = t.cell(i, j); cell.text = ""
            p = cell.paragraphs[0]; p.paragraph_format.space_after = Pt(0)
            add_inline(p, c)
            for run in p.runs:
                run.font.size = Pt(9.5)
                if i == 0: run.bold = True
            if i == 0: shade(cell, "DCE6F1")
    sp = doc.add_paragraph(); sp.paragraph_format.space_after = Pt(0)
    sp.add_run(" ").font.size = Pt(3)
    # keep each table on one page
    for row in t.rows:
        for cell in row.cells:
            for cp in cell.paragraphs:
                cp.paragraph_format.keep_with_next = True

def add_figs(key):
    for path, w, cap in FIGS.get(key, []):
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0); p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(path, width=Cm(w))
        c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = c.add_run(cap); r.italic = True; r.font.size = Pt(9)

lines = open(SRC, encoding="utf-8").read().split("\n")
i, para, last_heading = 0, [], None
# skip the appendix section (file list is for the repo, not the report)
end = next(k for k, l in enumerate(lines) if l.startswith("## Appendix"))
lines = lines[:end]

def flush():
    global para
    if para:
        p = doc.add_paragraph(); add_inline(p, " ".join(s.strip() for s in para)); para = []

def figs_after_section(next_heading_text):
    pass

pending_key = None
while i < len(lines):
    l = lines[i]
    if l.startswith("#"):
        flush()
        if pending_key: add_figs(pending_key); pending_key = None
        lvl = len(l) - len(l.lstrip("#")); text = l.lstrip("#").strip()
        if lvl == 1:
            t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = t.add_run(text); r.bold = True; r.font.size = Pt(19); r.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        else:
            doc.add_heading(text, level=lvl - 1)
        last_heading = text
        if text in FIGS: pending_key = text
    elif l.strip() == "---" or not l.strip():
        flush()
    elif l.startswith("|"):
        flush(); block = []
        while i < len(lines) and lines[i].startswith("|"):
            block.append(lines[i]); i += 1
        add_table(block); continue
    elif re.match(r"^(\* |- )", l):
        flush(); p = doc.add_paragraph(style="List Bullet")
        text = l[2:]; i += 1
        while i < len(lines) and lines[i].startswith("  ") and lines[i].strip():
            text += " " + lines[i].strip(); i += 1
        add_inline(p, text); p.paragraph_format.space_after = Pt(2); continue
    elif re.match(r"^\d+\. ", l):
        flush(); p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.9); p.paragraph_format.first_line_indent = Cm(-0.6)
        text = l; i += 1
        while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
            text += " " + lines[i].strip(); i += 1
        add_inline(p, text); p.paragraph_format.space_after = Pt(2); continue
    else:
        para.append(l)
    i += 1
flush()
if pending_key: add_figs(pending_key)

doc.save(OUT)
print("saved", OUT)
