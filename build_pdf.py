"""
Render thesis.md into a professionally-typeset research-paper PDF (thesis.pdf).

Bespoke converter for this paper's markdown subset: #/##/### headings, pipe
tables, fenced code blocks (used for equations), images with captions,
bullets, bold/italic/inline-code, and horizontal rules.

Run:  python build_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable, Image, KeepTogether, PageBreak, Paragraph, Preformatted,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

SRC = Path("thesis.md")
OUT = Path("thesis.pdf")

# ---- fonts: DejaVu covers the Greek/math glyphs the paper uses ------------- #
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
SERIF, SERIF_B, SERIF_I = "DejaVuSerif", "DejaVuSerif-Bold", "DejaVuSerif-Italic"
SANS_B, MONO = "DejaVuSans-Bold", "DejaVuSansMono"
pdfmetrics.registerFont(TTFont(SERIF, FONT_DIR / "DejaVuSerif.ttf"))
pdfmetrics.registerFont(TTFont(SERIF_B, FONT_DIR / "DejaVuSerif-Bold.ttf"))
pdfmetrics.registerFont(TTFont(SERIF_I, FONT_DIR / "DejaVuSerif-Italic.ttf"))
pdfmetrics.registerFont(TTFont(SANS_B, FONT_DIR / "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont(MONO, FONT_DIR / "DejaVuSansMono.ttf"))
pdfmetrics.registerFont(TTFont("DejaVuSerif-BoldItalic",
                               FONT_DIR / "DejaVuSerif-BoldItalic.ttf"))
pdfmetrics.registerFontFamily(SERIF, normal=SERIF, bold=SERIF_B, italic=SERIF_I,
                              boldItalic="DejaVuSerif-BoldItalic")

INK = colors.HexColor("#111418")
MUTED = colors.HexColor("#5A6472")
ACCENT = colors.HexColor("#0E7490")
RULE = colors.HexColor("#C9D1DB")
CODE_BG = colors.HexColor("#F2F5F8")

S = dict(
    title=ParagraphStyle("title", fontName=SERIF_B, fontSize=16.5, leading=21,
                         alignment=TA_CENTER, textColor=INK, spaceAfter=4),
    subtitle=ParagraphStyle("subtitle", fontName=SERIF_I, fontSize=11.5, leading=15,
                            alignment=TA_CENTER, textColor=MUTED, spaceAfter=2),
    author=ParagraphStyle("author", fontName=SERIF, fontSize=9.5, leading=13,
                          alignment=TA_CENTER, textColor=MUTED, spaceBefore=6),
    h1=ParagraphStyle("h1", fontName=SANS_B, fontSize=11.5, leading=15, textColor=INK,
                      spaceBefore=14, spaceAfter=5),
    h2=ParagraphStyle("h2", fontName=SANS_B, fontSize=10, leading=13, textColor=ACCENT,
                      spaceBefore=10, spaceAfter=4),
    body=ParagraphStyle("body", fontName=SERIF, fontSize=9.3, leading=13.4,
                        alignment=TA_JUSTIFY, textColor=INK, spaceAfter=6),
    abstract=ParagraphStyle("abstract", fontName=SERIF, fontSize=8.8, leading=12.6,
                            alignment=TA_JUSTIFY, textColor=INK,
                            leftIndent=36, rightIndent=36, spaceAfter=6),
    bullet=ParagraphStyle("bullet", fontName=SERIF, fontSize=9.3, leading=13.2,
                          alignment=TA_JUSTIFY, textColor=INK, leftIndent=16,
                          bulletIndent=4, spaceAfter=3.5),
    code=ParagraphStyle("code", fontName=MONO, fontSize=8.2, leading=11.5,
                        textColor=INK, backColor=CODE_BG, leftIndent=22,
                        borderPadding=6, spaceBefore=4, spaceAfter=8),
    caption=ParagraphStyle("caption", fontName=SERIF_I, fontSize=8.2, leading=11,
                           alignment=TA_CENTER, textColor=MUTED,
                           spaceBefore=3, spaceAfter=10),
    ref=ParagraphStyle("ref", fontName=SERIF, fontSize=8.3, leading=11.6,
                       alignment=TA_LEFT, textColor=INK, spaceAfter=3,
                       leftIndent=14, firstLineIndent=-14),
)

BODY_W = letter[0] - 2 * 0.95 * inch


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline(t: str) -> str:
    t = esc(t)
    t = t.replace("\\*", "\x00")          # protect escaped asterisks (n\*)
    t = re.sub(r"`([^`]+)`",
               rf'<font face="{MONO}" size="8" color="#0E5A74">\1</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\w)\*([^*\n]+)\*(?!\w)", r"<i>\1</i>", t)
    return t.replace("\x00", "*")


def make_table(lines: list[str]) -> Table:
    rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append(cells)
    rows = [r for r in rows if not all(set(c) <= {"-", ":", " ", ""} for c in r)]
    cell_style = ParagraphStyle("cell", fontName=SERIF, fontSize=8.2, leading=10.6,
                                textColor=INK)
    head_style = ParagraphStyle("hcell", fontName=SANS_B, fontSize=8.2, leading=10.6,
                                textColor=INK)
    data = [[Paragraph(inline(c), head_style if i == 0 else cell_style)
             for c in row] for i, row in enumerate(rows)]
    t = Table(data, hAlign="CENTER", colWidths=[BODY_W / len(rows[0])] * len(rows[0]))
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, INK),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, INK),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F8FA")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def build_story() -> list:
    text = SRC.read_text()
    blocks = text.split("\n\n")
    story: list = []
    section = ""
    fig_no = 0

    i = 0
    while i < len(blocks):
        raw = blocks[i].strip()
        i += 1
        if not raw:
            continue

        if raw.startswith("# ") and not story:
            story.append(Paragraph(inline(raw[2:]), S["title"]))
            continue
        if raw.startswith("**A Simulation Study"):
            story.append(Paragraph(raw.strip("*"), S["subtitle"]))
            continue
        if raw.startswith("*Mathisha"):
            story.append(Paragraph(inline(raw.strip("*")), S["author"]))
            story.append(Spacer(1, 6))
            continue
        if raw == "---":
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE,
                                    spaceBefore=6, spaceAfter=10))
            continue
        if raw.startswith("## "):
            section = raw[3:].strip()
            story.append(Paragraph(inline(section), S["h1"]))
            continue
        if raw.startswith("### "):
            story.append(Paragraph(inline(raw[4:]), S["h2"]))
            continue
        if raw.startswith("```"):
            code_lines = raw.split("\n")
            body = "\n".join(ln for ln in code_lines if not ln.startswith("```"))
            while not raw.rstrip().endswith("```") and i < len(blocks):
                nxt = blocks[i]
                i += 1
                body += "\n\n" + "\n".join(ln for ln in nxt.split("\n")
                                           if not ln.startswith("```"))
                raw = nxt
            story.append(Preformatted(body.strip("\n"), S["code"]))
            continue
        m = re.match(r"!\[(.*)\]\((.*)\)", raw)
        if m:
            alt, path = m.group(1), m.group(2)
            fig_no += 1
            if Path(path).exists():
                from PIL import Image as PILImage
                w, h = PILImage.open(path).size
                img_w = BODY_W
                img = Image(path, width=img_w, height=img_w * h / w)
                story.append(KeepTogether([
                    Spacer(1, 4), img, Paragraph(inline(alt), S["caption"]),
                ]))
            continue
        if raw.startswith("|"):
            tbl_lines = [ln for ln in raw.split("\n") if ln.strip().startswith("|")]
            story.append(Spacer(1, 2))
            story.append(make_table(tbl_lines))
            story.append(Spacer(1, 8))
            continue
        if raw.startswith("- "):
            for item in re.split(r"\n(?=- )", raw):
                story.append(Paragraph(inline(item[2:].replace("\n", " ")),
                                       S["bullet"], bulletText="•"))
            story.append(Spacer(1, 3))
            continue
        if section.startswith("References") and raw.startswith("["):
            story.append(Paragraph(inline(raw.replace("\n", " ")), S["ref"]))
            continue

        style = S["abstract"] if section == "Abstract" else S["body"]
        story.append(Paragraph(inline(raw.replace("\n", " ")), style))

    return story


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(SERIF_I, 7.5)
    canvas.setFillColor(MUTED)
    if doc.page > 1:
        canvas.drawString(0.95 * inch, 0.55 * inch,
                          "The Economics of LLM Inference Under Memory-Bandwidth-Bound Decoding")
    canvas.drawRightString(letter[0] - 0.95 * inch, 0.55 * inch, f"{doc.page}")
    canvas.restoreState()


def main():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.95 * inch, rightMargin=0.95 * inch,
        topMargin=0.85 * inch, bottomMargin=0.9 * inch,
        title="The Economics of LLM Inference Under Memory-Bandwidth-Bound Decoding",
        author="Mathisha Samarawickrama",
        subject="Simulation study of LLM inference hardware economics",
    )
    doc.build(build_story(), onFirstPage=on_page, onLaterPages=on_page)
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
