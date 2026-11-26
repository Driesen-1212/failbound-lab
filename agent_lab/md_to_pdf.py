# -*- coding: utf-8 -*-
"""把 Markdown 说明排成 PDF —— 主要用于核验「≤5 页」这条硬要求。

🔴 为什么需要：5 页说明是官方的四件套之一，且有**页数上限**。
   在 Markdown 里数汉字估算页数非常不可靠（表格、图片、标题占的版面差异极大），
   而超页是要扣分的。所以必须用真实排版来量。

用法：
    python md_to_pdf.py <输入.md> [-o 输出.pdf] [--fontsize 10.5]

特性：
  - 自动注册系统中文字体（微软雅黑 / 宋体 / 黑体），中文不会变方块
  - 支持 标题 / 段落 / 表格 / 图片 / 引用块 / 代码块 / 列表
  - 结尾打印**实际页数**，超限会明确报警
"""

import argparse
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

FONT_CANDIDATES = [
    ("msyh", r"C:\Windows\Fonts\msyh.ttc"),
    ("msyhbd", r"C:\Windows\Fonts\msyhbd.ttc"),
    ("simsun", r"C:\Windows\Fonts\simsun.ttc"),
    ("simhei", r"C:\Windows\Fonts\simhei.ttf"),
    ("deng", r"C:\Windows\Fonts\Deng.ttf"),
]

BODY_FONT = None


def register_fonts():
    """注册第一个可用的中文字体，返回字体名。"""
    global BODY_FONT
    for name, path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont(name, path))
                if BODY_FONT is None:
                    BODY_FONT = name
            except Exception:
                continue
    if BODY_FONT is None:
        raise SystemExit("找不到可用的中文字体（msyh / simsun / simhei / Deng）")
    return BODY_FONT


def styles(fs):
    b = BODY_FONT
    ss = getSampleStyleSheet()
    return {
        "h1": ParagraphStyle("h1", fontName=b, fontSize=fs + 5, leading=fs + 9,
                             spaceBefore=8, spaceAfter=5,
                             textColor=colors.HexColor("#1a3d5c")),
        "h2": ParagraphStyle("h2", fontName=b, fontSize=fs + 2, leading=fs + 6,
                             spaceBefore=6, spaceAfter=3,
                             textColor=colors.HexColor("#24506e")),
        "h3": ParagraphStyle("h3", fontName=b, fontSize=fs + 0.5, leading=fs + 4,
                             spaceBefore=4, spaceAfter=2),
        "p": ParagraphStyle("p", fontName=b, fontSize=fs, leading=fs * 1.45,
                            alignment=TA_LEFT, spaceAfter=3),
        "quote": ParagraphStyle("quote", fontName=b, fontSize=fs - 0.5,
                                leading=fs * 1.4, leftIndent=8,
                                textColor=colors.HexColor("#555555"),
                                spaceAfter=3),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=fs - 1.5,
                               leading=fs * 1.2, leftIndent=6,
                               backColor=colors.HexColor("#f4f4f4")),
        "cap": ParagraphStyle("cap", fontName=b, fontSize=fs - 1.5,
                              leading=fs * 1.3, alignment=1,
                              textColor=colors.HexColor("#666666")),
        "li": ParagraphStyle("li", fontName=b, fontSize=fs, leading=fs * 1.45,
                             leftIndent=10, bulletIndent=2, spaceAfter=1),
        "cell": ParagraphStyle("cell", fontName=b, fontSize=fs - 1,
                               leading=fs * 1.3),
        "cellh": ParagraphStyle("cellh", fontName=b, fontSize=fs - 1,
                                leading=fs * 1.3,
                                textColor=colors.HexColor("#ffffff")),
    }


def esc(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def inline(t):
    """行内 markdown：**粗体** 与 `代码` → reportlab 标签。"""
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`(.+?)`", r'<font face="Courier">\1</font>', t)
    return t


def build(md_path, out_path, fs=10.5, max_pages=5):
    register_fonts()
    st = styles(fs)
    base = os.path.dirname(os.path.abspath(md_path))

    with open(md_path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=14 * mm, rightMargin=14 * mm,
        topMargin=12 * mm, bottomMargin=12 * mm,
        title=os.path.basename(md_path))

    flow, i = [], 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        # 代码块
        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            for b in buf:
                flow.append(Paragraph(esc(b) or "&nbsp;", st["code"]))
            flow.append(Spacer(1, 3))
            continue

        if not s:
            i += 1
            continue

        # 图片
        m = re.match(r"^!\[(.*?)\]\((.*?)\)$", s)
        if m:
            cap, rel = m.group(1), m.group(2)
            p = rel if os.path.isabs(rel) else os.path.join(base, rel)
            if os.path.exists(p):
                try:
                    img = Image(p)
                    maxw = A4[0] - 28 * mm
                    r = img.imageHeight / img.imageWidth
                    img.drawWidth = maxw
                    img.drawHeight = maxw * r
                    if img.drawHeight > 90 * mm:
                        img.drawHeight = 90 * mm
                        img.drawWidth = img.drawHeight / r
                    flow.append(img)
                    if cap:
                        flow.append(Paragraph(inline(cap), st["cap"]))
                    flow.append(Spacer(1, 3))
                except Exception as e:
                    flow.append(Paragraph(f"[图片载入失败 {rel}: {e}]", st["cap"]))
            else:
                flow.append(Paragraph(f"[图片不存在 {rel}]", st["cap"]))
            i += 1
            continue

        # 表格
        if s.startswith("|") and i + 1 < len(lines) and \
                re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            header, body = rows[0], rows[2:]
            data = [[Paragraph(inline(c), st["cellh"]) for c in header]]
            data += [[Paragraph(inline(c), st["cell"]) for c in r] for r in body]
            ncol = len(header)
            avail = A4[0] - 28 * mm
            t = Table(data, colWidths=[avail / ncol] * ncol, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24506e")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#f2f6f9")]),
            ]))
            flow.append(t)
            flow.append(Spacer(1, 4))
            continue

        if s.startswith("# "):
            flow.append(Paragraph(inline(s[2:]), st["h1"]))
        elif s.startswith("## "):
            flow.append(Paragraph(inline(s[3:]), st["h2"]))
        elif s.startswith("### "):
            flow.append(Paragraph(inline(s[4:]), st["h3"]))
        elif s.startswith("> "):
            flow.append(Paragraph(inline(s[2:]), st["quote"]))
        elif s.startswith("- "):
            flow.append(Paragraph("• " + inline(s[2:]), st["li"]))
        elif re.match(r"^\d+\. ", s):
            flow.append(Paragraph(inline(s), st["li"]))
        elif s == "---":
            flow.append(Spacer(1, 4))
        else:
            flow.append(Paragraph(inline(s), st["p"]))
        i += 1

    doc.build(flow)

    # 数页数
    with open(out_path, "rb") as f:
        n = len(re.findall(rb"/Type\s*/Page[^s]", f.read()))
    flag = "✅ 符合 ≤5 页" if n <= max_pages else f"❌ 超出上限（>{max_pages} 页）"
    print(f"[pdf] {out_path} | 实际 {n} 页 | {flag}")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--fontsize", type=float, default=10.5)
    ap.add_argument("--max-pages", type=int, default=5)
    a = ap.parse_args()
    out = a.out or os.path.splitext(a.md)[0] + ".pdf"
    n = build(a.md, out, fs=a.fontsize, max_pages=a.max_pages)
    sys.exit(0 if n <= a.max_pages else 1)


if __name__ == "__main__":
    main()
