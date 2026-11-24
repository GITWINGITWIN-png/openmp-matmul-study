#!/usr/bin/env python3
"""แปลง report.md → PDF ด้วย pandoc + XeLaTeX (ภาษาไทย, บรรณานุกรม IEEE, อ้างอิงรูป/ตาราง)"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pypandoc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEX = HERE / "tex"
BUILD = HERE / "build"

PANDOC_VARS = {
    "documentclass": "article",
    "classoption": "11pt",
    "papersize": "a4",
    "geometry": "margin=2.5cm",
    "mainfont": "Sarabun",
    "sansfont": "Sarabun",
    "monofont": "DejaVu Sans Mono",
    "monofontoptions": "Scale=0.82",
    "colorlinks": "true",
    "linkcolor": "accent",
    "urlcolor": "accent",
    "citecolor": "accent",
    "toccolor": "black",
    "biblio-style": "ieee",
    "biblio-title": "เอกสารอ้างอิง",
    "toc-title": "สารบัญ",
}


def front_matter(text):
    m = re.match(r"^---\n(.*?)\n---", text, re.S)
    meta = {}
    if m:
        for line in m.group(1).splitlines():
            k, sep, v = line.partition(":")
            if sep and not line.startswith(" "):
                meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta


def lint(md_path, text):
    ok = True
    body = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    todos = len(re.findall(r"\[TODO", body))
    if todos:
        print(f"  ⚠️  ยังเหลือ [TODO …] {todos} จุด (จะไปโผล่ใน PDF)")
    for img in re.findall(r"!\[[^\]]*\]\(([^)\s]+)", body):
        if img.lower().endswith(".gif"):
            print(f"  ❌ {img}: PDF ใส่ GIF ไม่ได้ → ใช้ still_*.png แทน")
            ok = False
        elif not (md_path.parent / img).exists():
            print(f"  ❌ ไม่พบรูป {img}")
            ok = False
    return ok


def latex_escape(s):
    return re.sub(r"([&%$#_{}])", r"\\\1", s)


def main():
    md_path = (Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "report.md").resolve()
    text = md_path.read_text(encoding="utf-8")
    meta = front_matter(text)
    print(f"build {md_path.name}")
    if not lint(md_path, text):
        sys.exit("หยุด: แก้ปัญหาด้านบนก่อน")

    BUILD.mkdir(exist_ok=True)
    right = ", ".join(x for x in (meta.get("author", ""), meta.get("student-id", "")) if x)
    (BUILD / "meta.tex").write_text(
        "\\newcommand{\\reportheaderright}{" + latex_escape(right) + "}\n", encoding="utf-8")

    tex_out = BUILD / "report.tex"
    args = ["--standalone", f"--template={TEX / 'template.tex'}",
            f"--include-in-header={BUILD / 'meta.tex'}",
            f"--include-in-header={TEX / 'preamble.tex'}",
            f"--lua-filter={TEX / 'crossref.lua'}",
            "--biblatex", "--number-sections", "--toc",
            f"--resource-path={md_path.parent}"]
    args += [f"--variable={k}:{v}" for k, v in PANDOC_VARS.items()]
    if (md_path.parent / "references.bib").exists() and "bibliography" not in meta:
        args.append(f"--metadata=bibliography:{md_path.parent / 'references.bib'}")
    print("  1/3 pandoc  → build/report.tex")
    pypandoc.convert_file(str(md_path), "latex", format="markdown", outputfile=str(tex_out),
                          extra_args=args)

    print("  2/3 xelatex → build/report.pdf (latexmk + biber)")
    r = subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error",
                        f"-outdir={BUILD}", str(tex_out)],
                       cwd=md_path.parent, capture_output=True, text=True)
    if r.returncode != 0:
        log = (BUILD / "report.log").read_text(errors="ignore") if (BUILD / "report.log").exists() else r.stdout
        errors = [l for l in log.splitlines() if l.startswith("!") or "Error" in l][:10]
        print("\n".join(errors) or r.stdout[-2000:])
        sys.exit(f"xelatex ล้มเหลว — ดู {BUILD / 'report.log'}")
    log = (BUILD / "report.log").read_text(errors="ignore")
    missing = sorted(set(re.findall(r"Missing character: There is no (.+?) in font", log)))
    if missing:
        print(f"  ⚠️  ฟอนต์ไม่มีตัวอักษร: {' '.join(missing[:10])} (อักษรเหล่านี้จะหายจาก PDF)")
    undefined = sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", log)))
    if undefined:
        print(f"  ⚠️  อ้างอิงที่ไม่มีอยู่จริง: {', '.join(undefined)}")

    print("  3/3 copy")
    pdf = BUILD / "report.pdf"
    preview = HERE / "report.pdf"
    shutil.copy2(pdf, preview)
    print(f"  ✅ {preview.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
