"""Assemble the public GitHub repository (code + docs, no data) -> D:\\highwaymm\\github\\HighwayMM
  python code\\report\\make_github.py
Copies code/ (no caches), the datasheet, README figures, and writes docs/RESULTS.md from the generated paper tables,
so every number in the repo matches the paper. Hand-written files (README.md, docs/*.md except RESULTS.md, LICENSE,
requirements.txt, .gitignore) live in code/report/github_static/ and are copied as-is.
Never copies tokens, raw footage, labels of individual annotators or anything under E:.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(r"D:\highwaymm")
OUT = ROOT / "github" / "HighwayMM"
STATIC = ROOT / "code" / "report" / "github_static"
TABLES = ROOT / "paper" / "tables"
FIGS = ROOT / "paper" / "figs"
FORBIDDEN = re.compile(r"hf_[A-Za-z0-9]{20,}|@gmail\.com|token\.txt", re.I)


def tex_to_md(path, title):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.endswith("\\\\") or line.startswith("\\"):
            continue
        cells = [c.strip() for c in line[:-2].split("&")]
        cells = [re.sub(r"\\textbf\{(.*?)\}", r"**\1**", c) for c in cells]
        cells = [c.replace("\\,$\\pm$\\,", " ± ").replace("$\\pm$", " ± ").replace("\\quad", "&nbsp;&nbsp;").replace("\\ ", " ")
                 .replace("$\\le$", "≤").replace("\\,", " ").replace("\\%", "%").replace("{", "").replace("}", "") for c in cells]
        rows.append(cells)
    head, body = rows[0], rows[1:]
    md = [f"### {title}", "", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md) + "\n"


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / "code", OUT / "code", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "github_static"))
    for f in STATIC.rglob("*"):
        if f.is_file():
            d = OUT / f.relative_to(STATIC)
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d)
    shutil.copy2(ROOT / "code" / "data" / "release_static" / "DATASHEET.md", OUT / "docs" / "DATASHEET.md")
    (OUT / "docs" / "assets").mkdir(parents=True, exist_ok=True)
    for f in ["teaser.png", "source_map.png", "coverage.png", "warning_tradeoff.png", "method_pipeline.png",
              "query_attention.png"]:
        shutil.copy2(FIGS / f, OUT / "docs" / "assets" / f)
    res = ["# Results", "",
           "All numbers are generated from logged runs (3 seeds, mean ± std) by `code/report/make_tables.py`,",
           "`code/report/warning_report.py` and `code/train/make_gold_table.py`, and are identical to the paper tables.", ""]
    res.append(tex_to_md(TABLES / "main_results_main.tex", "Main results on the eight external test sets (human, physics or sensor labels)"))
    res.append(tex_to_md(TABLES / "gold.tex", "Gold set of our own Korean CCTV (human-labelled clips, macro-F1)"))
    res.append(tex_to_md(TABLES / "ablations.tex", "Ablations"))
    res.append(tex_to_md(TABLES / "warning.tex", "Early-warning system on human-labelled camera sequences"))
    (OUT / "docs" / "RESULTS.md").write_text("\n".join(res), encoding="utf-8")
    bad = [str(p) for p in OUT.rglob("*") if p.is_file() and p.suffix in {".py", ".md", ".ps1", ".yaml", ".txt", ".html"}
           and FORBIDDEN.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert not bad, f"forbidden content in {bad}"
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(n, "files ->", OUT)


if __name__ == "__main__":
    main()
