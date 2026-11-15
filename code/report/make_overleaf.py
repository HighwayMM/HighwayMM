"""Self-contained Overleaf project of the paper (main + supplement) -> paper/overleaf/ and paper/HighwayMM_overleaf.zip

Overleaf: New Project -> Upload Project -> HighwayMM_overleaf.zip. Compiler: pdfLaTeX (default). Main document: main.tex.
The supplement is supp.tex (Menu -> Main document -> supp.tex to compile it).
Sources stay in paper/latex, paper/tables and paper/figs; this script copies them into one tree and rewrites the
../tables and ../figs paths. Re-run it after every change; never edit paper/overleaf by hand.
"""
import re
import shutil
import zipfile
from pathlib import Path

P = Path(r"D:\highwaymm\paper")
SRC = P / "latex"
OUT = P / "overleaf"
KEEP_SEC = ["0_abstract", "1_intro", "2_related", "3_dataset", "4_method", "5_experiments", "6_conclusion"]


def fix(text):
    return text.replace("../tables/", "tables/").replace("../figs/", "figs/")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "sec").mkdir(parents=True)
    (OUT / "fig").mkdir()
    (OUT / "tables").mkdir()
    (OUT / "figs").mkdir()
    for f in ["main.tex", "supp.tex", "preamble.tex", "main.bib", "cvpr.sty", "ieeenat_fullname.bst"]:
        s = SRC / f
        if s.suffix == ".tex":
            (OUT / f).write_text(fix(s.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copy2(s, OUT / f)
    for n in KEEP_SEC:
        (OUT / "sec" / f"{n}.tex").write_text(fix((SRC / "sec" / f"{n}.tex").read_text(encoding="utf-8")), encoding="utf-8")
    for s in (SRC / "fig").glob("*.tex"):
        (OUT / "fig" / s.name).write_text(fix(s.read_text(encoding="utf-8")), encoding="utf-8")
    # only the tables and figures the documents actually use
    used = "\n".join(p.read_text(encoding="utf-8") for p in OUT.rglob("*.tex"))
    for kind, folder in (("tables", P / "tables"), ("figs", P / "figs")):
        for name in sorted(set(re.findall(kind + r"/([A-Za-z0-9_.\-]+)", used))):
            src = folder / name
            if not src.exists() and kind == "figs":
                src = folder / (name + ".pdf")
            assert src.exists(), f"missing {kind}/{name}"
            shutil.copy2(src, OUT / kind / src.name)
    (OUT / "README.md").write_text(
        "# HighwayMM - anonymous submission (Overleaf project)\n\n"
        "- Compiler: **pdfLaTeX** (Overleaf default). Main document: `main.tex` (8 pages + references).\n"
        "- Supplement: `supp.tex` (Menu > Main document > supp.tex).\n"
        "- `tables/*.tex` and `figs/*` are GENERATED from the experiment logs (`code/report/*.py`); edit captions and text in\n"
        "  `sec/*.tex`, but change numbers only by re-running the generators, then re-export this project\n"
        "  (`python code/report/make_overleaf.py`).\n"
        "- Search for `TODO(authors)` before submission.\n", encoding="utf-8")
    z = P / "HighwayMM_overleaf.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(OUT).as_posix())
    print("files:", sum(1 for f in OUT.rglob("*") if f.is_file()), "->", z, round(z.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
