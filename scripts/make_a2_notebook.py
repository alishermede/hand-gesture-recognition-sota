"""Convert a2_notebook_src.py (a `# %%` cell-marked script) into a2_experiments.ipynb.

Same converter as A1's make_notebook.py, pointed at the A2 files.
Usage: python scripts/make_a2_notebook.py [--kernel NAME]
"""

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "a2_notebook_src.py"
OUT = ROOT / "a2_experiments.ipynb"


def convert(kernel):
    cells, kind, buf = [], "code", []

    def flush():
        body = "\n".join(buf).strip("\n")
        if not body:
            return
        if kind == "markdown":
            text = "\n".join(
                line[2:] if line.startswith("# ") else ("" if line.strip() == "#" else line)
                for line in body.split("\n")
            )
            cells.append(nbf.v4.new_markdown_cell(text))
        else:
            cells.append(nbf.v4.new_code_cell(body))

    for line in SRC.read_text(encoding="utf-8").split("\n"):
        if line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": kernel},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, OUT)
    print(f"{OUT.name}: {len(cells)} cells "
          f"({sum(c.cell_type == 'markdown' for c in cells)} markdown, "
          f"{sum(c.cell_type == 'code' for c in cells)} code), kernel '{kernel}'")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="python3")
    convert(ap.parse_args().kernel)
