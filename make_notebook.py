"""Convert notebook_src.py (a `# %%` cell-marked script) into hand_gesture_recognition.ipynb."""

import nbformat as nbf

SRC = "notebook_src.py"
OUT = "hand_gesture_recognition.ipynb"

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


for line in open(SRC, encoding="utf-8").read().split("\n"):
    if line.startswith("# %%"):
        flush()
        kind = "markdown" if "[markdown]" in line else "code"
        buf = []
    else:
        buf.append(line)
flush()

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, OUT)
print(f"{OUT}: {len(cells)} cells "
      f"({sum(c.cell_type == 'markdown' for c in cells)} markdown, "
      f"{sum(c.cell_type == 'code' for c in cells)} code)")
