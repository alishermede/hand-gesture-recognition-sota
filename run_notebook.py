"""Execute hand_gesture_recognition.ipynb in place, embedding all outputs."""

import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NB = "hand_gesture_recognition.ipynb"

nb = nbformat.read(NB, as_version=4)
ep = ExecutePreprocessor(timeout=7200, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": os.getcwd()}})
nbformat.write(nb, NB)

n_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
print(f"executed: {n_out} code cells produced output")
