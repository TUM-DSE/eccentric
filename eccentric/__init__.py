import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for _path in (
    os.path.join(ROOT, "external", "qiskit_qec", "src"),
    os.path.join(ROOT, "external", "more-bacon-less-threshold", "src"),
):
    if _path not in sys.path:
        sys.path.append(_path)
