"""Python 2/3 RPMDrate bridge smoke test (XYZ -> q -> V, dVdq)."""
from __future__ import print_function
import sys
import numpy as np
from sub_250104 import initialize_potential, get_potential, close_potential

frames = []
with open(sys.argv[1]) as f:
    while True:
        line = f.readline()
        if not line:
            break
        if not line.strip():
            continue
        if int(line) != 10:
            raise ValueError("Expected 10 atoms")
        f.readline()
        rows = [f.readline().split() for _ in range(10)]
        if [row[0] for row in rows] != ['O','C','H','H','H','C','H','H','H','O']:
            raise ValueError("Incorrect atom order")
        frames.append([[float(v) for v in row[1:4]] for row in rows])
if not frames:
    raise ValueError("Empty XYZ")
q = np.transpose(np.asarray(frames), (2, 1, 0)) / 0.529177208
initialize_potential()
try:
    energy, gradient = get_potential(q)
    print("Energy (Hartree):", energy)
    print("Positive gradient (Hartree/bohr), shape:", gradient.shape)
    print(gradient)
finally:
    close_potential()
