"""Compare two behaviour_check.py outputs: every quantity must match to 1e-12 (strings exactly).

    python compare.py <baseline .npz> <new .npz>
"""
import sys

import numpy as np

TOLERANCE = 1e-12
baseline, new = np.load(sys.argv[1]), np.load(sys.argv[2])
failures = []
if set(baseline.files) != set(new.files):
    failures.append(f"different quantities: {sorted(set(baseline.files) ^ set(new.files))}")
largest = 0.0
for name in sorted(set(baseline.files) & set(new.files)):
    a, b = baseline[name], new[name]
    if a.dtype.kind in "USO" or b.dtype.kind in "USO":
        if not np.array_equal(a, b):
            failures.append(f"{name}: text differs")
        continue
    if a.shape != b.shape:
        failures.append(f"{name}: shape {a.shape} vs {b.shape}")
        continue
    difference = float(np.max(np.abs(a.astype(complex) - b.astype(complex)), initial=0.0))
    largest = max(largest, difference)
    if difference > TOLERANCE:
        failures.append(f"{name}: differs by {difference:.3e}")
print(f"{len(baseline.files)} quantities, largest difference {largest:.3e}")
print("PASS" if not failures else "FAIL\n  " + "\n  ".join(failures))
sys.exit(1 if failures else 0)
