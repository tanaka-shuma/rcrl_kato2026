
import numpy as np
from pathlib import Path

# Original simulation: 30 s
dat_files = list(
    Path("test_original_simu").glob("spkN144*.dat")
)

if len(dat_files) != 1:
    raise RuntimeError(
        f"Expected one .dat file, found {len(dat_files)}"
    )

original = np.loadtxt(
    dat_files[0],
    usecols=(0, 1)
)

# Integrated Core: use first 30 seconds of 120-s run
data = np.load(
    "paper144_fast_slow_seed1_120s.npz"
)

core_times = data["times_ms"]
core_ids = data["ids"]

mask = core_times < 30000.0
core_times = core_times[mask]
core_ids = core_ids[mask]

# Convert times into physical time-step indices
dt_ms = 0.2

original_ticks = np.rint(
    original[:, 0] * 1000.0 / dt_ms
).astype(np.int64)

original_ids = original[:, 1].astype(np.int64)

core_ticks = np.rint(
    core_times / dt_ms
).astype(np.int64)

core_ids = core_ids.astype(np.int64)

# Sort by time, then neuron ID
a = np.column_stack((original_ticks, original_ids))
b = np.column_stack((core_ticks, core_ids))

a = a[np.lexsort((a[:, 1], a[:, 0]))]
b = b[np.lexsort((b[:, 1], b[:, 0]))]

print("=== spike counts ===")
print("Original:", len(a))
print("Core    :", len(b))

print("\n=== exact event comparison ===")

same = np.array_equal(a, b)

print("All spike events identical:", same)

if not same and len(a) == len(b):
    mismatch = np.flatnonzero(
        np.any(a != b, axis=1)
    )
    print("Number of mismatched rows:", len(mismatch))

    if len(mismatch):
        i = mismatch[0]
        print("First mismatch index:", i)
        print("Original:", a[i])
        print("Core    :", b[i])

print("\n=== FINISHED ===")
