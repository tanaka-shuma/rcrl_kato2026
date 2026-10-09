
import numpy as np

BIN_MS = 5.0
THRESHOLD = 85.0
MERGE_GAP_MS = 200.0

for label in ["fast_only", "fast_slow"]:

    filename = f"paper144_{label}_seed1_120s.npz"
    data = np.load(filename)

    times = data["times_ms"]
    duration_s = float(data["duration_s"])
    ntot = int(data["Ne"] + data["Ni"])

    edges = np.arange(
        0.0,
        duration_s * 1000.0 + BIN_MS,
        BIN_MS
    )

    counts, _ = np.histogram(times, bins=edges)

    rate = counts / (ntot * BIN_MS / 1000.0)

    active = np.flatnonzero(rate >= THRESHOLD)

    if len(active) == 0:
        print(label, ": no bursts")
        continue

    bursts = []

    start = int(active[0])
    prev = int(active[0])

    for idx in active[1:]:
        idx = int(idx)

        gap_ms = (idx - prev - 1) * BIN_MS

        if gap_ms >= MERGE_GAP_MS:
            bursts.append((
                start * BIN_MS,
                (prev + 1) * BIN_MS
            ))
            start = idx

        prev = idx

    bursts.append((
        start * BIN_MS,
        (prev + 1) * BIN_MS
    ))

    durations = np.array([
        end - start for start, end in bursts
    ])

    print(f"\n=== {label} ===")
    print("threshold:", THRESHOLD)
    print("number of bursts:", len(bursts))
    print("frequency:", len(bursts) / duration_s)
    print("mean duration:", durations.mean())
    print("median duration:", np.median(durations))
    print("duration SD:", durations.std(ddof=1))
    print("first 10 durations:", durations[:10])
