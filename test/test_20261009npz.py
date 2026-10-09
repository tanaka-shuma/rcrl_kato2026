import numpy as np
import matplotlib.pyplot as plt

path = (
    "reservoir_plot/2026-10-09-175140-02ms/"
    "snn4_episode_001_02ms.npz"
)
d = np.load(path)

t = d["physical_times_ms"]
v = d["v_physical"]
v_ids = d["v_ids"]

nid = 0
j = np.where(v_ids == nid)[0][0]

# 150～300 ms内の最大膜電位を探す
mask = (t >= 150) & (t <= 300)
idx = np.where(mask)[0][np.argmax(v[mask, j])]
peak_t = t[idx]

print(f"E{nid} peak: {v[idx, j]:.3f} mV")
print(f"Time: {peak_t:.3f} ms")

# ピーク周辺の拡大図
spk = d["spike_times_ms"][
    d["spike_ids"] == nid
]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(t, v[:, j], label=f"E{nid} V", lw=1)
ax.axhline(-54, ls="--", color="gray",
           label="Threshold")

for st in spk:
    if peak_t - 5 <= st <= peak_t + 10:
        ax.axvline(st, color="red", alpha=0.3, lw=0.8)

ax.set_xlim(peak_t - 5, peak_t + 10)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Membrane potential (mV)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("test.png")