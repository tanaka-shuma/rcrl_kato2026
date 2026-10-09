import numpy as np

original = np.load(
    "reservoir_plot/2026-10-09-175140-02ms/"
    "snn4_episode_001_02ms.npz"
)

clipped = np.load(
    "reservoir_plot/2026-10-09-180854-02ms/"
    "snn4_episode_001_02ms.npz"
)

for key in [
    "spike_ids",
    "spike_times_ms",
    "ca_e_samples_physical",
    "ca_e_mean_physical",
    "q_values",
]:
    print(key, np.array_equal(original[key], clipped[key]))

print(
    "Max voltage difference:",
    np.max(np.abs(
        original["v_physical"] - clipped["v_physical"]
    ))
)