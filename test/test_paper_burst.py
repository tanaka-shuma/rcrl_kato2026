
import numpy as np
import matplotlib.pyplot as plt

from snn4tanaka.option.option import get_opts
from snn4tanaka.simu import CulturedSNNCore


# ============================================================
# Simulation settings
# ============================================================

# DURATION_S = 30.0
# CHUNK_MS = 50.0
# BIN_MS = 5.0
# SEED = 1

DURATION_S = 120.0
CHUNK_MS = 50.0
BIN_MS = 5.0
SEED = 1

NE = 120
NI = 24
NTOT = NE + NI


# ============================================================
# Paper parameters for N=144, Ke=10
# ============================================================

def make_paper_params(with_slow):

    p = get_opts()

    p["seed"] = SEED
    p["dt"] = 0.2

    # RS, IB, FS, LTS
    p["N"] = np.array(
        [NE, 0, NI, 0],
        dtype=np.int32
    )

    # Single-module network
    p["mod"] = np.array(
        [1, 1],
        dtype=np.int32
    )

    # Ca-dependent adaptation
    p["tca"] = 2550.0
    p["gk"] = 1.7e-5

    # Clear all synaptic weights
    # g[fast/slow, intra/inter, pre E/I, post E/I]
    p["g"][:] = 0.0

    # Fast: AMPA / GABA_A
    p["g"][0, 0] = np.array([
        [7.80937e-4, 2.34281e-4],
        [1.20000e-4, 3.60000e-5]
    ])

    # Slow: NMDA / GABA_B
    if with_slow:
        p["g"][1, 0, 0, :] = (
            0.12 * p["g"][0, 0, 0, :]
        )
        p["g"][1, 0, 1, :] = (
            0.10 * p["g"][0, 0, 1, :]
        )

    return p


# ============================================================
# Run a single condition
# ============================================================

def run_condition(with_slow):

    p = make_paper_params(with_slow)

    core = CulturedSNNCore(p)

    # Original 1-second preparation
    core.warmup()

    all_ids = []
    all_times = []

    max_slow_g = 0.0

    n_chunks = int(
        round(DURATION_S * 1000.0 / CHUNK_MS)
    )

    for k in range(n_chunks):

        # No external input: spontaneous activity only
        core.advance(CHUNK_MS)

        all_ids.append(
            core.get_step_spike_ids()
        )
        all_times.append(
            core.get_step_spike_times()
        )

        max_slow_g = max(
            max_slow_g,
            float(np.max(core.get_slow_g()))
        )

    ids = np.concatenate(all_ids)
    times = np.concatenate(all_times)

    return ids, times, max_slow_g


# ============================================================
# Plot raster and population firing rate
# ============================================================

def plot_result(ids, times, label):

    total_ms = DURATION_S * 1000.0

    edges = np.arange(
        0.0,
        total_ms + BIN_MS,
        BIN_MS
    )

    counts, _ = np.histogram(
        times,
        bins=edges
    )

    # Number of spikes per second across the whole network
    population_rate = counts / (BIN_MS / 1000.0)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # Spike raster
    mask_e = ids < NE
    mask_i = ids >= NE

    axes[0].scatter(
        times[mask_e] / 1000.0,
        ids[mask_e],
        s=2,
        color="tab:blue",
        label="Excitatory"
    )

    axes[0].scatter(
        times[mask_i] / 1000.0,
        ids[mask_i],
        s=2,
        color="tab:red",
        label="Inhibitory"
    )

    axes[0].set_ylabel("Neuron ID")
    axes[0].set_ylim(-1, NTOT)
    axes[0].set_title(label)
    axes[0].legend(loc="upper right")

    # Population rate, 5-ms bins
    axes[1].plot(
        edges[:-1] / 1000.0,
        population_rate,
        linewidth=0.8
    )

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Spikes/s\n(population)")
    axes[1].set_xlim(0, DURATION_S)

    fig.tight_layout()

    filename = f"paper144_{label}.png"
    fig.savefig(filename, dpi=160)
    plt.close(fig)

    print("saved:", filename)


def analyze_bursts(times, label):

    bin_ms = 5.0
    merge_gap_ms = 200.0

    # 5-ms binning
    edges = np.arange(
        0.0,
        DURATION_S * 1000.0 + bin_ms,
        bin_ms
    )

    counts, _ = np.histogram(
        times,
        bins=edges
    )

    # Population mean firing rate
    # [spikes/s/neuron]
    rate = counts / (NTOT * bin_ms / 1000.0)

    # ==========================================
    # Otsu threshold
    # ==========================================
    values, freq = np.unique(
        rate,
        return_counts=True
    )

    if len(values) < 2:
        print("No detectable bursts:", label)
        return

    weight = np.cumsum(freq)
    weighted_sum = np.cumsum(values * freq)

    total_weight = weight[-1]
    total_sum = weighted_sum[-1]

    w0 = weight[:-1]
    w1 = total_weight - w0

    mu0 = weighted_sum[:-1] / w0
    mu1 = (
        total_sum - weighted_sum[:-1]
    ) / w1

    between_variance = (
        (w0 / total_weight)
        * (w1 / total_weight)
        * (mu0 - mu1) ** 2
    )

    best = int(np.argmax(between_variance))

    threshold = (
        values[best] + values[best + 1]
    ) / 2.0

    # ==========================================
    # Identify active bins
    # ==========================================
    active = np.flatnonzero(rate >= threshold)

    if len(active) == 0:
        print("No bursts detected:", label)
        return

    # ==========================================
    # Merge bursts separated by < 200 ms
    # ==========================================
    bursts = []

    start = int(active[0])
    prev = int(active[0])

    for idx in active[1:]:

        idx = int(idx)

        gap_ms = (idx - prev - 1) * bin_ms

        if gap_ms >= merge_gap_ms:

            bursts.append(
                (start * bin_ms,
                 (prev + 1) * bin_ms)
            )

            start = idx

        prev = idx

    bursts.append(
        (start * bin_ms,
         (prev + 1) * bin_ms)
    )

    # ==========================================
    # Statistics
    # ==========================================
    durations = np.array([
        end - start
        for start, end in bursts
    ])

    frequency = len(bursts) / DURATION_S

    print("\n--- Burst analysis:", label, "---")
    print("Otsu threshold:", threshold)
    print("Number of bursts:", len(bursts))
    print("Burst frequency:", frequency, "events/s")
    print("Mean duration:", durations.mean(), "ms")

    if len(durations) > 1:
        print(
            "Duration SD:",
            durations.std(ddof=1),
            "ms"
        )

    print("Durations:", durations)


# ============================================================
# Compare fast-only and fast+slow
# ============================================================

for with_slow, label in [
    (False, "fast_only"),
    (True, "fast_slow")
]:

    print("\n============================")
    print("condition:", label)
    print("============================")

    ids, times, max_slow_g = run_condition(
        with_slow
    )

    # Save raster data for later analysis
    np.savez_compressed(
        f"paper144_{label}_seed{SEED}_{int(DURATION_S)}s.npz",
        ids=ids,
        times_ms=times,
        duration_s=DURATION_S,
        Ne=NE,
        Ni=NI
    )

    e_spikes = int(np.sum(ids < NE))
    i_spikes = int(np.sum(ids >= NE))

    firing_rate = (
        len(ids) / NTOT / DURATION_S
    )

    active_neurons = len(np.unique(ids))

    print("total spikes:", len(ids))
    print("E spikes:", e_spikes)
    print("I spikes:", i_spikes)

    print(
        "mean firing rate:",
        firing_rate,
        "spikes/s/neuron"
    )

    print(
        "active neurons:",
        active_neurons,
        "/",
        NTOT
    )

    print("max slow g:", max_slow_g)

    plot_result(ids, times, label)
    analyze_bursts(times, label)

print("\n=== FINISHED ===")
