"""SNN4 reservoir dynamics plotting, based on someplot.py's six-panel layout.

The unmodified someplot.py plots input, total internal current, membrane voltage,
spike raster, Ca and Q.  The SNN4 wrapper does not currently expose a verified
per-neuron total-current trace, so panel two displays E/I population spike rate.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save PNG without requiring a display server
import matplotlib.pyplot as plt
import numpy as np


def plot_dynamics_snn4(data, save_path, episode, cumulative_reward, ds_ms, n_e):
    """Create a six-panel single-episode PNG from collected SNN4 signals.

    Parameters
    ----------
    data : dict
        Keys: times_ms, input_e_mean_na, input_i_mean_na, input_e_max_na,
        input_i_max_na, spike_e, spike_i, v_samples, v_ids, ca_e_samples,
        ca_e_ids, ca_e_mean, ca_e_std, q_values, spike_times_ms, spike_ids.
        One value per RL step except spike_times_ms and spike_ids, which
        include every physical-step spike within the episode.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    time = np.asarray(data["times_ms"], dtype=float)
    q = np.asarray(data["q_values"], dtype=float)
    n_tot = data["n_tot"]
    max_ms = max(float(time[-1]), float(ds_ms))
    e_rate = np.asarray(data["spike_e"], dtype=float) * 1000.0 / (n_e * ds_ms)
    i_rate = np.asarray(data["spike_i"], dtype=float) * 1000.0 / ((n_tot - n_e) * ds_ms)

    fig, axes = plt.subplots(6, 1, figsize=(12, 16), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.15, 1.75, 1.2, 1]})
    fig.suptitle(
        f"SNN4 reservoir | Episode {episode + 1} | "
        f"steps={len(time)}, reward={cumulative_reward:.5f}", fontsize=14
    )

    # 1. Injected sensory input current (not total synaptic current)
    ax = axes[0]
    ax.plot(time, data["input_e_mean_na"], color="tab:blue", label="E mean", lw=1.3)
    ax.plot(time, data["input_i_mean_na"], color="tab:red", label="I mean", lw=1.3)
    ax.plot(time, data["input_e_max_na"], color="tab:blue", label="E max", ls="--", lw=0.9)
    ax.plot(time, data["input_i_max_na"], color="tab:red", label="I max", ls="--", lw=0.9)
    ax.set_ylabel("Injected input\n(nA)")
    ax.legend(loc="upper right", fontsize=8, ncol=4)

    # 2. Population spike rate per 50-ms interval
    ax = axes[1]
    ax.step(time, e_rate, where="mid", color="tab:blue", label="E rate", lw=1.2)
    ax.step(time, i_rate, where="mid", color="tab:red", label="I rate", lw=1.2)
    ax.set_ylabel("Population rate\n(spikes/s/neuron)")
    ax.legend(loc="upper right", fontsize=8)

    # 3. Selected membrane voltage traces at the RL sampling period
    ax = axes[2]
    voltage = np.asarray(data["v_samples"])
    for j, neuron_id in enumerate(data["v_ids"]):
        ax.plot(time, voltage[:, j], lw=0.95,
                label=f"E {neuron_id}" if neuron_id < n_e else f"I {neuron_id}",
                alpha=0.85)
    ax.set_ylabel("Membrane V\n(mV)")
    ax.legend(loc="upper right", ncol=min(7, len(data["v_ids"])), fontsize=7)
    ax.text(0.01, 0.03, "Sampled every 50 ms; spike peaks are not resolved",
            transform=ax.transAxes, fontsize=8)

    # 4. Full-resolution event raster from the Cython core
    ax = axes[3]
    ids = np.asarray(data["spike_ids"], dtype=np.int32)
    spike_t = np.asarray(data["spike_times_ms"], dtype=float)
    if ids.size:
        e_mask = ids < n_e
        ax.scatter(spike_t[e_mask], ids[e_mask], s=0.35, c="tab:blue",
                   rasterized=True, label="E", linewidths=0)
        ax.scatter(spike_t[~e_mask], ids[~e_mask], s=0.35, c="tab:red",
                   rasterized=True, label="I", linewidths=0)
        ax.legend(loc="upper right", fontsize=8, markerscale=8)
    ax.axhline(n_e - 0.5, color="0.45", ls=":", lw=0.7)
    ax.set_ylabel("Neuron ID\n(spike raster)")
    ax.set_ylim(-1, n_tot)

    # 5. Intracellular Ca: population average, spatial SD and selected neurons
    ax = axes[4]
    mean_e = np.asarray(data["ca_e_mean"], dtype=float)
    std_e = np.asarray(data["ca_e_std"], dtype=float)
    ax.plot(time, mean_e, color="black", lw=1.8, label="E population mean")
    ax.fill_between(time, mean_e - std_e, mean_e + std_e,
                    color="tab:gray", alpha=0.20, label="E mean +/- spatial SD")
    sampled_ca = np.asarray(data["ca_e_samples"])
    for j, nid in enumerate(data["ca_e_ids"]):
        ax.plot(time, sampled_ca[:, j], lw=0.9, alpha=0.55, label=f"E {nid}")
    ax.set_ylabel("Intracellular Ca\n(concentration)")
    ax.legend(loc="upper left", fontsize=7, ncol=4)

    # 6. Q values evaluated by the agent (before weight updates for that step)
    ax = axes[5]
    for a in range(q.shape[1]):
        ax.plot(time, q[:, a], lw=1.3, label=f"Q[{a}]")
    ax.set_ylabel("Action values\nQ(t)")
    ax.set_xlabel("Time since episode reset (ms)")
    ax.legend(loc="best", fontsize=8)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlim(0, max_ms)
    fig.tight_layout(rect=(0, 0, 1, 0.972))
    fig.savefig(save_path, dpi=170)
    plt.close(fig)
    return save_path
