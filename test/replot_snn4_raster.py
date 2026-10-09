"""Replot already-saved SNN4 episode NPZ files with episode-relative raster times.

Usage:
    python replot_snn4_raster.py reservoir_plot/YOUR_TIMESTAMP --offset-ms 11000

Requires someplot_snn4.py in the same working directory.
"""
import argparse
from pathlib import Path
import numpy as np
from someplot_snn4 import plot_dynamics_snn4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plot_dir", type=Path)
    ap.add_argument("--offset-ms", type=float, default=11000.0,
                    help="Pre-run duration included in core spike timestamps")
    ap.add_argument("--dt-rl-ms", type=float, default=50.0)
    ap.add_argument("--rewards", nargs="*", type=float, default=None,
                    help="Optional episode cumulative rewards in order")
    args = ap.parse_args()
    paths = sorted(args.plot_dir.glob("snn4_episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No snn4_episode_*.npz found in {args.plot_dir}")

    for ep, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as npz:
            data = {k: npz[k] for k in npz.files}
        spike_t = np.asarray(data["spike_times_ms"], dtype=float)
        if not spike_t.size:
            print(f"{path.name}: no spikes; skipping")
            continue
        max_episode_ms = float(data["times_ms"][-1])
        raw_range = (spike_t.min(), spike_t.max())
        if spike_t.min() >= -1e-3 and spike_t.max() <= max_episode_ms + 1e-3:
            print(f"{path.name}: timestamps already relative; not subtracting offset")
        else:
            data["spike_times_ms"] = spike_t - args.offset_ms
        fixed_t = np.asarray(data["spike_times_ms"])
        print(f"{path.name}: raw range={raw_range[0]:.1f}..{raw_range[1]:.1f}, "
              f"fixed range={fixed_t.min():.1f}..{fixed_t.max():.1f}, "
              f"episode=0..{max_episode_ms:.1f} ms")
        if fixed_t.min() < -1e-3 or fixed_t.max() > max_episode_ms + 1e-3:
            raise RuntimeError("Spike events still outside episode time range. Check --offset-ms")
        if len(fixed_t) != int(np.sum(data["spike_e"]) + np.sum(data["spike_i"])):
            raise RuntimeError("Spike-rate total does not match raster event count")
        n_tot = int(data["n_tot"])
        n_e = n_tot - int(round(n_tot / 6))
        out = args.plot_dir / f"snn4_episode_{ep + 1:03d}_raster_fixed.png"
        reward = (args.rewards[ep] if args.rewards is not None and ep < len(args.rewards)
                  else float("nan"))
        plot_dynamics_snn4(data, out, ep, reward, args.dt_rl_ms, n_e)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
