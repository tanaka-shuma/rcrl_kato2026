"""Plot RTDL + cultured SNN4 internal dynamics in someplot.py's six-panel style.

Copy both this file and someplot_snn4.py to ~/rcrl_k, then run:
    python test_plot_snn4.py
    python test_plot_snn4.py --readout raw --episodes 1
    python test_plot_snn4.py --readout centered --scale 0.03 --episodes 5
"""

import argparse
import importlib
from datetime import datetime
from pathlib import Path

import numpy as np

import common_configurator as common
from someplot_snn4 import plot_dynamics_snn4


def read_arguments():
    parser = argparse.ArgumentParser(description="Plot cultured SNN4 episode dynamics")
    parser.add_argument("--readout", choices=["centered", "raw"], default="centered")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--scale", type=float, default=0.02)
    parser.add_argument("--prerun-ms", type=float, default=11000.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main():
    args = read_arguments()
    c = common.load_config("config.config_rl_ninerooms_rtdl_snn")
    c.rc_module = "models.snn_reservoir_snn4"
    c.rc_class = "SpikingNeuralNetwork"
    c.Nx = 576
    c.Irate = 1.0 / 6.0
    c.snn4_paper_condition = True
    c.snn4_prerun_ms = args.prerun_ms
    c.input_const_snn4 = args.scale
    c.eta_init *= 0.001
    c.eta_final *= 0.001
    c.eps_greedy = False
    c.seed = args.seed
    np.random.seed(int(c.seed))

    agent_module = ("models.agent_rtdl_centered" if args.readout == "centered"
                    else "models.agent_rtdl")
    Agent = importlib.import_module(agent_module).Agent
    agent = Agent(c)
    env = common.generate_instance(c, module=c.env_module, class_=c.env_class)
    agent.initialize()

    outdir = Path("reservoir_plot") / datetime.now().strftime("%Y-%m-%d-%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    n_e = int(c.Nx - int(c.Nx * c.Irate))  # paper N=576 -> E=480, I=96
    v_ids = np.array([0, 1, 20, 100, n_e, n_e + 1], dtype=int)
    ca_ids = np.array([0, 1, 20, 100], dtype=int)
    print("=== SNN4 dynamics plot ===")
    print("readout:", args.readout, "scale:", args.scale,
          "episodes:", args.episodes, "save dir:", outdir)

    try:
        for episode in range(args.episodes):
            agent.reset(episode)
            state, reward, done, info = env.reset()
            action = int(agent.a)  # initial executed action matches internal TD action
            core_origin_ms = float(agent.reservoir.core.get_elapsed_ms())
            print(f"episode={episode} core origin={core_origin_ms:.1f} ms")

            data = {k: [] for k in [
                "times_ms", "input_e_mean_na", "input_i_mean_na",
                "input_e_max_na", "input_i_max_na", "spike_e", "spike_i",
                "v_samples", "ca_e_samples", "ca_e_mean", "ca_e_std", "q_values",
                "spike_times_ms_parts", "spike_ids_parts"
            ]}
            data.update(n_tot=c.Nx, v_ids=v_ids, ca_e_ids=ca_ids)
            cumulative_reward = 0.0
            terminated = False

            for k in range(args.max_steps):
                state, reward, done, info = env.step(action)
                action = int(agent.get_action(state, reward, done))
                cumulative_reward += float(reward)
                r = agent.reservoir

                # Physical events, not the final single physical-step firing vector
                ids = np.asarray(r.core.get_step_spike_ids(), dtype=np.int32)
                times = np.asarray(r.core.get_step_spike_times(), dtype=np.float64)
                if len(ids) != len(times):
                    raise RuntimeError("Spike IDs and spike times have different lengths")
                # Core timestamps can include the SNN prerun (e.g. 11000 ms).
                # Convert to time since the current RL episode reset.
                times_episode = times - core_origin_ms
                if times_episode.size:
                    lo, hi = k * float(c.ds), (k + 1) * float(c.ds)
                    if times_episode.min() < lo - 1e-3 or times_episode.max() > hi + 1e-3:
                        raise RuntimeError(
                            f"Spike time outside step {k+1}: "
                            f"core={times.min():.2f}..{times.max():.2f}, "
                            f"episode={times_episode.min():.2f}..{times_episode.max():.2f}, "
                            f"expected {lo:.2f}..{hi:.2f} ms. "
                            f"core_origin={core_origin_ms:.2f}"
                        )
                data["spike_ids_parts"].append(ids.copy())
                data["spike_times_ms_parts"].append(times_episode.copy())
                data["spike_e"].append(int(np.count_nonzero(ids < n_e)))
                data["spike_i"].append(int(np.count_nonzero(ids >= n_e)))

                # r.I_input is the already scaled and clipped current in microampere.
                # Thus x1000 converts uA -> nA.
                if not hasattr(r, "I_input"):
                    raise AttributeError("SNN4 wrapper has no I_input. Check the wrapper's applied-current name.")
                i_na = np.asarray(r.I_input, dtype=float) * 1000.0
                if i_na.shape != (c.Nx,):
                    raise ValueError(f"Unexpected I_input shape: {i_na.shape}")
                data["input_e_mean_na"].append(float(i_na[:n_e].mean()))
                data["input_i_mean_na"].append(float(i_na[n_e:].mean()))
                data["input_e_max_na"].append(float(i_na[:n_e].max()))
                data["input_i_max_na"].append(float(i_na[n_e:].max()))

                # Membrane voltage and raw Ca regardless of centered/raw readout.
                v = np.asarray(r.core.get_v(), dtype=float)
                ca = np.asarray(r.core.get_ca(), dtype=float)
                q = np.asarray(agent.q, dtype=float)
                if v.shape != (c.Nx,) or ca.shape != (c.Nx,) or q.shape != (c.Ny,):
                    raise ValueError(f"Unexpected states: V={v.shape}, Ca={ca.shape}, Q={q.shape}")
                if not (np.all(np.isfinite(v)) and np.all(np.isfinite(ca))
                        and np.all(np.isfinite(q))):
                    raise RuntimeError("Non-finite neuronal state or Q value")
                data["v_samples"].append(v[v_ids].copy())
                data["ca_e_samples"].append(ca[ca_ids].copy())
                data["ca_e_mean"].append(float(ca[:n_e].mean()))
                data["ca_e_std"].append(float(ca[:n_e].std()))
                data["q_values"].append(q.copy())
                data["times_ms"].append((k + 1) * float(c.ds))

                if done:
                    terminated = True
                    break
                if hasattr(env, "handle_event"):
                    env.handle_event()

            data["spike_ids"] = (np.concatenate(data.pop("spike_ids_parts"))
                                 if data["spike_ids_parts"] else np.empty(0, dtype=np.int32))
            data["spike_times_ms"] = (np.concatenate(data.pop("spike_times_ms_parts"))
                                       if data["spike_times_ms_parts"] else np.empty(0))
            print(
                f"raster: n={len(data['spike_ids'])}, "
                f"range={data['spike_times_ms'].min():.1f}..{data['spike_times_ms'].max():.1f} ms"
                if len(data['spike_ids']) else "raster: no spikes"
            )
            assert len(data["spike_ids"]) == sum(data["spike_e"]) + sum(data["spike_i"])
            png_path = outdir / f"snn4_{args.readout}_episode_{episode + 1:03d}.png"
            plot_dynamics_snn4(data, png_path, episode, cumulative_reward, float(c.ds), n_e)

            # Save raw trace data for later analysis, without overwriting other trials
            raw_arrays = {key: np.asarray(val) for key, val in data.items()}
            np.savez_compressed(outdir / f"snn4_episode_{episode + 1:03d}.npz", **raw_arrays)
            print(f"episode={episode} steps={len(data['times_ms'])} "
                  f"reward={cumulative_reward:.5f} terminated={terminated} "
                  f"E_spikes={sum(data['spike_e'])} I_spikes={sum(data['spike_i'])}")
            print("saved:", png_path)
    finally:
        env.close()

    print("=== FINISHED ===")


if __name__ == "__main__":
    main()
