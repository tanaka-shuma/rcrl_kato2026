"""Detailed (0.2-ms) membrane-potential/Ca/raster plotting for SNN4 RTDL.

Place this, snn4_physical_recorder.py, and someplot_snn4_02ms.py in
~/rcrl_k/ and run:
    python test_plot_snn4_02ms.py --episodes 1
    python test_plot_snn4_02ms.py --episodes 5 --readout raw
    python test_plot_snn4_02ms.py --episodes 1 --full-ca

RL actions/sensory inputs/TD updates are STILL performed every 50 ms.
Only the inner Cython advance is split into 0.2-ms calls when recording.
"""

import argparse
import importlib
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

import common_configurator as common
from test.snn4_physical_recorder import RecordingCoreProxy
from someplot_snn4_02ms import plot_dynamics_snn4


def arguments():
    p = argparse.ArgumentParser(description="0.2-ms SNN4 V/Ca and spike raster")
    p.add_argument("--readout", choices=["centered", "raw"], default="centered")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--scale", type=float, default=0.02)
    p.add_argument("--prerun-ms", type=float, default=11000.0)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--full-v", action="store_true",
                   help="Save all 576 voltage traces every 0.2 ms (larger files)")
    p.add_argument("--full-ca", action="store_true",
                   help="Save all 480 excitatory Ca traces every 0.2 ms (larger files)")
    p.add_argument("--v-ids", type=str, default="0,1,20,100,480,481",
                   help="Neuron IDs for the 0.2-ms voltage trace")
    p.add_argument("--ca-ids", type=str, default="0,1,20,100",
                   help="Excitatory neuron IDs for the 0.2-ms Ca trace")
    return p.parse_args()


def parse_ids(value):
    return np.array([int(x.strip()) for x in value.split(",") if x.strip()],
                    dtype=np.intp)


@contextmanager
def physical_recording(reservoir, dt, voltage_ids, calcium_ids, n_e,
                       full_ca=False, full_v=False):
    real_core = reservoir.core
    tracer = RecordingCoreProxy(real_core, dt, voltage_ids, calcium_ids,
                                n_e, record_full_ca=full_ca, record_full_v=full_v)

    # Some SNN wrappers retain aliases such as self.snn / self.simulator
    # pointing to the same Cython core. Replace *all* direct aliases rather
    # than replacing only self.core and accidentally bypassing the recorder.
    direct_aliases = [name for name, value in vars(reservoir).items()
                      if value is real_core]
    if "core" not in direct_aliases:
        direct_aliases.append("core")
    for name in direct_aliases:
        setattr(reservoir, name, tracer)
    print("Recording proxy installed on:", direct_aliases, flush=True)
    try:
        yield tracer
    finally:
        # Restore every alias. Keep the original core and its neural state.
        for name in direct_aliases:
            setattr(reservoir, name, real_core)


def main():
    args = arguments()
    # The SNN4 core calls snn4tanaka.option.option.get_opts(), which also
    # parses sys.argv.  The plotting options were already parsed above, so
    # hide them from the embedded parser (e.g. --episodes, --full-ca).
    sys.argv = sys.argv[:1]
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

    n_e = int(c.Nx - int(c.Nx * c.Irate))
    v_ids = parse_ids(args.v_ids)
    ca_ids = parse_ids(args.ca_ids)
    outdir = Path("reservoir_plot") / datetime.now().strftime("%Y-%m-%d-%H%M%S-02ms")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== SNN4: physical dt-resolved plot ===")
    print("RL step:", c.ds, "ms | physical step:", c.dt, "ms")
    print("readout:", args.readout, "scale:", args.scale,
          "episodes:", args.episodes, "full Ca:", args.full_ca, "full V:", args.full_v)
    print("saving to:", outdir, flush=True)

    try:
        for episode in range(args.episodes):
            agent.reset(episode)
            state, reward, done, info = env.reset()
            action = int(agent.a)
            r = agent.reservoir
            reward_sum = 0.0
            terminated = False
            data = {key: [] for key in [
                "times_ms", "input_e_mean_na", "input_i_mean_na",
                "input_e_max_na", "input_i_max_na", "spike_e", "spike_i",
                "q_values", "spike_times_ms_parts", "spike_ids_parts"
            ]}
            data.update(n_tot=int(c.Nx), v_ids=v_ids, ca_e_ids=ca_ids)

            with physical_recording(r, c.dt, v_ids, ca_ids, n_e,
                                    full_ca=args.full_ca, full_v=args.full_v) as tracer:
                core_origin_ms = tracer.origin_ms
                print(f"episode={episode} core origin={core_origin_ms:.1f} ms", flush=True)

                for k in range(args.max_steps):
                    state, reward, done, info = env.step(action)
                    action = int(agent.get_action(state, reward, done))
                    reward_sum += float(reward)

                    expected_physical = (k + 1) * int(round(float(c.ds) / float(c.dt)))
                    if tracer.physical_steps != expected_physical:
                        # Fail at the first mismatching step with useful details.
                        actual_elapsed = float(tracer.core.get_elapsed_ms()) - core_origin_ms
                        raise RuntimeError(
                            "Recorder was bypassed or advanced for the wrong duration: "
                            f"RL step={k+1}, recording calls={tracer.calls}, "
                            f"recorded physical steps={tracer.physical_steps}, "
                            f"expected={expected_physical}, "
                            f"actual core elapsed={actual_elapsed:.3f} ms. "
                            "Please show this output and the "
                            "models/snn_reservoir_snn4.py step() method."
                        )

                    ids = np.asarray(r.core.get_step_spike_ids(), dtype=np.int32)
                    ts_core = np.asarray(r.core.get_step_spike_times(), dtype=float)
                    ts = ts_core - core_origin_ms
                    if ids.size != ts.size:
                        raise RuntimeError("Spike IDs/times length mismatch")
                    if ts.size:
                        low = k * float(c.ds)
                        high = (k + 1) * float(c.ds)
                        if ts.min() < low - 1e-3 or ts.max() > high + 1e-3:
                            raise RuntimeError(f"Spike times outside RL step {k}: {ts.min()}..{ts.max()}")
                    data["spike_ids_parts"].append(ids.copy())
                    data["spike_times_ms_parts"].append(ts.copy())
                    data["spike_e"].append(int(np.count_nonzero(ids < n_e)))
                    data["spike_i"].append(int(np.count_nonzero(ids >= n_e)))

                    # Wrapper I_input is the applied external current, in microamperes.
                    injected_na = np.asarray(r.I_input, dtype=float) * 1000.0
                    if injected_na.shape != (c.Nx,):
                        raise ValueError("Unexpected I_input shape")
                    data["input_e_mean_na"].append(float(injected_na[:n_e].mean()))
                    data["input_i_mean_na"].append(float(injected_na[n_e:].mean()))
                    data["input_e_max_na"].append(float(injected_na[:n_e].max()))
                    data["input_i_max_na"].append(float(injected_na[n_e:].max()))

                    q = np.asarray(agent.q, dtype=float)
                    if not np.all(np.isfinite(q)):
                        raise RuntimeError("Non-finite Q")
                    data["q_values"].append(q.copy())
                    data["times_ms"].append((k + 1) * float(c.ds))

                    if done:
                        terminated = True
                        break
                    if hasattr(env, "handle_event"):
                        env.handle_event()

                trace = tracer.export()
                data.update(trace)
                expected_samples = len(data["times_ms"]) * int(round(float(c.ds) / float(c.dt))) + 1
                if len(trace["physical_times_ms"]) != expected_samples:
                    raise RuntimeError(
                        "0.2-ms sample count mismatch: "
                        f"actual={len(trace['physical_times_ms'])}, "
                        f"expected={expected_samples}, "
                        f"recording calls={tracer.calls}, "
                        f"physical steps={tracer.physical_steps}"
                    )

            # Underlying Cython core restored by 'with' before next reset.
            data["spike_ids"] = (np.concatenate(data.pop("spike_ids_parts"))
                                 if data["spike_ids_parts"] else np.empty(0, dtype=np.int32))
            data["spike_times_ms"] = (np.concatenate(data.pop("spike_times_ms_parts"))
                                      if data["spike_times_ms_parts"] else np.empty(0))
            assert len(data["spike_ids"]) == sum(data["spike_e"]) + sum(data["spike_i"])

            png = outdir / f"snn4_{args.readout}_episode_{episode+1:03d}_02ms.png"
            plot_dynamics_snn4(data, png, episode, reward_sum, float(c.ds), n_e)
            npz = outdir / f"snn4_episode_{episode+1:03d}_02ms.npz"
            np.savez_compressed(npz, **{key: np.asarray(value) for key, value in data.items()})
            print(f"episode={episode} steps={len(data['times_ms'])} "
                  f"phys_samples={len(trace['physical_times_ms'])} "
                  f"events={len(data['spike_ids'])} "
                  f"reward={reward_sum:.5f} done={terminated}")
            print("saved:", png)
            print("saved:", npz, flush=True)
    finally:
        env.close()

    print("=== FINISHED ===")


if __name__ == "__main__":
    main()
