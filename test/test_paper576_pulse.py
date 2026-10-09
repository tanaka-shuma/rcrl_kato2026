
import numpy as np

import common_configurator as common
from models.snn_reservoir_snn4 import SpikingNeuralNetwork


def run_condition(current_na):

    c = common.load_config(
        "config.config_rl_ninerooms_rtdl_snn"
    )

    c.Nx = 576
    c.Irate = 1.0 / 6.0
    c.snn4_paper_condition = True
    c.snn4_prerun_ms = 11000.0
    c.input_const_snn4 = 1.0
    c.seed = 1

    reservoir = SpikingNeuralNetwork(c)
    reservoir.reset()

    initial_v = reservoir.neurons.v.copy()
    initial_ca = reservoir.r.copy()

    zero_input = np.zeros(c.Nx, dtype=np.float64)

    pulse_input = zero_input.copy()
    pulse_input[:48] = current_na * 0.001  # nA -> uA

    step_spikes = []
    snapshots = {}
    all_times = []

    for k in range(20):

        # Apply external current only during 0-50 ms
        if k == 0:
            reservoir.step(pulse_input)
        else:
            reservoir.step(zero_input)

        step_spikes.append(
            reservoir.core.get_step_spike_count()
        )

        times = (
            reservoir.core.get_step_spike_times()
            - reservoir.episode_time_origin_ms
        )
        all_times.extend(times)

        if (k + 1) in [1, 5, 10, 20]:
            t_ms = int((k + 1) * c.ds)

            snapshots[t_ms] = {
                "ca": reservoir.r.copy(),
                "v": reservoir.neurons.v.copy()
            }

    return {
        "initial_v": initial_v,
        "initial_ca": initial_ca,
        "step_spikes": np.array(step_spikes),
        "times": np.asarray(all_times),
        "snapshots": snapshots
    }


results = {}

for current_na in [0.0, 5.0, 10.0]:
    print(f"Running: {current_na} nA", flush=True)
    results[current_na] = run_condition(current_na)

baseline = results[0.0]

for current_na in [5.0, 10.0]:

    result = results[current_na]

    print(f"\n=== {current_na} nA, 50-ms pulse ===")

    print(
        "Initial V identical:",
        np.array_equal(
            baseline["initial_v"],
            result["initial_v"]
        )
    )
    print(
        "Initial Ca identical:",
        np.array_equal(
            baseline["initial_ca"],
            result["initial_ca"]
        )
    )

    counts = result["step_spikes"]
    base_counts = baseline["step_spikes"]

    for start, end in [
        (0, 1),     # 0-50 ms
        (1, 5),     # 50-250 ms
        (5, 10),    # 250-500 ms
        (10, 20)    # 500-1000 ms
    ]:
        print(
            f"{start*50:4d}-{end*50:4d} ms: "
            f"baseline={base_counts[start:end].sum():7d}, "
            f"stimulated={counts[start:end].sum():7d}"
        )

    print(
        "Total additional spikes:",
        counts.sum() - base_counts.sum()
    )

    if len(result["times"]):
        print(
            "First spike (ms):",
            result["times"].min()
        )

    for t_ms in [50, 250, 500, 1000]:

        ca = result["snapshots"][t_ms]["ca"]
        ca_base = baseline["snapshots"][t_ms]["ca"]

        dca = ca - ca_base

        print(
            f"{t_ms:4d} ms: "
            f"|dCa| stimE="
            f"{np.mean(np.abs(dca[:48])):.6f}, "
            f"otherE="
            f"{np.mean(np.abs(dca[48:480])):.6f}"
        )

print("\n=== FINISHED ===")
