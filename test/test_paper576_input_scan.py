
import numpy as np

import common_configurator as common
from models.snn_reservoir_snn4 import SpikingNeuralNetwork


CHECK_STEPS = [1, 2, 5, 10, 20]


def run_condition(input_current):

    c = common.load_config(
        "config.config_rl_ninerooms_rtdl_snn"
    )

    c.Nx = 576
    c.Irate = 1.0 / 6.0
    c.snn4_paper_condition = True
    # c.snn4_prerun_ms = 10000.0
    c.snn4_prerun_ms = 11000.0
    c.input_const_snn4 = 1.0
    c.seed = 1

    reservoir = SpikingNeuralNetwork(c)
    reservoir.reset()

    initial_v = reservoir.neurons.v.copy()
    initial_ca = reservoir.r.copy()

    external_input = np.zeros(c.Nx)
    external_input[:48] = input_current

    snapshots = {}
    cumulative_spikes = 0

    for k in range(20):

        reservoir.step(external_input)

        cumulative_spikes += (
            reservoir.core.get_step_spike_count()
        )

        if (k + 1) in CHECK_STEPS:

            t_ms = int(round((k + 1) * c.ds))

            snapshots[t_ms] = {
                "v": reservoir.neurons.v.copy(),
                "ca": reservoir.r.copy(),
                "spikes": cumulative_spikes
            }

    return {
        "initial_v": initial_v,
        "initial_ca": initial_ca,
        "snapshots": snapshots
    }



# Input current scan in the quiescent state
# 1 nA = 0.001 µA

# CURRENTS_NA = [0.0, 0.1, 0.5, 1.0, 2.0]
CURRENTS_NA = [0.0, 2.0, 5.0, 10.0, 20.0]

results = {}

for current_na in CURRENTS_NA:

    print(f"Running: {current_na} nA", flush=True)

    current_ua = current_na * 0.001

    results[current_na] = run_condition(
        current_ua
    )


baseline = results[0.0]

print("\n=== Initial-state consistency ===")

for current_na in CURRENTS_NA[1:]:

    result = results[current_na]

    same_v = np.array_equal(
        baseline["initial_v"],
        result["initial_v"]
    )

    same_ca = np.array_equal(
        baseline["initial_ca"],
        result["initial_ca"]
    )

    print(
        f"{current_na} nA:",
        f"V={same_v}, Ca={same_ca}"
    )


for t_ms in [50, 250, 1000]:

    print(f"\n=== {t_ms} ms ===")

    base = baseline["snapshots"][t_ms]

    print(
        "Baseline cumulative spikes:",
        base["spikes"]
    )

    for current_na in CURRENTS_NA[1:]:

        stim = results[current_na]["snapshots"][t_ms]

        dv = stim["v"] - base["v"]
        dc = stim["ca"] - base["ca"]

        delta_spikes = (
            stim["spikes"] - base["spikes"]
        )

        print(
            f"{current_na:4.1f} nA | "
            f"delta spikes={delta_spikes:7d} | "
            f"stim E |dV|="
            f"{np.mean(np.abs(dv[:48])):.6f} mV | "
            f"stim E |dCa|="
            f"{np.mean(np.abs(dc[:48])):.6f} | "
            f"other E |dCa|="
            f"{np.mean(np.abs(dc[48:480])):.6f}"
        )

print("\n=== FINISHED ===")
