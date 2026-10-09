
import numpy as np

import common_configurator as common
from models.snn_reservoir_snn4 import SpikingNeuralNetwork


def run_condition(input_current):

    c = common.load_config(
        "config.config_rl_ninerooms_rtdl_snn"
    )

    c.Nx = 576
    c.Irate = 1.0 / 6.0
    c.snn4_paper_condition = True
    c.snn4_prerun_ms = 10000.0

    # Reproduce the same initial network state
    c.seed = 1

    # Directly specify external current in µA
    c.input_const_snn4 = 1.0

    reservoir = SpikingNeuralNetwork(c)
    reservoir.reset()

    initial_ca = reservoir.r.copy()
    initial_v = reservoir.neurons.v.copy()

    # Stimulate 48 excitatory neurons (10% of E)
    external_input = np.zeros(c.Nx)
    external_input[:48] = input_current

    n_spikes = 0

    # 1 second = 20 × 50 ms
    for k in range(20):

        reservoir.step(external_input)

        n_spikes += (
            reservoir.core.get_step_spike_count()
        )

    return {
        "initial_ca": initial_ca,
        "initial_v": initial_v,
        "final_ca": reservoir.r.copy(),
        "final_v": reservoir.neurons.v.copy(),
        "spikes": n_spikes,
        "elapsed": reservoir.t
    }


# Control condition: no external input
baseline = run_condition(0.0)

# External current: 0.0001 µA = 0.1 nA
stimulated = run_condition(0.0001)


print("=== Initial-state consistency ===")

print(
    "Ca identical:",
    np.array_equal(
        baseline["initial_ca"],
        stimulated["initial_ca"]
    )
)

print(
    "V identical:",
    np.array_equal(
        baseline["initial_v"],
        stimulated["initial_v"]
    )
)


print("\n=== After 1000 ms ===")

print("Baseline spikes:", baseline["spikes"])
print("Stimulated spikes:", stimulated["spikes"])

print("Baseline elapsed:", baseline["elapsed"])
print("Stimulated elapsed:", stimulated["elapsed"])


delta_v = (
    stimulated["final_v"]
    - baseline["final_v"]
)

delta_ca = (
    stimulated["final_ca"]
    - baseline["final_ca"]
)

print("\n=== Stimulated neurons (0:48) ===")

print(
    "Mean |delta V|:",
    np.mean(np.abs(delta_v[:48]))
)

print(
    "Mean |delta Ca|:",
    np.mean(np.abs(delta_ca[:48]))
)


print("\n=== Other neurons (48:576) ===")

print(
    "Mean |delta V|:",
    np.mean(np.abs(delta_v[48:]))
)

print(
    "Mean |delta Ca|:",
    np.mean(np.abs(delta_ca[48:]))
)

print("\n=== FINISHED ===")
