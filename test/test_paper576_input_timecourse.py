
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


baseline = run_condition(0.0)
stimulated = run_condition(0.0001)

print("=== Initial-state consistency ===")

print(
    "V identical:",
    np.array_equal(
        baseline["initial_v"],
        stimulated["initial_v"]
    )
)

print(
    "Ca identical:",
    np.array_equal(
        baseline["initial_ca"],
        stimulated["initial_ca"]
    )
)


for t_ms in sorted(baseline["snapshots"]):

    b = baseline["snapshots"][t_ms]
    s = stimulated["snapshots"][t_ms]

    dv = s["v"] - b["v"]
    dc = s["ca"] - b["ca"]

    print(f"\n=== {t_ms} ms ===")

    print(
        "Spikes baseline / stimulated:",
        b["spikes"], "/", s["spikes"]
    )

    print(
        "Delta spikes:",
        s["spikes"] - b["spikes"]
    )

    print("Mean absolute delta V [mV]")
    print(
        "  Stimulated E (0:48):",
        np.mean(np.abs(dv[:48]))
    )
    print(
        "  Other E (48:480):",
        np.mean(np.abs(dv[48:480]))
    )
    print(
        "  I (480:576):",
        np.mean(np.abs(dv[480:576]))
    )

    print("Mean absolute delta Ca")
    print(
        "  Stimulated E (0:48):",
        np.mean(np.abs(dc[:48]))
    )
    print(
        "  Other E (48:480):",
        np.mean(np.abs(dc[48:480]))
    )
    print(
        "  I (480:576):",
        np.mean(np.abs(dc[480:576]))
    )

print("\n=== FINISHED ===")
