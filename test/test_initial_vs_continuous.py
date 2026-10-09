import numpy as np

import common_configurator as common
from models.snn_reservoir_snn4 import SpikingNeuralNetwork

c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

c.Nx = 576
c.Irate = 1.0 / 6.0
c.snn4_paper_condition = True
c.input_const_snn4 = 0.0

reservoir = SpikingNeuralNetwork(c)

N_EPISODES = 3
DURATION_S = 12

steps_per_second = int(round(1000.0 / c.ds))

zero_input = np.zeros(
    c.Nx,
    dtype=np.float64
)

for episode in range(N_EPISODES):

    reservoir.reset()

    spikes_per_second = []

    for sec in range(DURATION_S):

        count = 0

        for k in range(steps_per_second):

            reservoir.step(zero_input)

            count += (
                reservoir.core.get_step_spike_count()
            )

        spikes_per_second.append(count)

    counts = np.asarray(spikes_per_second)

    print(f"\n=== episode {episode} ===")

    for sec, count in enumerate(counts):
        print(
            f"{sec:2d}-{sec+1:2d} s: "
            f"{count:7d} spikes"
        )

    print("\nSummary")
    print("initial 0-1 s:", counts[0])
    print("initial 1-2 s:", counts[1])

    print(
        "mean spikes/s (3-12 s):",
        np.mean(counts[3:])
    )

    print(
        "late 10-12 s:",
        np.sum(counts[10:12])
    )

print("\n=== FINISHED ===")