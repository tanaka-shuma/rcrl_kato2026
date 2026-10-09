
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

c.snn4_prerun_ms = 10000.0

reservoir = SpikingNeuralNetwork(c)

N_EPISODES = 5
STEPS = 40  # 2 seconds at ds=50 ms

for episode in range(N_EPISODES):

    reservoir.reset()

    all_times = []

    for k in range(STEPS):
        reservoir.step(
            np.zeros(c.Nx, dtype=np.float64)
        )
        all_times.append(
            reservoir.core.get_step_spike_times()
            - reservoir.episode_time_origin_ms
        )

    times = np.concatenate(all_times)

    print(f"\n=== episode {episode} ===")
    print("total spikes:", len(times))

    if len(times) > 0:
        print("first spike (ms):", times.min())

    for start, end in [
        (0, 200),
        (200, 500),
        (500, 1000),
        (1000, 2000)
    ]:

        count = np.count_nonzero(
            (times >= start) & (times < end)
        )

        print(
            f"{start:4d}-{end:4d} ms: "
            f"{count:7d} spikes"
        )

print("\n=== FINISHED ===")
