import numpy as np

import common_configurator as common
from models.snn_reservoir_snn4 import SpikingNeuralNetwork

c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

# Paper model
c.Nx = 576
c.Irate = 1.0 / 6.0
c.snn4_paper_condition = True

# No external input for the first verification
c.input_const_snn4 = 0.0

reservoir = SpikingNeuralNetwork(c)
reservoir.reset()

print("=== Initial ===")
print("N:", reservoir.Ntot)
print("E:", reservoir.NNe)
print("I:", reservoir.NNi)
print("r shape:", reservoir.r.shape)

# ============================================================
# Spontaneous activity for 10 seconds
# ============================================================

n_steps = 200  # 200 * 50 ms = 10 s

spikeE = 0
spikeI = 0

spikes_per_step = []
ca_max_per_step = []

for k in range(n_steps):

    reservoir.step(
        np.zeros(c.Nx, dtype=np.float64)
    )

    ids = reservoir.core.get_step_spike_ids()

    spikes_per_step.append(len(ids))
    ca_max_per_step.append(
        float(reservoir.r.max())
    )

    spikeE, spikeI = reservoir.sum_spike(
        spikeE, spikeI
    )


# ============================================================
# Summary for each 1-second interval
# ============================================================

print("\n=== Spontaneous activity ===")

for sec in range(10):

    start = sec * 20
    end = (sec + 1) * 20

    count = sum(
        spikes_per_step[start:end]
    )

    rate = count / c.Nx  # 1-second interval

    max_ca = max(
        ca_max_per_step[start:end]
    )

    print(
        f"{sec:2d}-{sec+1:2d} s: "
        f"spikes={count:7d}, "
        f"rate={rate:8.3f} sp/s/neuron, "
        f"max Ca={max_ca:8.3f}"
    )


print("\n=== Total ===")
print("elapsed:", reservoir.t)
print("total spikes:", spikeE + spikeI)

recorded = sum(
    len(x) for x in reservoir.spike_times
)

print("recorded:", recorded)
print("spike counts consistent:",
      recorded == spikeE + spikeI)

# print("\n=== After 1000 ms ===")
# print("elapsed:", reservoir.t)
# print("spikes:", spikeE + spikeI)
# print(
#     "recorded:",
#     sum(len(x) for x in reservoir.spike_times)
# )
# print("Ca max:", reservoir.r.max())
# print("V range:",
#       reservoir.neurons.v.min(),
#       reservoir.neurons.v.max())