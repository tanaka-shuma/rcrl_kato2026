import numpy as np

import common_configurator as common
from models.snn_reservoir_snn4 import SpikingNeuralNetwork


# ============================================================
# Load the existing rcrl configuration
# ============================================================
c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

print("=== config ===")
print("Nx:", c.Nx)
print("dt:", c.dt)
print("ds:", c.ds)


# ============================================================
# 1. Core / wrapper construction
# ============================================================
reservoir = SpikingNeuralNetwork(c)

print("\n=== construction ===")
print("Ntot:", reservoir.core.get_n_tot())
print("r shape:", reservoir.r.shape)
print("NNe:", reservoir.NNe)
print("NNi:", reservoir.NNi)


# ============================================================
# 2. reset + warmup
# ============================================================
reservoir.reset()

print("\n=== after reset + warmup ===")
print("wrapper t:", reservoir.t)
print("wrapper n:", reservoir.n)
print("core elapsed:", reservoir.core.get_elapsed_ms())
print("r shape:", reservoir.r.shape)
print("Ca max:", reservoir.r.max())
print(
    "spike_times empty:",
    sum(len(x) for x in reservoir.spike_times) == 0
)

# consistency between wrapper state and Core
print(
    "r == neurons.c:",
    np.array_equal(
        reservoir.r,
        reservoir.neurons.c
    )
)


# ============================================================
# 3. One RL step = ds ms
# ============================================================

# Dummy Nx-dimensional input.
# External input is not injected into Core yet,
# but this verifies the Python-side interface.
test_input = np.linspace(
    0.0,
    1.0,
    int(c.Nx),
    dtype=np.float64
)

reservoir.step(test_input)

step_spikes = reservoir.core.get_step_spike_count()
recorded_spikes = sum(
    len(x) for x in reservoir.spike_times
)

print("\n=== after one RL step ===")
print("wrapper t:", reservoir.t)
print("wrapper n:", reservoir.n)
print("core elapsed:", reservoir.core.get_elapsed_ms())
print("r shape:", reservoir.r.shape)
print("Core step spikes:", step_spikes)
print("recorded spikes:", recorded_spikes)

print(
    "input stored correctly:",
    np.array_equal(
        reservoir.I_input,
        test_input
    )
)

print(
    "spike recording consistent:",
    step_spikes == recorded_spikes
)


# ============================================================
# 4. Test sum_spike()
# ============================================================
spikeE = 0
spikeI = 0

spikeE, spikeI = reservoir.sum_spike(
    spikeE=spikeE,
    spikeI=spikeI
)

print("\n=== sum_spike after first step ===")
print("E spikes:", spikeE)
print("I spikes:", spikeI)
print("E + I:", spikeE + spikeI)
print("Core step spikes:", step_spikes)

print(
    "sum_spike consistent:",
    spikeE + spikeI == step_spikes
)


# ============================================================
# 5. Continue for another 19 RL steps
#    total = 20 * ds = 1000 ms if ds=50 ms
# ============================================================
for k in range(19):

    reservoir.step(test_input)

    # main_rl.py also calls sum_spike() immediately
    # after every reservoir step.
    spikeE, spikeI = reservoir.sum_spike(
        spikeE=spikeE,
        spikeI=spikeI
    )


total_recorded_spikes = sum(
    len(x) for x in reservoir.spike_times
)

print("\n=== after 20 RL steps ===")
print("wrapper t:", reservoir.t)
print("wrapper n:", reservoir.n)
print("core elapsed:", reservoir.core.get_elapsed_ms())

print("E spikes:", spikeE)
print("I spikes:", spikeI)
print("E + I:", spikeE + spikeI)

print(
    "spike_times total:",
    total_recorded_spikes
)

print(
    "total spike count consistent:",
    spikeE + spikeI == total_recorded_spikes
)


# ============================================================
# 6. done()
# ============================================================
reservoir.done()

print("\n=== done ===")
print(
    "average spike rate:",
    reservoir.avg_spike_rate
)


# ============================================================
# 7. reset again
# ============================================================
reservoir.reset()

print("\n=== after second reset ===")
print("wrapper t:", reservoir.t)
print("wrapper n:", reservoir.n)
print("core elapsed:", reservoir.core.get_elapsed_ms())
print("r shape:", reservoir.r.shape)

print(
    "spike_times empty:",
    sum(len(x) for x in reservoir.spike_times) == 0
)

print(
    "Core step spike count:",
    reservoir.core.get_step_spike_count()
)

print(
    "I_input zero:",
    np.all(reservoir.I_input == 0.0)
)


# ============================================================
# 8. Make sure it can run again after reset
# ============================================================
reservoir.step(test_input)

print("\n=== run again after reset ===")
print("wrapper t:", reservoir.t)
print("wrapper n:", reservoir.n)
print("core elapsed:", reservoir.core.get_elapsed_ms())
print("r shape:", reservoir.r.shape)

print("\n=== TEST FINISHED ===")