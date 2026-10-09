import numpy as np

import common_configurator as common
from models.snn_reservoir_snn4 import SpikingNeuralNetwork


c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

print("=== config ===")
print("Nx:", c.Nx)
print("dt:", c.dt)
print("ds:", c.ds)
print("const:", c.const)


# ============================================================
# A: zero external input
# ============================================================
reservoir_a = SpikingNeuralNetwork(c)
reservoir_a.reset()

zero_input = np.zeros(
    c.Nx,
    dtype=np.float64
)

reservoir_a.step(zero_input)

v_a = reservoir_a.neurons.v.copy()
ca_a = reservoir_a.r.copy()
spikes_a = reservoir_a.core.get_step_spike_count()


# ============================================================
# B: non-zero external input
#
# Choose the raw value so that, after multiplying by c.const,
# the injected current is exactly 0.0001 uA = 0.1 nA.
# ============================================================
reservoir_b = SpikingNeuralNetwork(c)
reservoir_b.reset()

target_current = 1.0e-4   # uA = 0.1 nA

test_input = np.zeros(
    c.Nx,
    dtype=np.float64
)

# Stimulate only the first 10 excitatory neurons.
test_input[:10] = target_current / c.const

reservoir_b.step(test_input)

v_b = reservoir_b.neurons.v.copy()
ca_b = reservoir_b.r.copy()
spikes_b = reservoir_b.core.get_step_spike_count()


# ============================================================
# Results
# ============================================================
print("\n=== external input ===")
print(
    "I_input min/max:",
    reservoir_b.I_input.min(),
    reservoir_b.I_input.max()
)

print(
    "first 10 I_input:",
    reservoir_b.I_input[:10]
)

print(
    "rest zero:",
    np.all(reservoir_b.I_input[10:] == 0.0)
)


print("\n=== zero vs non-zero input ===")
print("zero-input spikes:", spikes_a)
print("input spikes     :", spikes_b)

print(
    "V identical:",
    np.array_equal(v_a, v_b)
)

print(
    "V allclose:",
    np.allclose(v_a, v_b)
)

print(
    "max V difference:",
    np.max(np.abs(v_a - v_b))
)

print(
    "Ca identical:",
    np.array_equal(ca_a, ca_b)
)

print(
    "max Ca difference:",
    np.max(np.abs(ca_a - ca_b))
)


print("\n=== stimulated neurons ===")
for i in range(10):
    print(
        f"{i:3d}: "
        f"V_zero={v_a[i]:10.6f}, "
        f"V_input={v_b[i]:10.6f}, "
        f"dV={v_b[i]-v_a[i]:10.6f}"
    )


# ============================================================
# Negative-input clipping test
# ============================================================
reservoir_c = SpikingNeuralNetwork(c)
reservoir_c.reset()

negative_input = -np.ones(
    c.Nx,
    dtype=np.float64
)

reservoir_c.step(negative_input)

print("\n=== negative input clipping ===")
print(
    "all clipped to zero:",
    np.all(reservoir_c.I_input == 0.0)
)

print("\n=== TEST FINISHED ===")