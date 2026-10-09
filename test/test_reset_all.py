import numpy as np

from snn4tanaka.option.option import get_opts
from snn4tanaka.simu import CulturedSNNCore

p = get_opts()
core = CulturedSNNCore(p)

# initial state
v0 = core.get_v()

# change every dynamic state
core.warmup()
core.advance(500.0)

print("=== before reset ===")
print("elapsed:", core.get_elapsed_ms())
print("Ca max:", core.get_ca().max())
print("V changed:", np.any(core.get_v() != v0))
print("fast g nonzero:", np.any(core.get_fast_g() != 0))
print("bg g nonzero:", np.any(core.get_bg_g() != 0))
print("bg index:", core.get_bg_noise_index())
print("network firings:", core.get_network_n_firings())
print("step spikes:", core.get_step_spike_count())

# full reset
core.reset()

print("\n=== after reset ===")
print("elapsed:", core.get_elapsed_ms())

print(
    "Ca all zero:",
    np.all(core.get_ca() == 0.0)
)

print(
    "V restored:",
    np.array_equal(core.get_v(), v0)
)

print(
    "last spikes zero:",
    np.all(core.get_last_spikes() == 0)
)

print(
    "fast g zero:",
    np.all(core.get_fast_g() == 0.0)
)

print(
    "slow g zero:",
    np.all(core.get_slow_g() == 0.0)
)

print(
    "bg g zero:",
    np.all(core.get_bg_g() == 0.0)
)

print(
    "fast current zero:",
    np.all(core.get_fast_current() == 0.0)
)

print(
    "slow current zero:",
    np.all(core.get_slow_current() == 0.0)
)

print(
    "bg current zero:",
    np.all(core.get_bg_current() == 0.0)
)

print(
    "bg index:",
    core.get_bg_noise_index()
)

print(
    "network firings:",
    core.get_network_n_firings()
)

print(
    "step spikes:",
    core.get_step_spike_count()
)

print("\n=== run again after reset ===")

core.warmup()
core.advance(50.0)

print("elapsed:", core.get_elapsed_ms())
print("Ca max:", core.get_ca().max())
print("V changed:", np.any(core.get_v() != v0))
print("bg index:", core.get_bg_noise_index())
print("step spikes:", core.get_step_spike_count())