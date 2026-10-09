import numpy as np

from snn4tanaka.option.option import get_opts
from snn4tanaka.simu import CulturedSNNCore

p = get_opts()
core = CulturedSNNCore(p)

# initial state
ca0 = core.get_ca()
v0 = core.get_v()

# change neuronal state
core.warmup()
core.advance(1000.0)

ca_before = core.get_ca()
v_before = core.get_v()

print("=== before reset ===")
print("Ca changed:", np.any(ca_before != ca0))
print("V changed :", np.any(v_before != v0))
print("Ca max    :", ca_before.max())

# reset only neurons
core.reset_neurons()

ca_after = core.get_ca()
v_after = core.get_v()
sp_after = core.get_last_spikes()

print("\n=== after reset ===")
print("Ca all zero :", np.all(ca_after == 0.0))
print("V restored  :", np.array_equal(v_after, v0))
print("spikes zero :", np.all(sp_after == 0))

print("Ca max      :", ca_after.max())
print("V min/max   :", v_after.min(), v_after.max())