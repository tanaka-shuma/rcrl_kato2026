import numpy as np

from snn4tanaka.option.option import get_opts
from snn4tanaka.simu import CulturedSNNCore

p = get_opts()
core = CulturedSNNCore(p)

core.warmup()
core.advance(1000.0)

fast_before = core.get_fast_current()
slow_before = core.get_slow_current()
bg_before = core.get_bg_current()

print("=== before reset ===")

print("fast nonzero:", np.any(fast_before != 0.0))
print("slow nonzero:", np.any(slow_before != 0.0))
print("bg nonzero  :", np.any(bg_before != 0.0))

print("fast max:", np.max(np.abs(fast_before)))
print("slow max:", np.max(np.abs(slow_before)))
print("bg max  :", np.max(np.abs(bg_before)))

core.reset_currents()

fast_after = core.get_fast_current()
slow_after = core.get_slow_current()
bg_after = core.get_bg_current()

print("\n=== after reset ===")

print("fast all zero:", np.all(fast_after == 0.0))
print("slow all zero:", np.all(slow_after == 0.0))
print("bg all zero  :", np.all(bg_after == 0.0))

print("fast max:", np.max(np.abs(fast_after)))
print("slow max:", np.max(np.abs(slow_after)))
print("bg max  :", np.max(np.abs(bg_after)))