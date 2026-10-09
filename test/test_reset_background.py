import numpy as np

from snn4tanaka.option.option import get_opts
from snn4tanaka.simu import CulturedSNNCore

p = get_opts()
core = CulturedSNNCore(p)

# warm-up後は、新しい1秒区間の先頭
core.warmup()

# 50 ms進める
core.advance(50.0)

bg_before = core.get_bg_g()
s_before = core.get_bg_noise_index()

print("=== before reset ===")
print("bg nonzero:", np.any(bg_before != 0.0))
print("bg max    :", np.max(np.abs(bg_before)))
print("noise index:", s_before)

# backgroundだけreset
core.reset_background()

bg_after = core.get_bg_g()
s_after = core.get_bg_noise_index()

print("\n=== after reset ===")
print("bg all zero:", np.all(bg_after == 0.0))
print("bg max     :", np.max(np.abs(bg_after)))
print("noise index:", s_after)

core.advance(50.0)

print("after new 50 ms")
print("bg nonzero:", np.any(core.get_bg_g() != 0.0))
print("noise index:", core.get_bg_noise_index())