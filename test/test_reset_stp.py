import numpy as np

from snn4tanaka.option.option import get_opts
from snn4tanaka.simu import CulturedSNNCore

p = get_opts()
core = CulturedSNNCore(p)

core.warmup()
core.advance(1000.0)

x_before = core.get_stp_x()
u_before = core.get_stp_u()
U = core.get_stp_U()
Dt_before = core.get_stp_Dt()

valid = U > 0

print("=== before reset ===")
print("valid connections:", np.sum(valid))
print("x changed:", np.any(x_before[valid] != 1.0))
print("u changed:", np.any(u_before[valid] != 0.0))
print("Dt nonzero:", np.any(Dt_before[valid] != 0.0))
print("Dt max:", Dt_before[valid].max())

core.reset_stp()

x_after = core.get_stp_x()
u_after = core.get_stp_u()
Dt_after = core.get_stp_Dt()

print("\n=== after reset ===")
print("valid x all 1:", np.all(x_after[valid] == 1.0))
print("valid u all 0:", np.all(u_after[valid] == 0.0))
print("Dt all zero:", np.all(Dt_after == 0.0))

print(
    "invalid x all -1:",
    np.all(x_after[~valid] == -1.0)
)
print(
    "invalid u all -1:",
    np.all(u_after[~valid] == -1.0)
)