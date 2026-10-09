import numpy as np

from snn4tanaka.option.option import get_opts
from snn4tanaka.simu import CulturedSNNCore

p = get_opts()
core = CulturedSNNCore(p)

core.warmup()
core.advance(500.0)

n_before = core.get_network_n_firings()
firings_before = core.get_network_firings()

print("=== before reset ===")
print("N_firings:", n_before)
print("firings shape:", firings_before.shape)
print("first entries:")
print(firings_before[:, :10])

core.reset_network_history()

n_after = core.get_network_n_firings()
firings_after = core.get_network_firings()

print("\n=== after reset ===")
print("N_firings:", n_after)
print("firings shape:", firings_after.shape)
print("firings:")
print(firings_after)

print(
    "dummy time correct:",
    firings_after[0, 0] < 0
)

print(
    "dummy neuron correct:",
    firings_after[1, 0] == 0
)