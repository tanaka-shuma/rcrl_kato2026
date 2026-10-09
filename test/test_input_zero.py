import numpy as np

from snn4tanaka.option.option import get_opts
from snn4tanaka.simu import CulturedSNNCore


p = get_opts()

# ============================================================
# A: original advance()
# ============================================================
core_a = CulturedSNNCore(p)

core_a.warmup()
core_a.advance(1000.0)

ids_a = core_a.get_step_spike_ids()
times_a = core_a.get_step_spike_times()
ca_a = core_a.get_ca()
v_a = core_a.get_v()


# ============================================================
# B: advance_with_input(), but input = exactly zero
# ============================================================
core_b = CulturedSNNCore(p)

core_b.warmup()

zero_input = np.zeros(
    core_b.get_n_tot(),
    dtype=np.float64
)

core_b.advance_with_input(
    1000.0,
    zero_input
)

ids_b = core_b.get_step_spike_ids()
times_b = core_b.get_step_spike_times()
ca_b = core_b.get_ca()
v_b = core_b.get_v()


# ============================================================
# comparison
# ============================================================
print("=== elapsed ===")
print("A:", core_a.get_elapsed_ms())
print("B:", core_b.get_elapsed_ms())

print("\n=== spike count ===")
print("A:", len(ids_a))
print("B:", len(ids_b))

print("\n=== spike IDs ===")
print(
    "same:",
    np.array_equal(ids_a, ids_b)
)

print("\n=== spike times ===")
print(
    "allclose:",
    np.allclose(times_a, times_b)
)

if len(times_a) == len(times_b):
    if len(times_a) > 0:
        print(
            "max difference:",
            np.max(np.abs(times_a - times_b))
        )

print("\n=== Ca ===")
print(
    "allclose:",
    np.allclose(ca_a, ca_b)
)
print(
    "max difference:",
    np.max(np.abs(ca_a - ca_b))
)

print("\n=== V ===")
print(
    "allclose:",
    np.allclose(v_a, v_b)
)
print(
    "max difference:",
    np.max(np.abs(v_a - v_b))
)