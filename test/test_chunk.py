import numpy as np

from snn4tanaka.option.option import get_opts
from snn4tanaka.simu import CulturedSNNCore


# ============================
# Parameters
# ============================
p = get_opts()


# ============================
# A: 1000 ms at once
# ============================
core_a = CulturedSNNCore(p)
core_a.warmup()

core_a.advance(1000.0)

ids_a = core_a.get_step_spike_ids()
times_a = core_a.get_step_spike_times()

ca_a = core_a.get_ca()
v_a = core_a.get_v()


# ============================
# B: 50 ms x 20
# ============================
core_b = CulturedSNNCore(p)
core_b.warmup()

ids_b_list = []
times_b_list = []

for k in range(20):

    core_b.advance(50.0)

    ids_b_list.append(
        core_b.get_step_spike_ids()
    )

    times_b_list.append(
        core_b.get_step_spike_times()
    )


ids_b = np.concatenate(ids_b_list)
times_b = np.concatenate(times_b_list)

ca_b = core_b.get_ca()
v_b = core_b.get_v()


# ============================
# Compare
# ============================
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
    "same:",
    np.array_equal(times_a, times_b)
)

if len(times_a) == len(times_b):
    print(
        "max difference:",
        np.max(np.abs(times_a - times_b))
        if len(times_a) > 0 else 0.0
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


print("\n=== first spikes ===")
print("A ids  :", ids_a[:20])
print("B ids  :", ids_b[:20])
print("A times:", times_a[:20])
print("B times:", times_b[:20])

print(
    "allclose:",
    np.allclose(times_a, times_b)
)