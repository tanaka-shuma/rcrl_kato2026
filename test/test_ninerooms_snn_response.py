
import numpy as np

import common_configurator as common
from models.matrix_generator import generate_random_matrix
from models.snn_reservoir_snn4 import SpikingNeuralNetwork


# ============================================================
# Configuration
# ============================================================

c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

c.Nx = 576
c.Irate = 1.0 / 6.0
c.snn4_paper_condition = True
c.snn4_prerun_ms = 11000.0
c.seed = 1

SCALES = [0.0, 0.02, 0.03, 0.05, 0.08]

# ============================================================
# Generate the same kind of input used by Agent
# ============================================================

np.random.seed(int(c.seed))

Wi = generate_random_matrix(
    c.Nx, c.Nu,
    c.alpha_i, c.beta_i,
    distribution="one",
    normalization="none"
)

env = common.generate_instance(
    c,
    module=c.env_module,
    class_=c.env_class
)

env.reset()
u, reward, done, info = env.step(0)

u = np.asarray(u, dtype=np.float64)
raw_input = Wi @ u

print("=== Input ===")
print("Sensor vector:", u)
print("Raw input min/max:",
      raw_input.min(), raw_input.max())

# ============================================================
# Run each scale from the same neural state
# ============================================================

def run_condition(scale):

    c.input_const_snn4 = scale

    reservoir = SpikingNeuralNetwork(c)
    reservoir.reset()

    initial_v = reservoir.neurons.v.copy()
    initial_ca = reservoir.r.copy()

    zero_input = np.zeros(c.Nx, dtype=np.float64)

    counts = []
    counts_E = []
    counts_I = []
    snapshots = {}

    for k in range(20):

        # A single 50-ms sensor-input pulse
        if k == 0:
            reservoir.step(raw_input)
        else:
            reservoir.step(zero_input)

        counts.append(
            reservoir.core.get_step_spike_count()
        )

        spike_ids = np.asarray(
            reservoir.core.get_step_spike_ids(),
            dtype=np.int64
        )

        counts_E.append(
            np.count_nonzero(spike_ids < 480)
        )

        counts_I.append(
            np.count_nonzero(spike_ids >= 480)
        )

        assert (
            counts_E[-1] + counts_I[-1]
            == counts[-1]
        )

        if k + 1 in [1, 5, 20]:
            t_ms = int(round((k + 1) * c.ds))
            snapshots[t_ms] = reservoir.r[:480].copy()

    return {
        "counts_E": np.asarray(counts_E),
        "counts_I": np.asarray(counts_I),
        "initial_v": initial_v,
        "initial_ca": initial_ca,
        "counts": np.asarray(counts),
        "snapshots": snapshots
    }


results = {}

for scale in SCALES:
    print(f"Running scale={scale}", flush=True)
    results[scale] = run_condition(scale)

baseline = results[0.0]

# ============================================================
# Comparison
# ============================================================

print("\n=== Initial-state consistency ===")

for scale in SCALES[1:]:

    result = results[scale]

    print(
        f"scale={scale}:",
        "V=",
        np.array_equal(
            baseline["initial_v"],
            result["initial_v"]
        ),
        "Ca=",
        np.array_equal(
            baseline["initial_ca"],
            result["initial_ca"]
        )
    )


print("\n=== Response comparison ===")

for scale in SCALES:

    result = results[scale]

    current_na = (
        np.maximum(raw_input * scale, 0.0)
        * 1000.0
    )

    print(f"\n--- scale={scale} ---")

    print(
        "Positive E / I:",
        np.count_nonzero(current_na[:480] > 0),
        "/",
        np.count_nonzero(current_na[480:] > 0)
    )

    print(
        "Maximum input current [nA]:",
        current_na.max()
    )

    print(
        "Maximum E current [nA]:",
        current_na[:480].max()
    )

    print(
        "Maximum I current [nA]:",
        current_na[480:].max()
    )

    print(
        "E neurons receiving >5 nA:",
        np.count_nonzero(current_na[:480] > 5.0)
    )

    print(
        "E neurons receiving >8 nA:",
        np.count_nonzero(current_na[:480] > 8.0)
    )

    print(
        "Spikes 0-50 ms:",
        result["counts"][0]
    )

    print(
        "Spikes 0-250 ms:",
        result["counts"][:5].sum()
    )

    print(
        "Spikes 0-1000 ms:",
        result["counts"].sum()
    )


    print(
        "E/I spikes (0-50 ms):",
        result["counts_E"][0],
        "/",
        result["counts_I"][0]
    )

    print(
        "E/I spikes (0-1000 ms):",
        result["counts_E"].sum(),
        "/",
        result["counts_I"].sum()
    )

    for t_ms in [50, 250, 1000]:

        ca = result["snapshots"][t_ms]
        ca0 = baseline["snapshots"][t_ms]

        print(
            f"{t_ms} ms max |dCa|:",
            f"{np.max(np.abs(ca - ca0)):.12e}"
        )


    for t_ms in [50, 250, 1000]:

        ca = result["snapshots"][t_ms]
        ca0 = baseline["snapshots"][t_ms]

        dca = ca - ca0

        print(
            f"{t_ms:4d} ms: "
            f"mean |dCa|="
            f"{np.mean(np.abs(dca)):.6f}, "
            f"||dCa||="
            f"{np.linalg.norm(dca):.6f}"
        )

print("\n=== FINISHED ===")
