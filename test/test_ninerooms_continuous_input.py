
import numpy as np

import common_configurator as common
from models.matrix_generator import generate_random_matrix
from models.snn_reservoir_snn4 import SpikingNeuralNetwork


c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

c.Nx = 576
c.Irate = 1.0 / 6.0
c.snn4_paper_condition = True
c.snn4_prerun_ms = 11000.0
c.seed = 1

SCALES = [0.0, 0.02, 0.03, 0.04, 0.05]
CHECK_STEPS = [1, 2, 5, 10, 20]


# ===================================================
# Generate a fixed sequence of NineRooms observations
# ===================================================

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

sensor_inputs = []

for k in range(20):

    # Fixed action sequence, no learning
    action = k % c.Ny

    u, reward, done, info = env.step(action)

    sensor_inputs.append(
        np.asarray(u, dtype=np.float64).copy()
    )

    if done:
        env.reset()

sensor_inputs = np.asarray(sensor_inputs)
raw_inputs = sensor_inputs @ Wi.T

print("=== Sensor input sequence ===")
print("shape:", raw_inputs.shape)
print(
    "number of changing observations:",
    np.count_nonzero(
        np.any(np.abs(np.diff(
            sensor_inputs, axis=0
        )) > 1e-12, axis=1)
    )
)


# ===================================================
# SNN simulation
# ===================================================

def run_condition(scale):

    c.input_const_snn4 = float(scale)

    reservoir = SpikingNeuralNetwork(c)
    reservoir.reset()

    initial_v = reservoir.neurons.v.copy()
    initial_ca = reservoir.r.copy()

    spike_E = []
    spike_I = []
    snapshots = {}

    for k in range(20):

        # Update sensor input every 50 ms
        reservoir.step(raw_inputs[k])

        ids = np.asarray(
            reservoir.core.get_step_spike_ids(),
            dtype=np.int64
        )

        spike_E.append(
            np.count_nonzero(ids < 480)
        )
        spike_I.append(
            np.count_nonzero(ids >= 480)
        )

        if k + 1 in CHECK_STEPS:

            t_ms = int(round(
                (k + 1) * c.ds
            ))

            snapshots[t_ms] = (
                reservoir.r[:480].copy()
            )

    return {
        "initial_v": initial_v,
        "initial_ca": initial_ca,
        "spike_E": np.asarray(spike_E),
        "spike_I": np.asarray(spike_I),
        "snapshots": snapshots
    }


results = {}

for scale in SCALES:
    print(f"Running scale={scale}", flush=True)
    results[scale] = run_condition(scale)


# ===================================================
# Analysis
# ===================================================

baseline = results[0.0]

print("\n=== Initial-state consistency ===")

for scale in SCALES[1:]:

    r = results[scale]

    same_v = np.array_equal(
        baseline["initial_v"], r["initial_v"]
    )

    same_ca = np.array_equal(
        baseline["initial_ca"], r["initial_ca"]
    )

    print(
        f"scale={scale}: "
        f"V={same_v}, Ca={same_ca}"
    )

    assert same_v and same_ca


print("\n=== Continuous input response ===")

for scale in SCALES:

    r = results[scale]

    E = r["spike_E"]
    I = r["spike_I"]

    currents_na = (
        np.maximum(raw_inputs * scale, 0.0)
        * 1000.0
    )

    print(f"\n--- scale={scale} ---")

    print(
        "Max current [nA]:",
        currents_na.max()
    )

    print(
        "Spikes E/I (0-50 ms):",
        E[0], "/", I[0]
    )

    print(
        "Spikes E/I (0-1000 ms):",
        E.sum(), "/", I.sum()
    )

    print(
        "Spikes per 50-ms step:",
        (E + I).tolist()
    )

    active_E_steps = np.flatnonzero(E > 0)

    print(
        "First step with E spikes:",
        int(active_E_steps[0] + 1)
        if active_E_steps.size else None
    )

    for t_ms in [50, 100, 250, 500, 1000]:

        ca = r["snapshots"][t_ms]
        ca0 = baseline["snapshots"][t_ms]

        dca = ca - ca0

        print(
            f"{t_ms:4d} ms | "
            f"mean |dCa|="
            f"{np.mean(np.abs(dca)):.6f} | "
            f"max |dCa|="
            f"{np.max(np.abs(dca)):.6f} | "
            f"||dCa||="
            f"{np.linalg.norm(dca):.6f}"
        )

print("\n=== FINISHED ===")
