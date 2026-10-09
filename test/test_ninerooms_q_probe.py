
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

SCALES = [0.0, 0.02]
CHECK_STEPS = [1, 2, 5, 10, 20]


# ===================================================
# Generate a fixed sequence of NineRooms observations
# ===================================================

np.random.seed(int(c.seed))

# Wi = generate_random_matrix(
#     c.Nx, c.Nu,
#     c.alpha_i, c.beta_i,
#     distribution="one",
#     normalization="none"
# )

weights = np.load("rtdl_probe_weights.npz")
Wi = weights["Wi"]
Wo = weights["Wo"]

assert Wi.shape == (576, 8)
assert Wo.shape == (3, 576)

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

            snapshots[t_ms] = reservoir.r.copy()

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

    
    print("\n=== Fixed-readout Q comparison ===")

    for t_ms in [50, 100, 250, 500, 1000]:

        ca_none = results[0.0]["snapshots"][t_ms]
        ca_input = results[0.02]["snapshots"][t_ms]

        # Identical learned output weights for both conditions
        q_none = Wo @ ca_none
        q_input = Wo @ ca_input

        action_none = int(np.argmax(q_none))
        action_input = int(np.argmax(q_input))

        print(f"\n--- {t_ms} ms ---")

        print("Q without input:", np.round(q_none, 5))
        print("Q with input:   ", np.round(q_input, 5))

        print(
            "||dQ||:",
            np.linalg.norm(q_input - q_none)
        )

        print(
            "Selected action none/input:",
            action_none, "/", action_input
        )

        print(
            "Action changed:",
            action_none != action_input
        )


    print("\n=== Q contribution analysis ===")

    # Excitatory neurons (0:480)
    W_E = Wo[:, :480]

    for t_ms in [500, 1000]:

        ca_none = results[0.0]["snapshots"][t_ms]
        ca_input = results[0.02]["snapshots"][t_ms]

        # Input-induced Ca difference
        dCa = ca_input - ca_none
        dCa_E = dCa[:480]

        # Common and spatial parts
        mean_dCa = np.mean(dCa_E)

        dCa_common = np.full(480, mean_dCa)
        dCa_spatial = dCa_E - mean_dCa

        # Contributions to Q-value difference
        dQ_common = W_E @ dCa_common
        dQ_spatial = W_E @ dCa_spatial

        # Include inhibitory components separately
        dQ_I = Wo[:, 480:] @ dCa[480:]

        dQ_total = Wo @ dCa

        # Q-value difference between actions
        q_none = Wo @ ca_none
        q_input = Wo @ ca_input

        print(f"\n--- {t_ms} ms ---")

        print("Mean dCa (E):", mean_dCa)

        print("dQ total:  ", np.round(dQ_total, 6))
        print("dQ common: ", np.round(dQ_common, 6))
        print("dQ spatial:", np.round(dQ_spatial, 6))
        print("dQ from I: ", np.round(dQ_I, 6))

        print(
            "Reconstruction error:",
            np.linalg.norm(
                dQ_total -
                dQ_common -
                dQ_spatial -
                dQ_I
            )
        )

        print(
            "Common contribution norm:",
            np.linalg.norm(dQ_common)
        )

        print(
            "Spatial contribution norm:",
            np.linalg.norm(dQ_spatial)
        )

        print(
            "Action 0-1 gap (none/input):",
            q_none[0] - q_none[1],
            "/",
            q_input[0] - q_input[1]
        )

    
    print("\n=== Output weight structure ===")

    # Excitatory neurons only
    W = Wo[:, :480]

    for a in range(3):

        w = W[a]

        mean_w = np.mean(w)
        std_w = np.std(w)

        w_common = np.full_like(w, mean_w)
        w_spatial = w - w_common

        common_norm = np.linalg.norm(w_common)
        spatial_norm = np.linalg.norm(w_spatial)
        total_norm = np.linalg.norm(w)

        print(f"\n--- Action {a} ---")

        print("Mean weight:", mean_w)
        print("Std weight:", std_w)

        print("Total norm:", total_norm)
        print("Common norm:", common_norm)
        print("Spatial norm:", spatial_norm)

        print(
            "Spatial / total:",
            spatial_norm / max(total_norm, 1e-15)
        )


    # Singular values of the learned readout matrix
    s = np.linalg.svd(W, compute_uv=False)
    energy = s**2 / np.sum(s**2)

    print("\n=== Readout singular-value energy ===")
    print(energy)


    # How large is the spatial component of dCa itself?
    print("\n=== Ca response structure ===")

    for t_ms in [500, 1000]:

        ca0 = results[0.0]["snapshots"][t_ms][:480]
        ca1 = results[0.02]["snapshots"][t_ms][:480]

        dca = ca1 - ca0
        dca_spatial = dca - np.mean(dca)

        print(f"\n--- {t_ms} ms ---")

        print("Mean dCa:", np.mean(dca))
        print("Std dCa:", np.std(dca))

        print(
            "Spatial / total norm:",
            np.linalg.norm(dca_spatial)
            / max(np.linalg.norm(dca), 1e-15)
        )




print("\n=== FINISHED ===")
