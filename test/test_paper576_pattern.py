
import numpy as np

import common_configurator as common
from models.snn_reservoir_snn4 import SpikingNeuralNetwork


def run_condition(pattern):

    c = common.load_config(
        "config.config_rl_ninerooms_rtdl_snn"
    )

    c.Nx = 576
    c.Irate = 1.0 / 6.0
    c.snn4_paper_condition = True
    c.snn4_prerun_ms = 10000.0
    c.input_const_snn4 = 1.0
    c.seed = 1

    reservoir = SpikingNeuralNetwork(c)
    reservoir.reset()

    initial_ca = reservoir.r.copy()
    initial_v = reservoir.neurons.v.copy()

    pulse = np.zeros(c.Nx)

    if pattern == "A":
        pulse[0:48] = 0.008    # 8 nA
    elif pattern == "B":
        pulse[48:96] = 0.008   # 8 nA
    elif pattern != "none":
        raise ValueError(pattern)

    zero = np.zeros(c.Nx)

    snapshots = {}
    spikes = []

    for k in range(20):

        if k == 0:
            reservoir.step(pulse)
        else:
            reservoir.step(zero)

        spikes.append(
            reservoir.core.get_step_spike_count()
        )

        if k + 1 in [1, 5, 10, 20]:
            t_ms = int(round((k + 1) * c.ds))
            snapshots[t_ms] = reservoir.r.copy()

    return {
        "initial_ca": initial_ca,
        "initial_v": initial_v,
        "snapshots": snapshots,
        "spikes": np.asarray(spikes)
    }


results = {}

for pattern in ["none", "A", "B"]:
    print("Running:", pattern, flush=True)
    results[pattern] = run_condition(pattern)


baseline = results["none"]

print("\n=== Initial-state consistency ===")

for pattern in ["A", "B"]:
    print(
        pattern,
        "V:",
        np.array_equal(
            baseline["initial_v"],
            results[pattern]["initial_v"]
        ),
        "Ca:",
        np.array_equal(
            baseline["initial_ca"],
            results[pattern]["initial_ca"]
        )
    )


print("\n=== Spike counts ===")

for pattern, result in results.items():
    print(
        pattern,
        "0-50 ms:",
        result["spikes"][0],
        "0-1000 ms:",
        result["spikes"].sum()
    )


print("\n=== Ca pattern comparison ===")

for t_ms in [50, 250, 500, 1000]:

    # Compare excitatory neurons only
    ca0 = baseline["snapshots"][t_ms][:480]
    caA = results["A"]["snapshots"][t_ms][:480]
    caB = results["B"]["snapshots"][t_ms][:480]

    # Input-induced change relative to baseline
    dA = caA - ca0
    dB = caB - ca0

    # Cosine similarity
    denom = np.linalg.norm(dA) * np.linalg.norm(dB)

    if denom > 1e-12:
        similarity = np.dot(dA, dB) / denom
    else:
        similarity = np.nan

    # Difference relative to response magnitude
    scale = max(
        np.linalg.norm(dA),
        np.linalg.norm(dB)
    )

    relative_difference = (
        np.linalg.norm(dA - dB) / scale
        if scale > 1e-12 else np.nan
    )

    print(f"\n--- {t_ms} ms ---")

    print("Cosine similarity:", similarity)
    print("Relative difference:", relative_difference)

    print(
        "A stimulation, mean dCa A/B:",
        np.mean(dA[0:48]),
        np.mean(dA[48:96])
    )

    print(
        "B stimulation, mean dCa A/B:",
        np.mean(dB[0:48]),
        np.mean(dB[48:96])
    )

    
    print("\n=== Spatial Ca pattern analysis ===")

    for t_ms in [50, 250, 500, 1000]:

        ca0 = baseline["snapshots"][t_ms][:480]
        caA = results["A"]["snapshots"][t_ms][:480]
        caB = results["B"]["snapshots"][t_ms][:480]

        dA = caA - ca0
        dB = caB - ca0

        # Remove population-wide mean response
        zA = dA - np.mean(dA)
        zB = dB - np.mean(dB)

        normA = np.linalg.norm(dA)
        normB = np.linalg.norm(dB)

        norm_zA = np.linalg.norm(zA)
        norm_zB = np.linalg.norm(zB)

        # Spatially centered cosine similarity
        if norm_zA > 1e-12 and norm_zB > 1e-12:
            spatial_cos = np.dot(zA, zB) / (
                norm_zA * norm_zB
            )
        else:
            spatial_cos = np.nan

        # Fraction of squared response norm in common mode
        common_fraction_A = (
            480 * np.mean(dA)**2 / normA**2
            if normA > 1e-12 else np.nan
        )

        common_fraction_B = (
            480 * np.mean(dB)**2 / normB**2
            if normB > 1e-12 else np.nan
        )

        # Difference between spatially centered patterns
        spatial_difference = (
            np.linalg.norm(zA - zB)
            / max(norm_zA, norm_zB)
            if max(norm_zA, norm_zB) > 1e-12
            else np.nan
        )

        # Difference of raw Ca vectors available to readout
        absolute_difference = (
            np.linalg.norm(caA - caB)
            / max(np.linalg.norm(caA),
                np.linalg.norm(caB))
        )

        print(f"\n--- {t_ms} ms ---")

        print("Common-mode energy fraction:")
        print("  A:", common_fraction_A)
        print("  B:", common_fraction_B)

        print("Spatial cosine similarity:", spatial_cos)
        print("Spatial relative difference:", spatial_difference)

        print("Raw Ca relative difference:", absolute_difference)



print("\n=== FINISHED ===")
