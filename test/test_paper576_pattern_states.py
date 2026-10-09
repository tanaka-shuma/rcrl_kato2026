
import numpy as np

import common_configurator as common
from models.snn_reservoir_snn4 import SpikingNeuralNetwork


PRERUN_TIMES = [10000, 10500, 11000, 11500, 12000]


def run_condition(prerun_ms, pattern):

    c = common.load_config(
        "config.config_rl_ninerooms_rtdl_snn"
    )

    c.Nx = 576
    c.Irate = 1.0 / 6.0
    c.snn4_paper_condition = True
    c.snn4_prerun_ms = float(prerun_ms)
    c.input_const_snn4 = 1.0
    c.seed = 1

    reservoir = SpikingNeuralNetwork(c)
    reservoir.reset()

    initial_v = reservoir.neurons.v.copy()
    initial_ca = reservoir.r.copy()

    pulse = np.zeros(c.Nx)

    if pattern == "A":
        pulse[:48] = 0.008
    elif pattern == "B":
        pulse[48:96] = 0.008

    zero = np.zeros(c.Nx)
    snapshots = {}
    spikes = []

    for k in range(5):

        reservoir.step(pulse if k == 0 else zero)

        spikes.append(
            reservoir.core.get_step_spike_count()
        )

        if k + 1 in [1, 5]:
            snapshots[(k + 1) * 50] = (
                reservoir.r[:480].copy()
            )

    return {
        "initial_v": initial_v,
        "initial_ca": initial_ca,
        "snapshots": snapshots,
        "spikes": np.asarray(spikes)
    }


print("=== Different spontaneous states ===")

for prerun_ms in PRERUN_TIMES:

    results = {
        pattern: run_condition(prerun_ms, pattern)
        for pattern in ["none", "A", "B"]
    }

    base = results["none"]

    for pattern in ["A", "B"]:
        assert np.array_equal(
            base["initial_v"],
            results[pattern]["initial_v"]
        )
        assert np.array_equal(
            base["initial_ca"],
            results[pattern]["initial_ca"]
        )

    print(f"\n=== Prerun {prerun_ms / 1000:.1f} s ===")

    print(
        "Spikes (0-50 ms), none/A/B:",
        *[
            results[p]["spikes"][0]
            for p in ["none", "A", "B"]
        ]
    )

    for t_ms in [50, 250]:

        ca0 = base["snapshots"][t_ms]
        caA = results["A"]["snapshots"][t_ms]
        caB = results["B"]["snapshots"][t_ms]

        dA = caA - ca0
        dB = caB - ca0

        zA = dA - dA.mean()
        zB = dB - dB.mean()

        denom = np.linalg.norm(zA) * np.linalg.norm(zB)

        spatial_cos = (
            np.dot(zA, zB) / denom
            if denom > 1e-12 else np.nan
        )

        raw_difference = (
            np.linalg.norm(caA - caB)
            / max(
                np.linalg.norm(caA),
                np.linalg.norm(caB),
                1e-12
            )
        )

        response_difference = np.linalg.norm(
            caA - caB
        )

        print(
            f"{t_ms:3d} ms | "
            f"spatial cos={spatial_cos:8.4f} | "
            f"raw diff={raw_difference:9.5f} | "
            f"||CaA-CaB||={response_difference:9.5f}"
        )

print("\n=== FINISHED ===")
