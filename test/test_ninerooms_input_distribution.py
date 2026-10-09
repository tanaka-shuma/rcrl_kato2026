
import numpy as np

import common_configurator as common
from models.matrix_generator import generate_random_matrix

c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

c.Nx = 576
np.random.seed(int(c.seed))

# Same input-matrix generation rule as Agent.initialize()
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

inputs = []
sensor_samples = []

# Collect 20 observations without running the SNN
for k in range(20):
    action = k % 3
    u, reward, done, info = env.step(action)

    u = np.asarray(u, dtype=np.float64)
    sensor_samples.append(u.copy())

    inputs.append(Wi @ u)

    if done:
        env.reset()

inputs = np.asarray(inputs)

print("=== Input matrix ===")
print("Wi shape:", Wi.shape)
print("Input shape:", inputs.shape)
print("First sensor vector:", sensor_samples[0])

print("\n=== Raw Wi @ u ===")
print("min:", inputs.min())
print("max:", inputs.max())
print(
    "positive fraction:",
    np.mean(inputs > 0)
)

# Convert effective injected current from microampere to nA
SCALES = [0.0005, 0.002, 0.005, 0.01, 0.02]

for scale in SCALES:

    current_na = (
        np.maximum(inputs * scale, 0.0) * 1000.0
    )

    positive = current_na[current_na > 0]

    print(f"\n=== scale = {scale} ===")

    print(
        "positive current count per step (mean):",
        np.mean(np.count_nonzero(current_na > 0, axis=1))
    )

    print(
        "E positive per step (mean):",
        np.mean(
            np.count_nonzero(current_na[:, :480] > 0, axis=1)
        )
    )

    print(
        "I positive per step (mean):",
        np.mean(
            np.count_nonzero(current_na[:, 480:] > 0, axis=1)
        )
    )

    if positive.size:
        print(
            "positive current [nA], "
            "p50 / p90 / p99 / max:",
            np.percentile(positive, [50, 90, 99, 100])
        )

print("\n=== FINISHED ===")
