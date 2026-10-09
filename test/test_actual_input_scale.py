import numpy as np

import common_configurator as common
from models.agent_rtdl import Agent


# ============================================================
# Load config
# ============================================================
c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

# Use the new reservoir only for this test
c.rc_module = "models.snn_reservoir_snn4"
c.rc_class = "SpikingNeuralNetwork"

np.random.seed(int(c.seed))


# ============================================================
# Construct in the same order as main_rl.py
# ============================================================
agent = Agent(c)

env = common.generate_instance(
    c,
    module=c.env_module,
    class_=c.env_class
)

agent.initialize()

# Same order as the beginning of an episode
agent.reset(0)
state, reward, done, info = env.reset()


# ============================================================
# Measure Wi @ u without actually giving it to the SNN
# ============================================================
raw_values = []
current_values = []

n_samples = 100

for k in range(n_samples):

    # Cycle through the three actions only to obtain
    # different environmental states.
    action = k % c.Ny

    state, reward, done, info = env.step(action)

    # This is exactly what Agent.step() computes
    raw = agent.Wi @ state

    # This is what snn_reservoir_snn4.step() will inject
    current = raw * c.input_const_snn4
    # current = raw * c.const
    current = np.where(
        current < 0.0,
        0.0,
        current
    )

    raw_values.append(raw.copy())
    current_values.append(current.copy())

    if done:
        state, reward, done, info = env.reset()


raw_values = np.concatenate(raw_values)
current_values = np.concatenate(current_values)


# ============================================================
# Statistics
# ============================================================
print("=== config ===")
print("Nx:", c.Nx)
print("Nu:", c.Nu)
print("const:", c.const)
print("input_const_snn4:", c.input_const_snn4)

print("\n=== Wi @ u ===")
print("min   :", raw_values.min())
print("max   :", raw_values.max())
print("mean  :", raw_values.mean())
print("median:", np.median(raw_values))

print(
    "95 percentile:",
    np.percentile(raw_values, 95)
)

print(
    "99 percentile:",
    np.percentile(raw_values, 99)
)


print("\n=== injected current after scaling/clipping ===")
print("unit: uA")
print("min   :", current_values.min())
print("max   :", current_values.max())
print("mean  :", current_values.mean())
print("median:", np.median(current_values))

print(
    "95 percentile:",
    np.percentile(current_values, 95)
)

print(
    "99 percentile:",
    np.percentile(current_values, 99)
)

print(
    "nonzero fraction:",
    np.mean(current_values > 0.0)
)


print("\n=== converted to nA ===")

current_nA = current_values * 1000.0

print("max   :", current_nA.max(), "nA")
print("mean  :", current_nA.mean(), "nA")

print(
    "95 percentile:",
    np.percentile(current_nA, 95),
    "nA"
)

print(
    "99 percentile:",
    np.percentile(current_nA, 99),
    "nA"
)


env.close()