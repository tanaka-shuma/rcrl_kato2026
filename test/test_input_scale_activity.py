import numpy as np

import common_configurator as common
from models.agent_rtdl import Agent
from models.snn_reservoir_snn4 import SpikingNeuralNetwork


c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

c.rc_module = "models.snn_reservoir_snn4"
c.rc_class = "SpikingNeuralNetwork"

np.random.seed(int(c.seed))


# ============================================================
# Generate ONE fixed sequence of actual Wi @ u inputs
# ============================================================
agent = Agent(c)

env = common.generate_instance(
    c,
    module=c.env_module,
    class_=c.env_class
)

agent.initialize()

state, reward, done, info = env.reset()

raw_inputs = []

n_steps = 100       # 100 * 50 ms = 5 s

for k in range(n_steps):

    action = k % c.Ny

    state, reward, done, info = env.step(action)

    raw = agent.Wi @ state
    raw_inputs.append(raw.copy())

    if done:
        state, reward, done, info = env.reset()

env.close()


# ============================================================
# Compare input scales
# ============================================================
scales = [
    0.0,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
]



print("=== input-scale comparison ===")

for scale in scales:

    c.input_const_snn4 = scale

    reservoir = SpikingNeuralNetwork(c)
    reservoir.reset()

    total_spikes = 0
    active_neurons = set()

    max_ca = 0.0
    mean_ca_sum = 0.0
    
    mean_v_sum = 0.0
    max_v = -np.inf

    for raw in raw_inputs:

        reservoir.step(raw)

        v = reservoir.neurons.v

        mean_v_sum += float(np.mean(v))
        max_v = max(
            max_v,
            float(np.max(v))
        )

        ids = reservoir.core.get_step_spike_ids()

        total_spikes += len(ids)

        for i in ids:
            active_neurons.add(int(i))

        ca = reservoir.r

        max_ca = max(
            max_ca,
            float(np.max(ca))
        )

        mean_ca_sum += float(np.mean(ca))

    total_time_s = (
        n_steps * c.ds / 1000.0
    )

    firing_rate = (
        total_spikes
        / c.Nx
        / total_time_s
    )

    mean_ca = (
        mean_ca_sum / n_steps
    )

    print()
    print("scale:", scale)

    print(
        "total spikes:",
        total_spikes
    )

    print(
        "mean firing rate:",
        firing_rate,
        "spikes/s/neuron"
    )

    print(
        "active neurons:",
        len(active_neurons),
        "/",
        c.Nx
    )

    print(
        "active fraction:",
        len(active_neurons) / c.Nx
    )

    print(
        "mean Ca:",
        mean_ca
    )

    print(
        "max Ca:",
        max_ca
    )

    mean_v = mean_v_sum / n_steps

    print(
        "mean V:",
        mean_v
    )

    print(
        "max V:",
        max_v
    )