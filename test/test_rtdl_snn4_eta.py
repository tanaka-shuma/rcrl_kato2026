
import numpy as np

import common_configurator as common
from models.agent_rtdl import Agent

c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

# Cultured SNN: paper condition
c.rc_module = "models.snn_reservoir_snn4"
c.rc_class = "SpikingNeuralNetwork"

c.Nx = 576
c.Irate = 1.0 / 6.0
c.snn4_paper_condition = True
c.snn4_prerun_ms = 11000.0
c.input_const_snn4 = 0.02

# Keep the original TD learning rule
c.eps_greedy = False

np.random.seed(int(c.seed))

# Diagnostic: reduce learning rate only
c.eta_init *= 0.001
c.eta_final *= 0.001

agent = Agent(c)

env = common.generate_instance(
    c,
    module=c.env_module,
    class_=c.env_class
)

agent.initialize()
agent.reset(0)

print("eta2:", agent.eta2())

state, reward, done, info = env.reset()

# Before: action = 0
action = int(agent.a)

executed_action_counts = np.zeros(c.Ny, dtype=int)
max_abs_q = 0.0

print("Initial action:", action)

print("=== RTDL + Cultured SNN smoke test ===")

try:
    for k in range(200):
        executed_action_counts[action] += 1
        state, reward, done, info = env.step(action)

        action = agent.get_action(
            state, reward, done
        )

        ca = np.asarray(agent.reservoir.r)
        q = np.asarray(agent.q)
        wo = np.asarray(agent.Wo)

        ids = np.asarray(
            agent.reservoir.core.get_step_spike_ids()
        )

        e_spikes = np.count_nonzero(ids < 480)
        i_spikes = np.count_nonzero(ids >= 480)

        assert ca.shape == (576,)
        assert q.shape == (c.Ny,)
        assert np.all(np.isfinite(ca))
        assert np.all(np.isfinite(q))
        assert np.all(np.isfinite(wo))

        effective_gain = agent.eta2() * np.dot(ca, ca)

        max_abs_q = max(
            max_abs_q,
            float(np.max(np.abs(q)))
        )

        if k < 5 or (k + 1) % 20 == 0 or done:
            print(
                f"step={k+1:3d} "
                f"reward={reward:8.4f} "
                f"action={action} "
                f"E/I={e_spikes}/{i_spikes} "
                f"Ca_mean={np.mean(ca[:480]):8.4f} "
                f"Q={np.round(q, 4)} "
                f"Wo_norm={np.linalg.norm(wo):.6f} "
                f"gain={effective_gain:.4f}"
            )

        if done:
            print("Environment terminated.")
            break

    print("\n=== Summary ===")
    print("RL steps:", agent.n)
    print("RL elapsed [ms]:", agent.reservoir.t)
    print("Total reward:", agent.sum_reward)
    print("All values finite: True")
    print("Executed action counts:", executed_action_counts)
    print("Maximum |Q|:", max_abs_q)
    print("=== FINISHED ===")

finally:
    env.close()
