
import numpy as np
import common_configurator as common
from models.agent_rtdl import Agent

c = common.load_config(
    "config.config_rl_ninerooms_rtdl_snn"
)

c.rc_module = "models.snn_reservoir_snn4"
c.rc_class = "SpikingNeuralNetwork"

c.Nx = 576
c.Irate = 1.0 / 6.0
c.snn4_paper_condition = True
c.snn4_prerun_ms = 11000.0
c.input_const_snn4 = 0.02

c.eta_init *= 0.001
c.eta_final *= 0.001
c.eps_greedy = False

NUM_EPISODES = 5
MAX_STEPS = 200

np.random.seed(int(c.seed))

agent = Agent(c)
env = common.generate_instance(
    c, module=c.env_module,
    class_=c.env_class
)

agent.initialize()

print("=== Five-episode RTDL test ===")
print("eta:", c.eta_init, c.eta_final)

try:
    for episode in range(NUM_EPISODES):

        agent.reset(episode)
        state, reward, done, info = env.reset()

        # Match the first executed action
        action = int(agent.a)

        actions = np.zeros(c.Ny, dtype=int)
        max_abs_q = 0.0
        terminated = False

        for k in range(MAX_STEPS):

            actions[action] += 1

            state, reward, done, info = env.step(action)
            action = agent.get_action(state, reward, done)

            q = np.asarray(agent.q)
            ca = np.asarray(agent.reservoir.r)
            wo = np.asarray(agent.Wo)

            assert np.all(np.isfinite(q))
            assert np.all(np.isfinite(ca))
            assert np.all(np.isfinite(wo))

            max_abs_q = max(
                max_abs_q,
                float(np.max(np.abs(q)))
            )

            if done:
                terminated = True
                break

        # Recalculate Q using the updated weights
        q_after_update = (
            agent.Wo @ agent.reservoir.r
        )

        print(
            f"\nepisode={episode}"
            f" steps={agent.n}"
            f" terminated={terminated}"
            f" reward={agent.sum_reward:.5f}"
        )

        print("actions:", actions.tolist())
        print("max |Q|:", max_abs_q)
        print("Q after update:", np.round(q_after_update, 5))
        print("Wo norm:", np.linalg.norm(agent.Wo))
        print("Ca mean E:", np.mean(ca[:480]))

    print("\n=== FINISHED ===")

    np.savez(
        "rtdl_probe_weights.npz",
        Wi=agent.Wi.copy(),
        Wo=agent.Wo.copy()
    )

finally:
    env.close()
