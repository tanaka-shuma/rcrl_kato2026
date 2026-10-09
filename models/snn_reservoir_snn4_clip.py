import numpy as np
from types import SimpleNamespace

from snn4tanaka.option.option import get_opts
from snn4tanaka.simu_clip import CulturedSNNCore


class SpikingNeuralNetwork:

    def __init__(self, c):

        self.c = c

        # ==========================================
        # snn4tanaka parameters
        # ==========================================
        p = get_opts()

        Ntot = int(c.Nx)
        Ni = int(Ntot * c.Irate)
        Ne = Ntot - Ni

        # RS, IB, FS, LTS
        p["N"] = np.array(
            [Ne, 0, Ni, 0],
            dtype=np.int32
        )

        p["dt"] = c.dt
        p["seed"] = int(c.seed)

        # ==========================================
        # Paper condition: N=576, Ke=20
        # ==========================================
        if getattr(c, "snn4_paper_condition", False):

            if (Ntot, Ne, Ni) != (576, 480, 96):
                raise ValueError(
                    "Paper condition requires N=576 "
                    "(Ne=480, Ni=96)"
                )

            p["mod"] = np.array([1, 1], dtype=np.int32)

            p["tca"] = 2550.0
            p["gk"] = 1.7e-5

            # Fast / slow synaptic weights
            p["g"][:] = 0.0

            p["g"][0, 0] = np.array([
                [4.93000e-4, 1.47900e-4],
                [6.85714e-5, 2.05714e-5]
            ])

            # NMDA
            p["g"][1, 0, 0, :] = (
                0.12 * p["g"][0, 0, 0, :]
            )

            # GABA_B
            p["g"][1, 0, 1, :] = (
                0.10 * p["g"][0, 0, 1, :]
            )

        self.core = CulturedSNNCore(p)

        self.Ntot = Ntot
        self.NNe = Ne
        self.NNi = Ni

        # snn4tanaka:
        # excitatory neurons first, inhibitory neurons second
        self.n_type = np.zeros(Ntot, dtype=np.int32)
        self.n_type[Ne:] = 1

        # ==========================================
        # Compatibility state for rcrl
        # ==========================================
        self.r = self.core.get_ca()

        self.spike_times = [
            [] for _ in range(Ntot)
        ]

        self.n = 0
        self.t = 0.0

        self.I_input = np.zeros(
            Ntot,
            dtype=np.float64
        )

        self.I_ext = np.zeros(
            Ntot,
            dtype=np.float64
        )

        # Compatibility view for main_rl.py
        self.neurons = SimpleNamespace()

        self.neurons.v = self.core.get_v()
        self.neurons.c = self.core.get_ca()
        self.neurons.spike = self.core.get_last_spikes()


    def reset(self):

        # reset dynamic state without rebuilding network
        self.core.reset()

        # 1 s preparation period used by snn4tanaka
        self.core.warmup()

        # Normal spontaneous activity before RL starts
        prerun_ms = float(
            getattr(self.c, "snn4_prerun_ms", 0.0)
        )

        n_steps = int(round(prerun_ms / self.c.ds))

        if not np.isclose(
            n_steps * self.c.ds,
            prerun_ms
        ):
            raise ValueError(
                "snn4_prerun_ms must be a multiple of ds"
            )

        for _ in range(n_steps):
            self.core.advance(self.c.ds)

        # Store the physical time at the beginning of the RL episode
        self.episode_time_origin_ms = (
            self.core.get_elapsed_ms()
        )

        # RL-visible state after warmup
        self.r = self.core.get_ca()

        self.neurons.v = self.core.get_v()
        self.neurons.c = self.r.copy()
        self.neurons.spike = self.core.get_last_spikes()

        self.spike_times = [
            [] for _ in range(self.Ntot)
        ]

        self.n = 0
        self.t = 0.0

        self.I_input[:] = 0.0
        self.I_ext[:] = 0.0


    def step(self, input):

        input = np.asarray(
            input,
            dtype=np.float64
        )

        if input.shape != (self.Ntot,):
            raise ValueError(
                f"input shape must be ({self.Ntot},), "
                f"but got {input.shape}"
            )

        # Agent gives Wi @ u.
        # Convert it to external current in the same way as old rcrl.
        Iin = input * self.c.input_const_snn4

        # Negative external current is clipped, as in the original model.
        Iin = np.where(
            Iin < 0.0,
            0.0,
            Iin
        )

        self.I_input[:] = Iin

        # One RL step (= ds ms), with constant external current
        # during all physical dt steps.
        self.core.advance_with_input(
            self.c.ds,
            self.I_input
        )

        # reservoir state = intracellular Ca concentration
        self.r = self.core.get_ca()

        self.neurons.v = self.core.get_v()
        self.neurons.c = self.r.copy()
        self.neurons.spike = self.core.get_last_spikes()

        # Record all spikes generated during this RL step
        ids = self.core.get_step_spike_ids()
        times = self.core.get_step_spike_times()

        for neuron_id, spike_time in zip(ids, times):
            self.spike_times[int(neuron_id)].append(
                float(spike_time - self.episode_time_origin_ms)
            )

        self.n += 1
        # self.t = self.core.get_elapsed_ms()
        self.t = (
            self.core.get_elapsed_ms()
            - self.episode_time_origin_ms
        )

        # Total current shown by plot_dynamics.
        # I_input is already in the same current unit here.
        self.I_ext[:] = (
            self.core.get_fast_current()
            + self.core.get_slow_current()
            + self.core.get_bg_current()
            + self.I_input
        )

    def done(self):

        total_spikes = sum(
            len(times)
            for times in self.spike_times
        )

        if self.t > 0.0:
            self.avg_spike_rate = (
                total_spikes
                / self.Ntot
                / (self.t / 1000.0)
            )
        else:
            self.avg_spike_rate = 0.0

        if hasattr(self.c, "avg_spike_rate"):
            self.c.avg_spike_rate = self.avg_spike_rate

        return None


    def sum_spike(self, spikeE, spikeI):

        # Count ALL spikes generated during the latest RL step,
        # not only the final 0.2-ms physical step.
        ids = self.core.get_step_spike_ids()

        for neuron_id in ids:

            i = int(neuron_id)

            if self.n_type[i] == 1:
                spikeI += 1
            else:
                spikeE += 1

        return spikeE, spikeI


    def spike_rate(self, c, sumE, sumI, nt):

        # nt is kept only for compatibility with main_rl.py.
        # Use actual elapsed physical time instead.
        Tsim = self.t

        if Tsim <= 0.0:
            return 0.0, 0.0, 0.0

        sumN = sumE + sumI

        spike_rate = (
            sumN
            / (Tsim / 1000.0)
            / self.Ntot
        )

        I_spike_rate = (
            sumI
            / (Tsim / 1000.0)
            / self.NNi
        )

        E_spike_rate = (
            sumE
            / (Tsim / 1000.0)
            / self.NNe
        )

        return (
            spike_rate,
            I_spike_rate,
            E_spike_rate
        )