"""Opt-in full-dt tracer for an existing CulturedSNNCore.

Does NOT change the Cython solver or the reinforcement-learning step size.
Only when enabled, partitions each core.advance_with_input(50 ms, I)
into 250 calls of core.advance_with_input(0.2 ms, I), keeping the
external current fixed and recording V and Ca after every physical update.
Also supports the input-free core.advance(duration_ms) path.

The proxy reassembles get_step_spike_* so callers continue to see ALL
spikes generated during the complete 50-ms RL step.
"""

import numpy as np


class RecordingCoreProxy:
    def __init__(self, core, dt_ms, voltage_ids, calcium_ids, n_exc,
                 record_full_ca=False, record_full_v=False):
        self.core = core
        self.dt_ms = float(dt_ms)
        self.voltage_ids = np.asarray(voltage_ids, dtype=np.intp)
        self.calcium_ids = np.asarray(calcium_ids, dtype=np.intp)
        self.n_exc = int(n_exc)
        self.record_full_ca = bool(record_full_ca)
        self.record_full_v = bool(record_full_v)
        if self.dt_ms <= 0:
            raise ValueError("dt_ms must be positive")
        if np.any(self.voltage_ids < 0) or np.any(self.voltage_ids >= len(core.get_v())):
            raise ValueError("voltage_ids out of range")
        if np.any(self.calcium_ids < 0) or np.any(self.calcium_ids >= self.n_exc):
            raise ValueError("calcium_ids must identify excitatory neurons")

        self.origin_ms = float(core.get_elapsed_ms())
        self.times_ms = []
        self.voltage = []
        self.calcium = []
        self.ca_mean = []
        self.ca_std = []
        self.full_calcium = [] if record_full_ca else None
        self.full_voltage = [] if record_full_v else None
        self._last_ids = np.empty(0, dtype=np.int32)
        self._last_times = np.empty(0, dtype=np.float64)
        self.calls = 0
        self.physical_steps = 0
        self._sample()  # t=0 sample, before the first physical integration

    def __getattr__(self, attr):
        # Pass other attributes and methods (neurons, set_external_input,
        # get_ca, get_v, get_elapsed_ms, etc.) through unchanged.
        return getattr(self.core, attr)

    def _sample(self):
        v = np.asarray(self.core.get_v(), dtype=np.float64)
        ca = np.asarray(self.core.get_ca(), dtype=np.float64)
        if len(v) != len(ca) or len(ca) < self.n_exc:
            raise ValueError("Unexpected V/Ca state lengths")
        self.times_ms.append(float(self.core.get_elapsed_ms()) - self.origin_ms)
        self.voltage.append(v[self.voltage_ids].copy())
        if self.record_full_v:
            self.full_voltage.append(v.copy())
        self.calcium.append(ca[self.calcium_ids].copy())
        exc = ca[:self.n_exc]
        self.ca_mean.append(float(np.mean(exc)))
        self.ca_std.append(float(np.std(exc)))
        if self.record_full_ca:
            self.full_calcium.append(exc.copy())

    def _advance_recorded(self, duration_ms, step_fn):
        """Run a complete RL interval, sample physical state, aggregate events.

        The original core overwrites its most-recent-step spike arrays on each
        call. We therefore collect those arrays before advancing again.
        """
        duration_ms = float(duration_ms)
        steps = int(round(duration_ms / self.dt_ms))
        if steps < 0 or not np.isclose(steps * self.dt_ms, duration_ms,
                                       rtol=0, atol=1e-7):
            raise ValueError(
                f"Duration {duration_ms} not a multiple of dt {self.dt_ms}"
            )

        ids_parts = []
        times_parts = []
        for _ in range(steps):
            step_fn()
            ids = np.asarray(
                self.core.get_step_spike_ids(), dtype=np.int32
            ).copy()
            times = np.asarray(
                self.core.get_step_spike_times(), dtype=np.float64
            ).copy()
            if ids.size != times.size:
                raise RuntimeError("Mismatched spike IDs/times in physical step")
            if ids.size:
                ids_parts.append(ids)
                times_parts.append(times)
            self._sample()
            self.physical_steps += 1

        self._last_ids = (np.concatenate(ids_parts) if ids_parts else
                          np.empty(0, dtype=np.int32))
        self._last_times = (np.concatenate(times_parts) if times_parts else
                            np.empty(0, dtype=np.float64))
        self.calls += 1

    def advance_with_input(self, duration_ms, input_current):
        """Intercept the actual SNN4 wrapper update used by RTDL.

        Keep the same input unchanged for the whole RL interval. Each call
        advances one physical dt and the raw Cython core performs the actual
        neural/synaptic update (not a Python approximation).
        """
        current = np.asarray(input_current, dtype=np.float64)
        self._advance_recorded(
            duration_ms,
            lambda: self.core.advance_with_input(self.dt_ms, current),
        )

    def advance(self, duration_ms):
        """Legacy path for callers using the input-free core.advance()."""
        self._advance_recorded(
            duration_ms, lambda: self.core.advance(self.dt_ms)
        )

    def get_step_spike_ids(self):
        return self._last_ids

    def get_step_spike_times(self):
        # Same absolute/core-clock convention as underlying CulturedSNNCore.
        return self._last_times

    def get_step_spike_count(self):
        return int(self._last_ids.size)

    def export(self):
        if self.physical_steps + 1 != len(self.times_ms):
            raise RuntimeError("Incomplete physical trace")
        d = {
            "physical_times_ms": np.asarray(self.times_ms, dtype=np.float64),
            "v_physical": np.asarray(self.voltage, dtype=np.float64),
            "ca_e_samples_physical": np.asarray(self.calcium, dtype=np.float64),
            "ca_e_mean_physical": np.asarray(self.ca_mean, dtype=np.float64),
            "ca_e_std_physical": np.asarray(self.ca_std, dtype=np.float64),
            "physical_dt_ms": np.asarray(self.dt_ms),
        }
        if self.record_full_v:
            d["v_all_physical"] = np.asarray(self.full_voltage, dtype=np.float64)
        if self.record_full_ca:
            d["ca_e_all_physical"] = np.asarray(self.full_calcium, dtype=np.float64)
        return d
