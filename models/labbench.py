# Copyright (c) 2023 Katori lab. All Rights Reserved
# 物理リザバーのためのモジュール、時間遅れ、パルスによる刺激、計測値の時間平均
# TODO 刺激と計測の方法を拡充する
import numpy as np

class DelayLine:
    def __init__(self, depth: int, N: int):
        """Initialize a DelayLine object with a given depth and length N."""
        self.depth = depth
        self.state = np.zeros((depth, N))

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Process the incoming signal through the delay line.
        Returns the output of the delay line.
        """
        # If depth is 0, return the signal as is
        if self.depth == 0:
            return signal

        out = self.state[0, :]  # Output the first row
        self.state = np.roll(self.state, -1, axis=0)  # Shift rows upwards
        self.state[-1, :] = signal.reshape(-1)  # Add the signal to the last row
        return out

class StimulatorPulse:
    def __init__(self, dt: float = 0.1, pulse_width: float = 10, gain: float = 1.0,  N:int = 0):
        """Initialize the Stimulation class with given parameters."""
        self.pulse_width = pulse_width  # (ms)
        self.gain = gain
        self.dt = dt  # (ms)
        self.N = N
        self.t_start = 0
        self.input_signal = np.zeros(N)
        
    def set(self, t: float, input_signal: np.ndarray):
        """Set the starting time and input signal."""
        self.t_start = t
        self.input_signal = input_signal
        self.N = len(input_signal)

    def get(self, t: float) -> np.ndarray:
        """Return the pulse signal"""
        time = (t - self.t_start) * self.dt
        if time < self.pulse_width:
            out = self.input_signal * self.gain
        else:
            out = np.zeros(self.N)
        return out
    
class ProbeAverage:
    def __init__(self, dt: float = 1.0, gain: float = 1.0, N: int = 0):
        """Initialize the Probe class with given parameters."""
        self.gain = gain
        self.N = N
        self.count = 0
        self.sum = np.zeros(self.N)

    def set(self, t: float):
        """Initialize the count and sum variables."""
        self.count = 0
        self.sum = np.zeros(self.N)

    def measure(self, x: np.ndarray):
        """Measure the given input."""
        assert len(x) == self.N, "Input length must match N"
        self.sum += x
        self.count += 1

    def get(self) -> np.ndarray:
        """Calculate and return the average of the measured values."""
        return self.sum / self.count if self.count else np.zeros(self.N)