import numpy as np

class ExponentialFilter: 
    """
    指数減衰フィルター
    """
    def __init__(self, N, dt=0.2, tau=5.0, ex_max=1.0):
        self.N = N
        self.dt = dt
        self.tau = tau
        self.r =np.zeros(N)
        self.ex_max = ex_max

    def initialize_states(self):
        self.r=np.zeros(self.N)

    def __call__(self, spike):
        self.r += self.ex_max * spike - self.r / self.tau*self.dt

        return self.r
    

class DoubleExponentialFilter:
    """
    二重指数減衰フィルター
    """
    def __init__(self, N, dt=0.2, tau_d=5, tau_r=2,ex_max=1.0):
        self.N = N
        self.dt = dt
        self.tau_d = tau_d
        self.tau_r = tau_r
        self.rmax = 1 / ((1/tau_d) * (tau_r/tau_d)**(tau_r/(tau_d-tau_r)))
        self.r = np.zeros(N)
        self.hr = np.zeros(N)
        self.ex_max = ex_max

    def initialize_states(self):
        self.r = np.zeros(self.N)
        self.hr = np.zeros(self.N)

    def __call__(self, spike):
        self.r = self.r*(1 - self.dt/self.tau_d) + self.hr * self.dt
        self.hr = self.hr * (1 - self.dt/self.tau_r) + (spike / (self.tau_r*self.tau_d)) * self.rmax * self.ex_max

        return self.r