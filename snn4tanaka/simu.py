# This program simulates a single excitatory spiking neuron
# created by H. Kato, Oita Univ., Japan
# <last modified 13:40 23-Apr-2021 JST>

# cython: language_level = 3
# cython: boundscheck = False
# cython: initializedcheck = False
# cython: cdivision = True
# cython: always_allow_keywords = False
# cython: unraisable_tracebacks = False
# cython: binding = False

import numpy as np
import networkx as nx
from itertools import product
import cython
from option.option import get_opts
from cython.cimports.libc.math import exp

#Neuron model
@cython.cclass
class frenchLIF:
    """
    File name: French_LIF.py
    All right reserved by H. Kato (Oita Univ. Japan)
    <Last modified: Sun Apr 16 2:28PM>
    
    Definition of improved French model

    [References]
    French, D. A. & Gruenstein, E. I.
    "An integrate-and-fire model for synchronized bursting in a network of cultured cortical neurons"
    J Comput Neurosci (2006) 21:227–241

    The dynamics of gref in the model is replaced with a multi-time scale exponential function instead of a inverse-propotional function
    """

    # original variables
    N: np.ndarray
    Npop: cython.int
    Ntot: cython.int
    Ne: cython.int
    Ni: cython.int
    dt: cython.double
    sabs: np.ndarray
    smem: np.ndarray
    sca: np.ndarray
    Vrest: np.ndarray
    Vreset: np.ndarray
    Vthre: np.ndarray
    Rin: np.ndarray
    Ek: np.ndarray
    gk: np.ndarray
    Cstep: np.ndarray
    gref_: np.ndarray
    m_gref: cython.int
    sref: np.ndarray

    V: np.ndarray
    Itot: np.ndarray
    Ica: np.ndarray
    Iref: np.ndarray
    gref: np.ndarray
    c: np.ndarray
    sabs_counter: np.ndarray
    firings_vec: np.ndarray

    # for memoryviews
    _sabs: cython.int[:]
    _smem: cython.double[:]
    _sca: cython.double[:]
    _Vrest: cython.double[:]
    _Vreset: cython.double[:]
    _Vthre: cython.double[:]
    _Rin: cython.double[:]
    _Ek: cython.double[:]
    _gk: cython.double[:]
    _Cstep: cython.double[:]
    _gref_: cython.double[:,:]
    _sref: cython.double[:,:]

    _V: cython.double[:]
    _Itot: cython.double[:,:]
    _Ica: cython.double[:]
    _Iref: cython.double[:]
    _gref: cython.double[:,:]
    _c: cython.double[:]
    _sabs_counter: cython.long[:]
    _firings_vec: cython.int[:]
    
    def __init__(self, N=np.array([800, 0, 200, 0]), dt=0.2, tabs=np.array([5.0, 5.0, 2.0, 2.0]), tmem=np.array([20.0, 20.0, 10.0, 10.0]), vrest=np.array([-74.0, -74.0, -70.0, -70.0]), vreset=np.array([-60.0, -60.0, -60.0, -60.0]), vthre=np.array([-54.0, -54.0, -50.0, -50.0]), Rin=np.array([40e3, 40e3, 50e3, 50e3]), Cstep=0.1, tca=2700, gk=10e-6, Ek=-75, gref_=np.array([147e-6, 2e-6, 1e-6]), tref=np.array([12.0, 550.0, 1800.0])):

        """
        All variables and parameters in the neuron model are prepared here.
        
        [Parameters]
        N: list of individual types of neuron #.
           the order of list is RS, IB, FS, LTS types of neurons. {default: N=np.array([800, 0, 200, 0]}
        dt: time resolution (default: dt=0.2 ms)
        tmem: list of membrane time constants {default: tmem=np.array([20.0, 20.0, 10.0, 10.0]) ms}
        vrest: list of resting potentials {default: vrest=np.array([-74.0, -74.0, -70.0, -70.0]) mV}
        vreset: list of reset values of potential {default: vreset=np.array([-60.0, -60.0, -60.0, -60.0]) mV}
        vthre: list of threshold values of potentials {default: vthre=np.array([-54.0, -54.0, -50.0, -50.0]) mV}
        Rin: list of registance values of input {default: Rin=np.array([40.0, 40.0, 50.0, 50.0]) kΩ}
        tabs: list of absolute refractory periods {default: tabs=np.array([2.0, 2.0, 2.0, 2.0]) ms}
        Cstep: imcreament value of Calcium ion density when spiking {default: Cstep=0.1}
        tca: time constant of Calcium ion density dynamics {default: tca=2700 ms}
        gk: conductance value for Calcium-dependent Potacium current {default: gk=10e-6 mS}
        Ek: reversal potential for Calcium-dependent Potacium current {default: Ek=-75.0 mV}
        gref_: list of decrement values of multi-time-scale conductance for refractory current {default: gref_=np.array([147e-6, 2e-6, 1e-6]) mS}
        tref: list of time constants of multi-time-scale conductance for refractory current {default: tref=np.array([12.0, 550.0, 1800.0]) ms}
        """

        # parameters
        self.N      = N
        self.Npop   = np.size(self.N)                                     # types of neurons
        self.Ntot   = np.sum(self.N)                                      # total # of neurons
        self.Ne     = np.sum(self.N[:2])                                  # # of E neurons
        self.Ni     = np.sum(self.N[2:])                                  # # of I neurons
        self.dt     = dt                                                  # dt
        self.sabs   = np.array(np.r_[np.ones(self.N[0])*tabs[0]/self.dt,
                                     np.ones(self.N[1])*tabs[1]/self.dt,
                                     np.ones(self.N[2])*tabs[2]/self.dt,
                                     np.ones(self.N[3])*tabs[3]/self.dt], dtype=np.int32)  # absolute reflactory period (steps)
        self.smem   = np.r_[np.ones(self.N[0])*self.dt/tmem[0],
                            np.ones(self.N[1])*self.dt/tmem[1],
                            np.ones(self.N[2])*self.dt/tmem[2],
                            np.ones(self.N[3])*self.dt/tmem[3]]           # dt/tmem
        self.sca    = np.r_[np.exp(-self.dt*np.ones(self.Ne)/tca),
                            np.zeros(self.Ni)]                            # decay rate for Calcium ion density
        self.Vrest  = np.r_[vrest[0]*np.ones(self.N[0]),
                            vrest[1]*np.ones(self.N[1]),
                            vrest[2]*np.ones(self.N[2]),
                            vrest[3]*np.ones(self.N[3])]                  # Vrest
        self.Vreset = np.r_[vreset[0]*np.ones(self.N[0]),
                            vreset[1]*np.ones(self.N[1]),
                            vreset[2]*np.ones(self.N[2]),
                            vreset[3]*np.ones(self.N[3])]                 # Vreset
        self.Vthre  = np.r_[vthre[0]*np.ones(self.N[0]),
                            vthre[1]*np.ones(self.N[1]),
                            vthre[2]*np.ones(self.N[2]),
                            vthre[3]*np.ones(self.N[3])]                  # Vthre
        self.Rin    = np.r_[Rin[0]*np.ones(self.N[0]),
                            Rin[1]*np.ones(self.N[1]),
                            Rin[2]*np.ones(self.N[2]),
                            Rin[3]*np.ones(self.N[3])]                    # Rin
        self.Ek     = Ek*np.ones(self.Ntot)                               # Ek
        self.gk     = np.r_[gk*np.ones(self.Ne),
                            np.zeros(self.Ni)]                            # gk
        self.Cstep  = np.r_[Cstep*np.ones(self.Ne),
                            np.zeros(self.Ni)]                            # Cstep
        self.m_gref = len(gref_)
        self.gref_  = np.r_[np.tile(gref_, (self.Ne, 1)),
                            np.tile(0*gref_, (self.Ni, 1))]               # gref_
        self.sref   = np.r_[np.tile(np.exp(-self.dt/tref), (self.Ne, 1)),
                            np.tile(0*np.exp(-self.dt/tref), (self.Ni, 1))] # decay rate for variables of relative refractory period

        # variables
        self.V            = self.Vrest.copy()
        self.Itot         = np.zeros((self.Ntot,2))           
        self.Ica          = np.zeros(self.Ntot)
        self.Iref         = np.zeros(self.Ntot)
        self.gref         = np.zeros(np.shape(self.gref_))
        self.c            = np.zeros(self.Ntot)
        self.sabs_counter = np.zeros(self.Ntot, dtype=np.int64)
        self.firings_vec  = np.zeros(self.Ntot, dtype=np.int32)

        self._sabs = self.sabs 
        self._smem = self.smem
        self._sca = self.sca
        self._Vrest = self.Vrest
        self._Vreset = self.Vreset
        self._Vthre = self.Vthre
        self._Rin = self.Rin
        self._Ek = self.Ek
        self._gk = self.gk
        self._Cstep = self.Cstep
        self._gref_ = self.gref_
        self._sref = self.sref

        self._V = self.V
        self._Itot = self.Itot
        self._Ica = self.Ica
        self._Iref = self.Iref
        self._gref = self.gref
        self._c = self.c
        self._sabs_counter = self.sabs_counter
        self._firings_vec = self.firings_vec
        

    @cython.cfunc
    def update(self, Isyn: cython.double[:], Isyn_slow: cython.double[:], Ibg: cython.double[:]) -> cython.void:
        """
        Time evolution of internal variables with Eular's method
        Cython version
        """

        i: cython.int
        j: cython.int
        k: cython.int
        gref: cython.double

        for i in range(self.Ntot):
            # computing currents
            self._Ica[i] = self._gk[i]*self._c[i]*(self._Ek[i]-self._V[i])
            gref = .0
            for j in range(self.m_gref):
                gref += self._gref[i,j]
            self._Iref[i] = gref*(self._V[i]-self._Vreset[i])
            self._Itot[i,1] = self._Ica[i] + self._Iref[i]*(1.0-self._firings_vec[i]) + Isyn[i] + Isyn_slow[i] + Ibg[i]
                    
            # update variables
            k=int(self._sabs_counter[i]>self._sabs[i])
            self._V[i] += (self._Vrest[i]-self._V[i] + self._Rin[i]*self._Itot[i,k])*self._smem[i]
            self._c[i] *= self._sca[i]
            for j in range(self.m_gref):
                self._gref[i,j] *= self._sref[i,j]
            self._sabs_counter[i] += 1


    @cython.cfunc
    def det_firings(self) -> cython.void:
        """
        Determine firing or not
        Cython version
        """
            
        i: cython.int
        j: cython.int

        for i in range(self.Ntot):
            # firing?
            if (self._V[i]<self._Vthre[i]) or (self._sabs_counter[i]<=self._sabs[i]):
                self._firings_vec[i] = 0
            else:
                self._V[i] = self._Vreset[i]
                self._c[i] += self._Cstep[i]
                    
                for j in range(self.m_gref):
                    self._gref[i,j] -= self._gref_[i,j]
                        
                self._sabs_counter[i] = 0
                self._firings_vec[i] = 1
    

    def get_parms(self):
        """ Get parameter values of neurons """

        return np.array([self.Vrest[0], self.Vthre[0]])

    @cython.cfunc
    def get_V(self) -> cython.double[:]:
        """ 
        Get membrane values 
        Cython version
        """
            
        return self._V

        
    @cython.cfunc
    def get_firings_vec(self) -> cython.int[:]:

        return self._firings_vec
        

    @cython.cfunc
    def display_neuronal_internal_states(self, idx: cython.int, t: cython.int, s: cython.int, Isyn2: cython.double[:,:], Isyn: cython.double[:], Isyn2_: cython.double[:,:], Isyn_: cython.double[:], Ibg2: cython.double[:,:], Ibg: cython.double[:]) -> cython.void:
        if (self._V[idx]<self._Vthre[idx]) or (self._sabs_counter[idx]<=self._sabs[idx]):
            print('%.5f %e %e %e %e %e %e %e %e %e %e %e %e' % (t+0.001*s*self.dt, self._V[idx], self._Iref[idx], self._Ica[idx], Isyn2[idx,0], Isyn2[idx,1], Isyn[idx], Isyn2_[idx,0], Isyn2_[idx,1], Isyn_[idx], Ibg2[idx,0], Ibg2[idx,1], Ibg[idx]))
        else:
            print('%.5f %e %e %e %e %e %e %e %e %e %e %e %e' % (t+0.001*s*self.dt, 15.0, self._Iref[idx], self._Ica[idx], Isyn2[idx,0], Isyn2[idx,1], Isyn[idx], Isyn2_[idx,0], Isyn2_[idx,1], Isyn_[idx], Ibg2[idx,0], Ibg2[idx,1], Ibg[idx]))
            


#Synapse model
@cython.cclass
class ExpModel:
    """
    Definition of exponetial model

    ### variables ###
    Nreg: # of regular spiking neurons
    """

    Nreg: cython.int
    Nib: cython.int
    Nfs: cython.int
    Nlts: cython.int
    Ne: cython.int
    Ni: cython.int
    N: cython.int

    dt: cython.double
    tau: np.ndarray
    g: np.ndarray
    
    _tau: cython.double[:,:]
    _g: cython.double[:,:]


    def __init__(self, N=np.array([2500, 2500, 800, 200]), dt=0.2,
                 tau = np.array([2.0, 5.0])):
        """
        All the parameters and the variables in the synaose model are set here.

        [Argments]
        N:       # of each type of neurons, Regular spiking (RS), Intrinsic bursting (IB), Fast spking (FS),
                 Low-threshold spiking (LTS)
                 (default: =np.array([2500, 2500, 800, 200]))
        dt:      time resolution (default: =0.1)
        tau_syn: synaptic time constant
        """

        self.dt = dt
        self.Nreg, self.Nib, self.Nfs, self.Nlts, self.Ne, self.Ni, self.N = N[0], N[1], N[2], N[3], N[0]+N[1], N[2]+N[3], np.sum(N)

        self.tau = np.exp(-self.dt/np.tile(tau, (self.N, 1)))
        self.g   = np.zeros(self.tau.shape)

        
        self._tau = self.tau
        self._g = self.g


    @cython.cfunc
    def update(self) -> cython.void:
        """
        Time evolution of internal variables with Eular's method

        Cython version
        """

        i: cython.int
        j: cython.int

        for i in range(self.N):
            for j in range(2):
                self._g[i,j] *= self._tau[i,j]


    @cython.cfunc
    def presynaptic_input(self, g_: cython.double[:,:]) -> cython.void:
        """
        Imcrement synaptic variables

        Cython version
        """

        i: cython.int
        j: cython.int

        for i in range(self.N):
            for j in range(2):
                self._g[i,j] += g_[i,j]
            

    @cython.cfunc
    def get_g(self) -> cython.double[:,:]:
        """
        Get synaptic conductance values

        Cython version
        """
            
        return self._g


@cython.cclass
class OU_like_process:
    """ 
    Definition of OU like process.

    [References]
    Destexhe A, Rudolph M, Fellous JM, Sejnowski TJ,
    "Fluctuating synaptic conductances recreate in vivo–like activity in neocortical neurons,"
    Neuroscience 107: 13–24, 2001

    ### variables ###
    Nreg: # of regular spiking neurons
    Nib:  # of intrinsic bursting neurons
    Nfs:  # of fast spiking neurons
    Nlts: # of low threshold spiking neurons
    Ne:   # of excitatory neurons (Nreg + Nib)
    Ni:   # of inhibitory neurons (Nfs + Nlts)
    N:    total # of neurons (Ne + Ni)
    dt:   time resolution of computation (ms)
    s:    time steps
    S:    steps for 1 sec

    g:       synaptic conductance
    tau_syn: synaptic time constant
    g0:      mean value of g
    sigma:   noise strength
    w:       white Gaussian noise for 1 sec
    """

    # for memoryviews
    dt: cython.double
    S: cython.int
    s: cython.int
    Nreg: cython.int
    Nib: cython.int
    Nfs: cython.int
    Nlts: cython.int
    Ne: cython.int
    Ni: cython.int
    N: cython.int
    tau_syn: np.ndarray
    g0: np.ndarray
    sigma: np.ndarray
    g: np.ndarray
    w: np.ndarray
        
    _tau_syn: cython.double[:,:]
    _g0: cython.double[:,:]
    _sigma: cython.double[:,:]
    _g: cython.double[:,:]
    _w: cython.double[:,:,:]

    def __init__(self, N=np.array([800, 0, 200, 0]), dt=0.2,
                 g0=np.array([[2.550000e-04, 0, 1.925000e-04, 0],[1.483636e-03, 0, 7.700000e-04, 0]]),
                 sigma=np.array([[1.275000e-04, 0, 7.931000e-05, 0],[2.967273e-04, 0, 1.268960e-04, 0]]),
                 tau_syn=np.array([2.0, 5.0])):
        """ 
        All the parameters and variables of OU like process are set here 

        [Argments]
        N:       # of each type of neurons, Regular spiking (RS), Intrinsic bursting (IB), Fast spking (FS), 
                 Low-threshold spiking (LTS) 
                 (default: =np.array([2500, 2500, 800, 200]))
        dt:      time resolution (default: =0.1)
        g0:      the mean value of OU like process 
                 1st row: for excitatory
                 2nd row: for inhibitory
                 (defalt: =np.array([[4.0e-1, 4.0e-1, 4.0e-1, 4.0e-1],[1.2e0, 1.2e0, 1.2e0, 1.2e0]]))
        sigma:   noise strength of white Gaussian
                 1st row: for excitatory
                 2nd row: for inhibitory
                 (defalt: =np.array([[2.5e-1, 2.5e-1, 2.5e-1, 2.5e-1],[3.0e-1, 3.0e-1, 3.0e-1, 3.0e-1]]))
        tau_syn: excitatory and inihibitory synaptic time constants
                 (defalt: =np.array([2.0, 5.0]))
        """

        self.dt = dt
        self.s, self.S = 0, int(np.round(1000/self.dt))
        self.Nreg, self.Nib, self.Nfs, self.Nlts, self.Ne, self.Ni, self.N = N[0], N[1], N[2], N[3], N[0]+N[1], N[2]+N[3], np.sum(N)
        self.tau_syn = np.c_[tau_syn[0]*np.ones(self.N), tau_syn[1]*np.ones(self.N)]
        self.tau_syn = 1.0/self.tau_syn
        self.g0 = np.c_[np.r_[g0[0,0]*np.ones(self.Nreg), g0[0,1]*np.ones(self.Nib), g0[0,2]*np.ones(self.Nfs), g0[0,3]*np.ones(self.Nlts)],
                        np.r_[g0[1,0]*np.ones(self.Nreg), g0[1,1]*np.ones(self.Nib), g0[1,2]*np.ones(self.Nfs), g0[1,3]*np.ones(self.Nlts)]]
        self.sigma = np.c_[np.r_[sigma[0,0]*np.ones(self.Nreg), sigma[0,1]*np.ones(self.Nib), sigma[0,2]*np.ones(self.Nfs), sigma[0,3]*np.ones(self.Nlts)],
                        np.r_[sigma[1,0]*np.ones(self.Nreg), sigma[1,1]*np.ones(self.Nib), sigma[1,2]*np.ones(self.Nfs), sigma[1,3]*np.ones(self.Nlts)]]
        self.w = np.random.randn(self.N, 2, self.S) # white Gaussian random numbers for 1 sec
        self.g = np.zeros((self.N, 2))

        # for memoryviews
        self._tau_syn = self.tau_syn
        self._g0 = self.g0
        self._sigma = self.sigma
        self._g = self.g
        self._w = self.w

    @cython.cfunc
    def update(self) -> cython.void:
        """ 
        Time evolution of internal variables with Eular's method 
        dg/dt = (g0 - g)/tau_syn + sigma * N(0, 0)
        where N(0, 0) is the white Gaussian random number with zero mean and SD
        """

        i: cython.int
        j: cython.int
            
        for i in range(self.N):
            for j in range(2):
                self._g[i,j] += ((self._g0[i,j] - self._g[i,j])*self._tau_syn[i,j] + self._sigma[i,j]*self._w[i,j,self.s])*self.dt
                    
        self.s += 1

    @cython.cfunc
    def ready_for_next_sec(self)->cython.void:
        """ Ready for the next sec """

        self._w = np.random.randn(self.N, 2, self.S) # update white Gaussian random numbers for 1 sec
        self.s = 0


    @cython.cfunc
    def get_g(self) -> cython.double[:,:]:
        """ 
        Get synaptic conductances 
        Cython version
        """
            
        return self._g

        
    
@cython.cclass
class Current:
    """ 
    Definition of current

    ### variables ###
    """

    Nreg: cython.int
    Nib: cython.int
    Nfs: cython.int
    Nlts: cython.int
    Ne: cython.int
    Ni: cython.int
    N: cython.int
    E: np.ndarray
    I: np.ndarray
    Itot: np.ndarray
    flag_slow: cython.bint

    _E: cython.double[:,:]
    _I: cython.double[:,:]
    _Itot: cython.double[:]
        

    def __init__(self, N=np.array([2500, 2500, 800, 200]), Vrev=np.array([.0, -80.0]), flag_slow=False):
        """ All the parameters and variables of current are set here """

        self.Nreg, self.Nib, self.Nfs, self.Nlts, self.Ne, self.Ni, self.N = N[0], N[1], N[2], N[3], N[0]+N[1], N[2]+N[3], np.sum(N)
        self.E = np.c_[Vrev[0]*np.ones(self.N), Vrev[1]*np.ones(self.N)]
        self.I = np.zeros(self.E.shape)
        self.Itot = np.zeros(self.N)
        self.flag_slow = flag_slow

        # for memoryviews
        self._E = self.E
        self._I = self.I
        self._Itot = self.Itot


    @cython.cfunc
    def B(self, v: cython.double, flag_slow: cython.bint) -> cython.double:
        b: cython.double

        if not flag_slow:
            b = 1.0
        else:
            b = 1.0 / (1.0 + 0.28*exp(-0.062*v))

        return b

            
    @cython.cfunc
    def update(self, v: cython.double[:], g: cython.double[:,:]) -> cython.void:
        """ 
        Calculate  current from conductance 
        v: membrane potential
        g: conductance

        When the current value of the potential is given to v, the current is the conductance-based model.
        If v is the constant as the resting potential, the current is the current-based model.

        Cython version
        """

        i: cython.int
            
        for i in range(self.N):
            self._I[i,0] = self.B(v[i], self.flag_slow)*g[i,0]*(self._E[i,0] - v[i])
            self._I[i,1] = g[i,1]*(self._E[i,1] - v[i])
            self._Itot[i] = self._I[i,0] + self._I[i,1]

    @cython.cfunc
    def get_I(self) -> cython.double[:]:
        """ 
        Get current
            
        Cython version
        """

        return self._Itot


    @cython.cfunc
    def get_I2(self) -> cython.double[:,:]:
        """ 
        Get current
            
        Cython version
        """

        return self._I
        

@cython.cclass
class ModularNetwork:
    """
    All right reserved by H. Kato (Oita Univ., Japan)
    <Last modified: Sat May 27 9:00AM>

    Definition of synaptically connected modular neuronal network based on electrophysiological experiments conducted by Prof. H. Yamamoto.
    """

    # __init__()
    dt: cython.double
    S: cython.int
    N: np.ndarray
    Npop: cython.int
    Ntot: cython.int
    Ne: cython.int
    Ni: cython.int
    mod_shape: np.ndarray
    neuron_type: np.ndarray
    _neuron_type: cython.int[:]
    N_firings_max: cython.int
    N_firings: cython.int
    firings: np.ndarray
    _firings: cython.int[:,:]

    # __gen_modullars()
    Nmod:cython.int
    mod_str: tuple

    # __assign_neurons_to_modulars()
    Neach: np.ndarray
    assigned_modular: np.ndarray
    mod_base_idx: np.ndarray
    _assigned_modular: cython.int[:]
    _mod_base_idx: cython.int[:]

    # __prepare_pres()
    Npre_intra: np.ndarray
    Npre_inter: np.ndarray

    # __gen_syn_connectivity()
    Npre: np.ndarray
    pre: np.ndarray

    # __trans_pre_post()
    Npost: np.ndarray
    post: np.ndarray
    _Npost: cython.int[:]
    _post: cython.int[:,:]

    # __det_synaptic_strength()
    g: np.ndarray
    g_slow: np.ndarray
    _g: cython.double[:,:]
    _g_slow: cython.double[:,:]

    # __det_conduction_delays()
    Ds_min: cython.int
    Ds_max: cython.int
    delays_length: np.ndarray
    delays: np.ndarray
    _delays_length: cython.int[:,:]
    _delays: cython.int[:,:,:]

    
    def __init__(self, N=np.array([800, 0, 200, 0]), mod_shape=np.array([3,4]), sigma_intra=0.0, mu_inter=5.0, sigma_inter=0.0, g_=np.array([[[[5e-6, 5e-6], [2e-5, 1.5e-5]], [[2.5e-6, .0], [0, 0]]], [[[5e-6, 5e-6], [2e-5, 1.5e-5]], [[2.5e-6, .0], [0, 0]]]]), dt=0.2, D_=np.array([[[0.6, 1.0],[0.6, 1.0]], [[1.2, 1.4],[1.2, 1.4]]])):
        self.dt     = dt
        self.S      = int(1000*round(1.0/self.dt))
        self.N      = N
        self.Npop   = np.size(self.N)
        self.Ntot   = np.sum(self.N)
        self.Ne     = np.sum(self.N[:2])
        self.Ni     = np.sum(self.N[2:])
        self.mod_shape = mod_shape
        self.neuron_type = np.array(np.zeros(self.Ntot), dtype=np.int32)
        self.neuron_type[self.Ne:] = 1

        self.__gen_modulars()                                                 # generage modular connectivity
        self.__assign_neurons_to_modulars()                                   # assign neurons to each modular
        self.__gen_syn_connectivity(sigma_intra, mu_inter, sigma_inter) # generate synaptic connectivity
        self.__trans_pre_post()                                               # translate pre information to post information

        self.__det_synaptic_strength(g_)     # generate synaptic strength matrix
        # self.__det_synaptic_strength()         # generate synaptic strength matrix according to the scaling rule
        self.__det_conduction_delays(dt, D_)   # determine spike conduction delays

        self.N_firings_max = 1000*self.Ntot # maximum firings
        self.firings = np.array(np.zeros((2, self.N_firings_max)), dtype=np.int32)
        # a dummy spike
        self.firings[0,0] = -self.Ds_max
        self.firings[1,0] = 0
        self.N_firings = 1

        self._firings = self.firings
        self._neuron_type = self.neuron_type
        self._assigned_modular = self.assigned_modular
        self._mod_base_idx = self.mod_base_idx
        
    def __gen_modulars(self):
        """
        generate modular connectivity
        """

        self.Nmod    = self.mod_shape.prod() # total # of modulars
        self.mod_str = np.where(nx.to_numpy_array(nx.grid_graph(dim=list(self.mod_shape)))) # connectivity among modulars

        
    def __assign_neurons_to_modulars(self):
        """
        assign neurons to modulars
        """
        
        self.Neach=np.tile(np.array(self.N/self.Nmod, dtype=np.int32), (self.Nmod, 1)) # set neuron # asigned to each module
        Nlack=self.N-np.array(np.sum(self.Neach, axis=0), dtype=np.int32)
        for i in range(self.Npop):
            self.Neach[np.random.choice(range(self.Nmod), Nlack[i], replace=False), i] += 1 # complete neuron # in eahc module

        Neach_cum=self.Neach.reshape(np.prod(self.Neach.shape), order='F').cumsum().reshape(self.Neach.shape, order='F').T # generation of ranges

        # assign neurons to modules
        self.assigned_modular = -np.ones(self.Ntot, dtype=np.int32)
        for i, j in product(range(Neach_cum.shape[0]), range(Neach_cum.shape[1])):
            if i!=0 or j!=0:
                s, e = e, Neach_cum[i,j]
            else:
                s, e = 0, Neach_cum[i,j]
            self.assigned_modular[s:e] = j

        # neuron indecies based on modulues
        k, self.mod_base_idx = 0, -np.ones(self.Ntot, dtype=np.int32)
        for i in range(self.Nmod):
            tmp = np.where(self.assigned_modular==i)[0]
            self.mod_base_idx[tmp] = np.arange(len(tmp)) + k
            k += len(tmp)

        
    def __gen_syn_connectivity(self, sigma_intra, mu_inter, sigma_inter):
        """
        generate synaptic connectivity
        """
        
        self.__prepare_pres(sigma_intra, mu_inter, sigma_inter)

        # count # of presynaptic neurons on each postsynaptic neuron
        self.Npre = self.Npre_intra[:,self.Npre_intra.shape[1]-1]

        # prepare for all connections
        self.pre = np.array(-np.ones((self.Ntot, np.max(self.Npre))), dtype=np.int32)

        self.__det_pre_within_modular() # make independent modules
        if self.Nmod>1:
            self.__det_pre_inter_modulars(mu_inter) # swap E presynaptic neurons between neighboring modules

        
    def __prepare_pres(self, sigma_intra, mu_inter, sigma_inter):
        """
        prepare to determine presynaptic neurons
        this function is used in __gen_syn_connectivity__
        this is for controling # of afferent connections on each neuron
        """
        
        # prepare for intra-connections
        NeiPop=np.zeros((self.Neach.shape[0],3), dtype=np.int32)
        NeiPop[:,0], NeiPop[:,1], NeiPop[:,2] = np.sum(self.Neach[:,0:2], axis=1), np.sum(self.Neach[:,2:4], axis=1), np.sum(self.Neach, axis=1)
        Mintra = np.sqrt(NeiPop[:,NeiPop.shape[1]-1]) # total # of afferent connections
        self.Npre_intra=np.array(np.zeros((self.Ntot, 3)), dtype=np.int32) # 0: E (intra), 1: I (intra), 2: total (intra)
        self.Npre_intra[:,self.Npre_intra.shape[1]-1] = np.array(np.round(Mintra[self.assigned_modular] + sigma_intra*np.random.randn(self.Ntot)), dtype=np.int32)
        self.Npre_intra[:,0] = np.array(np.round(self.Npre_intra[:,self.Npre_intra.shape[1]-1]*NeiPop[self.assigned_modular,0]/NeiPop[self.assigned_modular,NeiPop.shape[1]-1]), dtype=np.int32) # total number of E connections per neuron
        self.Npre_intra[:,1] = self.Npre_intra[:,self.Npre_intra.shape[1]-1]-self.Npre_intra[:,0] # total number of I connections per neuron

                    
    def __det_pre_within_modular(self):
        """
        determine presynaptic neurons of each neuron within modulars
        this function is used in __gen_syn_connectivity__
        """
        
        neurons_in_mod_E_, neurons_in_mod_I_ = np.array([]), np.array([])

        for idx_mod in range(self.Nmod): # for each modular
            neurons_in_mod=np.where(self.assigned_modular==idx_mod)[0]  # identify neurons within the modular
            neurons_in_mod_E, neurons_in_mod_I=neurons_in_mod[neurons_in_mod<self.Ne], neurons_in_mod[neurons_in_mod>=self.Ne] # devide sets of E and I neurons

            for idx_neur in neurons_in_mod:
                if idx_neur<self.Ne: # avoid self-connection
                    neurons_in_mod_E_=np.delete(neurons_in_mod_E, np.where(neurons_in_mod_E==idx_neur)[0])
                    neurons_in_mod_I_=neurons_in_mod_I
                else:
                    neurons_in_mod_E_=neurons_in_mod_E
                    neurons_in_mod_I_=np.delete(neurons_in_mod_I, np.where(neurons_in_mod_I==idx_neur)[0])

                self.pre[idx_neur,:self.Npre_intra[idx_neur,self.Npre_intra.shape[1]-1]]=np.r_[np.random.choice(neurons_in_mod_E_, self.Npre_intra[idx_neur,0], replace=False),
                                                                                               np.random.choice(neurons_in_mod_I_, self.Npre_intra[idx_neur,1], replace=False)] # determine presynaptic neurons from neurons in the same modular



    def __det_pre_inter_modulars(self, mu_inter):
        """
        determine presynaptic neurons of each neuron among modulars
        this function is used in __gen_syn_connectivity__
        """

        Nswap = int(mu_inter)
        pre_mods, post_mods=self.mod_str[0], self.mod_str[1]
        neurons_in_mod_E=self.assigned_modular[:self.Ne]
        unswapped_neurons = np.array(np.ones(self.Ne), dtype=bool) # only E neurons

        for i, pre_mod in enumerate(pre_mods):
            post_mod = post_mods[i] # the crresponding post module

            if pre_mod<post_mod: # treat both directions
                # select Nswap neurons from both modules
                pre_mod_set = np.random.choice(np.where((neurons_in_mod_E==pre_mod)*(unswapped_neurons))[0], Nswap, replace=False)
                post_mod_set = np.random.choice(np.where((neurons_in_mod_E==post_mod)*(unswapped_neurons))[0], Nswap, replace=False)

                for j, pre_mod_neuron in enumerate(pre_mod_set):
                    post_mod_neuron = post_mod_set[j] # the corresponding neuron in the post module

                    # store locations of the two selected neurons
                    pre_mod_neuron_idx = np.where(self.pre==pre_mod_neuron)
                    post_mod_neuron_idx = np.where(self.pre==post_mod_neuron)

                    # swap indices
                    self.pre[pre_mod_neuron_idx] = post_mod_neuron
                    self.pre[post_mod_neuron_idx] = pre_mod_neuron

                unswapped_neurons[np.r_[pre_mod_set, post_mod_set]] = False # check the used FLAG

        self.Npre_inter = np.zeros(np.shape(self.Npre_intra), dtype=np.int32)
        post_asgn_mod = np.tile(self.assigned_modular, (np.shape(self.pre)[1], 1)).T
        pre_asgn_mod = self.assigned_modular[self.pre]

        self.Npre_intra[:,0] = np.sum((self.pre>=0)*(self.pre<self.Ne)*(post_asgn_mod==pre_asgn_mod),axis=1)          # intra E conns
        self.Npre_intra[:,1] = np.sum((self.pre>=self.Ne)*(self.pre<self.Ntot)*(post_asgn_mod==pre_asgn_mod),axis=1)  # intra I conns
        self.Npre_intra[:,2] = self.Npre_intra[:,0] + self.Npre_intra[:,1]
        self.Npre_inter[:,0] = np.sum((self.pre>=0)*(self.pre<self.Ne)*(post_asgn_mod!=pre_asgn_mod),axis=1)          # inter E conns
        self.Npre_inter[:,1] = np.sum((self.pre>=self.Ne)*(self.pre<self.Ntot)*(post_asgn_mod!=pre_asgn_mod),axis=1)  # inter I conns
        self.Npre_inter[:,2] = self.Npre_inter[:,0] + self.Npre_inter[:,1]

        self.Npre = self.Npre_intra[:,2] + self.Npre_inter[:,2]  # total afferent connections

                
    def __trans_pre_post(self):
        """
        post-pre information is translated to pre-post information for efficient data access in simulations
        """
        
        self.Npost=np.array(np.zeros(self.Ntot), dtype=np.int32)
        idx, cnt = np.unique(self.pre[self.pre>-1], return_counts=True) # counts post-synaptic neurons of each neuron
        self.Npost[idx]=cnt
        
        self.post=-np.array(np.ones((self.Ntot, self.Npost.max())), dtype=np.int32)
        for i in range(self.Ntot):
            self.post[i,:self.Npost[i]] = np.where(self.pre==i)[0]  # register post-synaptic neuron indecies

        for idx_neur, idx_mod in enumerate(self.assigned_modular):
            post_i=self.post[idx_neur,:self.Npost[idx_neur]]
            same_mod, diff_mod = post_i[self.assigned_modular[post_i]==idx_mod], post_i[self.assigned_modular[post_i]!=idx_mod]
            np.random.shuffle(same_mod), np.random.shuffle(diff_mod)
            self.post[idx_neur,:self.Npost[idx_neur]]=np.r_[same_mod, diff_mod]


        self._Npost = self.Npost
        self._post = self.post


    def __det_synaptic_strength(self, g_):
        """
        determine synaptic strength
        """

        self.g=np.zeros(self.post.shape)
        self.g_slow=np.zeros(self.post.shape)
        same_or_diff_mod=np.tile(self.assigned_modular, (self.post.shape[1],1)).T==self.assigned_modular[self.post] # same: True, other: False
        pre_neuron_type=np.zeros(self.post.shape, dtype=bool) # E: True, I: False
        pre_neuron_type[:self.Ne] = True
        post_neuron_type = self.post<self.Ne # E: True, I: False
        effective_data=self.post!=-1  # effective: True, inneffective: False

        self.g[same_or_diff_mod*pre_neuron_type*post_neuron_type*effective_data]    = g_[0, 0, 0, 0] # E->E intra
        self.g[same_or_diff_mod*pre_neuron_type*~post_neuron_type*effective_data]   = g_[0, 0, 0, 1] # E->I intra
        self.g[same_or_diff_mod*~pre_neuron_type*post_neuron_type*effective_data]   = g_[0, 0, 1, 0] # I->E intra
        self.g[same_or_diff_mod*~pre_neuron_type*~post_neuron_type*effective_data]  = g_[0, 0, 1, 1] # I->I intra
        self.g[~same_or_diff_mod*pre_neuron_type*post_neuron_type*effective_data]   = g_[0, 1, 0, 0] # E->E inter
        self.g[~same_or_diff_mod*pre_neuron_type*~post_neuron_type*effective_data]  = g_[0, 1, 0, 1] # E->I inter
        self.g[~same_or_diff_mod*~pre_neuron_type*post_neuron_type*effective_data]  = g_[0, 1, 1, 0] # I->E inter
        self.g[~same_or_diff_mod*~pre_neuron_type*~post_neuron_type*effective_data] = g_[0, 1, 1, 1] # I->I inter

        self.g_slow[same_or_diff_mod*pre_neuron_type*post_neuron_type*effective_data]    = g_[1, 0, 0, 0] # E->E intra
        self.g_slow[same_or_diff_mod*pre_neuron_type*~post_neuron_type*effective_data]   = g_[1, 0, 0, 1] # E->I intra
        self.g_slow[same_or_diff_mod*~pre_neuron_type*post_neuron_type*effective_data]   = g_[1, 0, 1, 0] # I->E intra
        self.g_slow[same_or_diff_mod*~pre_neuron_type*~post_neuron_type*effective_data]  = g_[1, 0, 1, 1] # I->I intra
        self.g_slow[~same_or_diff_mod*pre_neuron_type*post_neuron_type*effective_data]   = g_[1, 1, 0, 0] # E->E inter
        self.g_slow[~same_or_diff_mod*pre_neuron_type*~post_neuron_type*effective_data]  = g_[1, 1, 0, 1] # E->I inter
        self.g_slow[~same_or_diff_mod*~pre_neuron_type*post_neuron_type*effective_data]  = g_[1, 1, 1, 0] # I->E inter
        self.g_slow[~same_or_diff_mod*~pre_neuron_type*~post_neuron_type*effective_data] = g_[1, 1, 1, 1] # I->I inter


        self._g      = self.g
        self._g_slow = self.g_slow

            
    def __det_conduction_delays(self, dt, D_):
        """
        determine conduction delays
        """
        
        Ds_=np.array(np.round(D_/dt), dtype=np.int32)                                        # translate time to steps up to dt
        Ds_[:,:,0] -= 1
        self.Ds_min, self.Ds_max = np.min(Ds_), np.max(Ds_)                                  # min & max steps of conduction delays 
        delays_length_each = np.array(np.zeros((Ds_.shape[0], Ds_.shape[1], self.Ntot, self.Ds_max)), dtype=np.int32)

        neuron_type=np.zeros(self.post.shape, dtype=bool) # E: True, I: False
        neuron_type[:self.Ne] = True
        same_or_diff_mod=np.tile(self.assigned_modular, (self.post.shape[1],1)).T==self.assigned_modular[self.post] # same: True, other: False
        effective_data=self.post!=-1  # effective: True, inneffective: False
        conn_type=np.array(-np.zeros(list(Ds_.shape[:2]) + list(self.post.shape)), dtype=bool)
        conn_type[0,0] = neuron_type*same_or_diff_mod*effective_data
        conn_type[0,1] = ~neuron_type*same_or_diff_mod*effective_data
        conn_type[1,0] = neuron_type*~same_or_diff_mod*effective_data
        conn_type[1,1] = ~neuron_type*~same_or_diff_mod*effective_data

        for mod_flag, ei_flag in product(range(Ds_.shape[0]), range(Ds_.shape[1])): # determine delays_length
            num=np.sum(conn_type[mod_flag,ei_flag], axis=1)
            Ds_min, Ds_max = Ds_[mod_flag,ei_flag,0], Ds_[mod_flag,ei_flag,1]
            tmp=np.array(num/(Ds_max-Ds_min), dtype=np.int32)
            num_=num-tmp*(Ds_max-Ds_min)
            delays_length_each[mod_flag,ei_flag,:,Ds_min:Ds_max] += np.tile(np.array([tmp]).T, (1, Ds_max-Ds_min))
            a=np.arange(Ds_min, Ds_max)
            for i in range(self.Ntot):
                delays_length_each[mod_flag,ei_flag,i,np.random.choice(a, num_[i], replace=False)]+=1

        self.delays_length = np.array(np.sum(np.sum(delays_length_each, axis=0), axis=0), dtype=np.int32) # store total number of connections
        self.delays = -np.array(np.ones((self.Ntot, self.Ds_max, np.max(self.delays_length))), dtype=np.int32)
        delays_length_each_cum=np.cumsum(delays_length_each, axis=3)

        conn_type_ = np.where(conn_type)

        for i in range(self.Ntot):
            mod_flag, ei_flag = 0, 0
            e_intra=np.split(conn_type_[3][(conn_type_[0]==mod_flag)*(conn_type_[1]==ei_flag)*(conn_type_[2]==i)], delays_length_each_cum[mod_flag,ei_flag,i])
            mod_flag, ei_flag = 0, 1
            i_intra=np.split(conn_type_[3][(conn_type_[0]==mod_flag)*(conn_type_[1]==ei_flag)*(conn_type_[2]==i)], delays_length_each_cum[mod_flag,ei_flag,i])
            mod_flag, ei_flag = 1, 0
            e_inter=np.split(conn_type_[3][(conn_type_[0]==mod_flag)*(conn_type_[1]==ei_flag)*(conn_type_[2]==i)], delays_length_each_cum[mod_flag,ei_flag,i])
            mod_flag, ei_flag = 1, 1
            i_inter=np.split(conn_type_[3][(conn_type_[0]==mod_flag)*(conn_type_[1]==ei_flag)*(conn_type_[2]==i)], delays_length_each_cum[mod_flag,ei_flag,i])

            for d in range(self.Ds_min, self.Ds_max):
                self.delays[i,d,:self.delays_length[i,d]] = np.r_[e_intra[d], i_intra[d], e_inter[d], i_inter[d]]


        self._delays_length = self.delays_length
        self._delays = self.delays


    @cython.cfunc
    def store_firing_info(self, s: cython.int, firings_vec: cython.int[:]) -> cython.bint:
        i: cython.int
        flag: cython.bint = True

        for i in range(self.Ntot):
            if firings_vec[i]==1:
                self._firings[0, self.N_firings] = s
                self._firings[1, self.N_firings] = i
                self.N_firings += 1

                if (self.N_firings>=self.N_firings_max):
                    self.N_firings = 1 # ignoring all
                    flag = False

            if flag==False:
                break

        return flag
            

    @cython.cfunc
    def spike_interaction(self, s: cython.int, g_: cython.double[:,:], g_slow_: cython.double[:,:], _x: cython.double[:,:], _u: cython.double[:,:], _U: cython.double[:,:], _tau_f: cython.double[:,:], _tau_d: cython.double[:,:], _Dt: cython.double[:,:]) -> cython.void:
        i: cython.int
        j: cython.int
        k: cython.int

        k = self.N_firings-1
        while(s-self._firings[0,k]<self.Ds_min):
            k -= 1
        while(self.Ds_min<=s-self._firings[0,k]<self.Ds_max):
            for j in range(self._delays_length[self._firings[1,k], s-self._firings[0,k]]):
                i=self._post[self._firings[1,k], self._delays[self._firings[1,k],s-self._firings[0,k],j]]
                _x[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]] = 1.0 - (1.0 - (1.0 - _u[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]])*_x[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]])*exp(-_Dt[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]]*_tau_d[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]]) # update variable x
                _u[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]] = _U[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]] + (1.0 - _U[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]])*_u[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]]*exp(-_Dt[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]]*_tau_f[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]]) # update variable u
                g_[i,self._neuron_type[self._firings[1,k]]] += self._g[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]]*_x[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]]*_u[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]] # spike interaction
                g_slow_[i,self._neuron_type[self._firings[1,k]]] += self._g_slow[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]]*_x[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]]*_u[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]] # spike interaction
                _Dt[self._firings[1,k],self._delays[self._firings[1,k],s-self._firings[0,k],j]] = .0 # reset synaptic firing interval
            k -= 1


    @cython.cfunc
    def ready_for_next_sec(self) -> cython.void:
        i: cython.int
        k: cython.int

        k = self.N_firings-1
        while(self.S-self._firings[0,k]<self.Ds_max):
            k -= 1

        for i in range(1, self.N_firings-k):
            self._firings[0,i] = self._firings[0,i+k]-self.S
            self._firings[1,i] = self._firings[1,i+k]

        self.N_firings = self.N_firings - k


            
    @cython.cfunc
    def output_spk_data(self, t: cython.int, parms: dict) -> cython.void:
        # NOTE: gk and Cstep are degenerate for the adaptation current
        # (Ica = gk*c*(Ek-V), c += Cstep per spike -> Ica ~ gk*Cstep*spikes), so only
        # Gk (depth) and Tca (Ca time constant) are in the filename -- these two are the
        # independent adaptation knobs; sweep gk & tca, not Cstep.  Keeps <255 bytes.
        # coupling g's kept at %.5e (primary identity, need precision); plasticity/neuron
        # params (U, tau_d, tau_f, gk, tca) at %.2e (3 sig figs, enough for all sweeps) to
        # stay under the 255-byte filename limit.
        
        fname: str = 'spkN%dMcol%dMrow%dGee%.5eGei%.5eGie%.5eGii%.5eGinter%.5eSlowGee%.5eGei%.5eGie%.5eGii%.5eGinter%.5eKinter%dUee%.2eTaudee%.2eTaufee%.2eGk%.2eTca%.2eSeed%04d.dat' % (self.Ntot, parms['mod'][0], parms['mod'][1], parms['g'][0,0,0,0], parms['g'][0,0,0,1], parms['g'][0,0,1,0], parms['g'][0,0,1,1], parms['g'][0,1,0,0], parms['g'][1,0,0,0], parms['g'][1,0,0,1], parms['g'][1,0,1,0], parms['g'][1,0,1,1], parms['g'][1,1,0,0], parms['mu_inter'], parms['U'][0,0], parms['tau_d'][0,0], parms['tau_f'][0,0], parms['gk'], parms['tca'], parms['seed'])
        i: cython.int
                        
        with open(fname, 'a') as o:
            for k in range(1, self.N_firings):
                if self._firings[0,k]>=0:
                    print('%.4f %d %d %d' % (t+0.001*self._firings[0,k]*self.dt, self._firings[1,k], self._mod_base_idx[self._firings[1,k]], self._assigned_modular[self._firings[1,k]]), file=o)

    @cython.cfunc
    def output_net_data(self, parms: dict) -> cython.void:
        fname: str = 'netN%dMcol%dMrow%dKinter%dUee%.2eTaudee%.2eTaufee%.2eGk%.2eTca%.2eSeed%04d.dat' % (self.Ntot, parms['mod'][0], parms['mod'][1], parms['mu_inter'], parms['U'][0,0], parms['tau_d'][0,0], parms['tau_f'][0,0], parms['gk'], parms['tca'], parms['seed'])
        i: cython.int
        j: cython.int

        with open(fname, 'a') as o:
            for i in range(self.Ntot):
                for j in range(self._Npost[i]):
                    print('%d %d %d %d %d %d' % (i, self._post[i,j], self._mod_base_idx[i], self._mod_base_idx[self._post[i,j]], self._assigned_modular[i], self._assigned_modular[self._post[i,j],]), file=o)


#short-term plasticity
@cython.cclass
class STP:
    """
    File name: STP.py
    All right reserved by H. Kato (Oita Univ. Japan)

    Definition of standard STP model

    [References]
    M.V. Tsodyks and H. Markram, “The neural code between neocortical pyramidal neurons de-
pends on neurotransmitter release probability,” Proceedings of the National Academy ofSciences
of the United States of America, vol. 94, no. 2, pp. 719–723, 1997.
    """

    # __init__()
    dt: cython.double
    S: cython.int
    N: np.ndarray
    Ntot: cython.int
    Ne: cython.int
    Ni: cython.int
    x: np.ndarray
    u: np.ndarray
    U: np.ndarray
    tau_f: np.ndarray
    tau_d: np.ndarray
    Dt: np.ndarray
    _x: cython.double[:,:]
    _u: cython.double[:,:]
    _U: cython.double[:,:]
    _tau_f: cython.double[:,:]
    _tau_d: cython.double[:,:]
    _Dt: cython.double[:,:]

    def __init__(self, dt: cython.double, N: np.ndarray, Ntot: cython.int, Ne: cython.int, Ni: cython.int, post: np.ndarray, g:np.ndarray, g_slow:np.ndarray, U=np.array([[0.21,0.3],[0.14, 0.3]]), tau_f=np.array([[5.0, 5.0],[2.0, 2.0]]), tau_d=np.array([[463.0, 227.0],[875.0, 400.0]])):
        """
        All variables and paraeters in the STP model are initialized here.
        """

        self.dt     = dt
        self.S      = int(1000*round(1.0/self.dt))
        self.N      = N
        self.Ntot   = Ntot
        self.Ne     = Ne
        self.Ni     = Ni

        # print(self.dt, self.S, self.N, self.Ntot, self.N)

        self.u, self.x, self.U = -np.ones(post.shape), -np.ones(post.shape), -np.ones(post.shape)
        self.tau_f, self.tau_d = -np.ones(post.shape), -np.ones(post.shape)
        self.u[post>=0] = 0.0 # initialize using elements
        self.x[post>=0] = 1.0 # initialize using elements
        i_pre, i_post = np.where(post>=0)

        flag_i = (i_pre<self.Ne)*(i_post<self.Ne)
        self.tau_f[(i_pre[flag_i], i_post[flag_i])] = tau_f[0,0]
        flag_i = (i_pre<self.Ne)*(i_post>=self.Ne)
        self.tau_f[(i_pre[flag_i], i_post[flag_i])] = tau_f[0,1]
        flag_i = (i_pre>=self.Ne)*(i_post<self.Ne)
        self.tau_f[(i_pre[flag_i], i_post[flag_i])] = tau_f[1,0]
        flag_i = (i_pre>=self.Ne)*(i_post>=self.Ne)
        self.tau_f[(i_pre[flag_i], i_post[flag_i])] = tau_f[1,1]

        flag_i = (i_pre<self.Ne)*(i_post<self.Ne)
        self.tau_d[(i_pre[flag_i], i_post[flag_i])] = tau_d[0,0]
        flag_i = (i_pre<self.Ne)*(i_post>=self.Ne)
        self.tau_d[(i_pre[flag_i], i_post[flag_i])] = tau_d[0,1]
        flag_i = (i_pre>=self.Ne)*(i_post<self.Ne)
        self.tau_d[(i_pre[flag_i], i_post[flag_i])] = tau_d[1,0]
        flag_i = (i_pre>=self.Ne)*(i_post>=self.Ne)
        self.tau_d[(i_pre[flag_i], i_post[flag_i])] = tau_d[1,1]

        flag_i = (i_pre<self.Ne)*(i_post<self.Ne)
        self.U[(i_pre[flag_i], i_post[flag_i])] = U[0,0]
        flag_i = (i_pre<self.Ne)*(i_post>=self.Ne)
        self.U[(i_pre[flag_i], i_post[flag_i])] = U[0,1]
        flag_i = (i_pre>=self.Ne)*(i_post<self.Ne)
        self.U[(i_pre[flag_i], i_post[flag_i])] = U[1,0]
        flag_i = (i_pre>=self.Ne)*(i_post>=self.Ne)
        self.U[(i_pre[flag_i], i_post[flag_i])] = U[1,1]
        
        # for efficient computing
        self.tau_f[self.tau_f>0] = 1.0/self.tau_f[self.tau_f>0]
        self.tau_d[self.tau_d>0] = 1.0/self.tau_d[self.tau_d>0]

        self.Dt = np.zeros(post.shape)

        g[self.U>0] /= self.U[self.U>0]
        g_slow[self.U>0] /= self.U[self.U>0]

        # print(self.tau_f, self.tau_d, self.U)
        # print(g)
        

        self._tau_f = self.tau_f
        self._tau_d = self.tau_d
        self._x = self.x
        self._u = self.u
        self._U = self.U
        self._Dt = self.Dt

    @cython.cfunc
    def update(self, Npost: cython.int[:]) -> cython.void:
        i: cython.int
        j: cython.int

        for i in range(self.Ntot):
            for j in range(Npost[i]):
                self._Dt[i,j] += self.dt

    @cython.cfunc
    def get_x(self) -> cython.double[:,:]:
        return self._x
        
    @cython.cfunc
    def get_u(self) -> cython.double[:,:]:
        return self._u
        
    @cython.cfunc
    def get_U(self) -> cython.double[:,:]:
        return self._U
        
    @cython.cfunc
    def get_tau_d(self) -> cython.double[:,:]:
        return self._tau_d
        
    @cython.cfunc
    def get_tau_f(self) -> cython.double[:,:]:
        return self._tau_f
        
    @cython.cfunc
    def get_Dt(self) -> cython.double[:,:]:
        return self._Dt
        
        
def simu():
    parms: dict
    parms = get_opts() # get options

    if 'seed' in parms.keys():
        np.random.seed(seed=parms['seed'])

    # basic simulation paramters
    t: cython.int
    T: cython.int = parms['T']
    s: cython.int
    S: cython.int = int(1000*round(1.0/parms['dt']))
    i: cython.int

    spk_flag: cython.bint = parms['spk']
    idx_flag: cython.bint = parms['idx_flag']
    if idx_flag:
        idx: cython.int = parms['idx']

    neurons = frenchLIF(N=parms['N'], dt=parms['dt'], gk=parms['gk'], Cstep=parms['Cstep'], tca=parms['tca'])
    synapses = ExpModel(N=parms['N'], dt=parms['dt'], tau=parms['Tsyn'])            # AMPA & GABA A
    synapses_slow = ExpModel(N=parms['N'], dt=parms['dt'], tau=parms['Tsyn_slow'])  # NMDA & GABA B
    syn_currents = Current(N=parms['N'], Vrev=parms['Vrev'])                        # AMPA & GABA A
    syn_currents_slow = Current(N=parms['N'], Vrev=parms['Vrev'], flag_slow=True)   # NMDA & BABA B
    background_inputs = OU_like_process(N=parms['N'], dt=parms['dt'], tau_syn=parms['Tsyn'], g0=parms['g0'], sigma=parms['sigma'])
    bg_currents = Current(N=parms['N'], Vrev=parms['Vrev'])
    mod_net = ModularNetwork(N=parms['N'], dt=parms['dt'], mod_shape=parms['mod'], mu_inter=parms['mu_inter'], sigma_inter=parms['sig_inter'], g_=parms['g'], D_=parms['D'])
    stp = STP(dt=mod_net.dt, N=mod_net.N, Ntot=mod_net.Ntot, Ne=mod_net.Ne, Ni=mod_net.Ni, post=mod_net.post, g=mod_net.g, g_slow=mod_net.g_slow, U=parms['U'], tau_f=parms['tau_f'], tau_d=parms['tau_d'])

    # for stable parameter search
    for t in range(1):
        for s in range(S):
            neurons.det_firings()                                                     # did neurons fire?
            mod_net.store_firing_info(s, neurons.get_firings_vec())                   # store firing information
            bg_currents.update(neurons.get_V(), background_inputs.get_g())            # get background input currents
            syn_currents.update(neurons.get_V(), synapses.get_g())                    # get synaptic input currents
            syn_currents_slow.update(neurons.get_V(), synapses_slow.get_g())          # get slow synaptic input currents

            # differential equations
            neurons.update(syn_currents.get_I(), syn_currents_slow.get_I(), bg_currents.get_I())
            synapses.update()
            synapses_slow.update()
            background_inputs.update() 

        background_inputs.ready_for_next_sec()
        mod_net.ready_for_next_sec()

    
    #start simulation
    for t in range(T):
        for s in range(S):
            if idx_flag:
                neurons.display_neuronal_internal_states(idx, t, s, syn_currents.get_I2(), syn_currents.get_I(), syn_currents_slow.get_I2(), syn_currents_slow.get_I(), bg_currents.get_I2(), bg_currents.get_I())
            
            neurons.det_firings() # did neurons fire?
            mod_net.store_firing_info(s, neurons.get_firings_vec()) # store firing information

            bg_currents.update(neurons.get_V(), background_inputs.get_g()) # get background input currents
            mod_net.spike_interaction(s, synapses.get_g(), synapses_slow.get_g(), stp.get_x(), stp.get_u(), stp.get_U(), stp.get_tau_f(), stp.get_tau_d(), stp.get_Dt()) # spike interaction
            syn_currents.update(neurons.get_V(), synapses.get_g()) # get synaptic input currents
            syn_currents_slow.update(neurons.get_V(), synapses_slow.get_g()) # get slow synaptic input currents


            # differential equations
            neurons.update(syn_currents.get_I(), syn_currents_slow.get_I(), bg_currents.get_I())
            synapses.update()
            synapses_slow.update()
            background_inputs.update()
            stp.update(mod_net.Npost)


        if spk_flag:
            mod_net.output_spk_data(t, parms)

        background_inputs.ready_for_next_sec()
        mod_net.ready_for_next_sec()

# EOF
