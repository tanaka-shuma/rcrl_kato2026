# Copyright (c) 2021-2023 Katori lab. All Rights Reserved

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from .matrix_generator import *
#import os
#os.environ["OMP_NUM_THREADS"] = "4"
#os.environ["OPENBLAS_NUM_THREADS"] = "4"

class LeakyIntegrator:
    def __init__(self,c=None):
        if c!= None: 
            self.c=c
            self.initialize(c)

    def initialize(self,c=None):
        if c!= None: self.c=c
        self.Nx = c.Nx
        self.Ny = c.Ny
        self.Nu = c.Nu
        self.tau_x = c.tau_x
        self.ds = 1.0
        self.Wr = generate_random_matrix(self.Nx, self.Nx, self.c.alpha_r, self.c.beta_r, distribution="one", normalization="sr")
        
        self.n = 0 
        self.x = np.zeros(self.Nx)
        self.r = np.zeros(self.Nx)

    def reset(self):
        self.x = np.zeros(self.Nx)

    def step(self,input):
        sum = self.Wr @ self.r + input
        self.x = self.x + (-self.x + sum)*self.ds / self.tau_x
        self.r = np.tanh(self.x)
        
    def done(self):
        pass