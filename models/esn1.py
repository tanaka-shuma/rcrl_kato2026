# Copyright (c) 2021-2023 Katori lab. All Rights Reserved
# 基本的なESN (フィードバックなし)、クラスで実装
# TODO フィードバック
import numpy as np
from models.matrix_generator import *
from models.utilities import *

class ESN:
    def __init__(self,c=None):
        if c!= None: 
            self.c=c
            self.initialize(c)
        
    def initialize(self,c=None):
        if c!= None: self.c=c
        if self.c.seed != None: np.random.seed(int(self.c.seed))

        self.is_feedback = False
        self.is_static_node = True

        self.Wr = generate_random_matrix(self.c.Nx, self.c.Nx, self.c.alpha_r, self.c.beta_r, distribution="uniform", normalization="sr")
        self.Wi = generate_random_matrix(self.c.Nx, self.c.Nu, self.c.alpha_i, self.c.beta_i, distribution="one", normalization="none")
        self.Wb = generate_random_matrix(self.c.Nx, self.c.Ny, self.c.alpha_b, self.c.beta_b, distribution="one", normalization="none")
        
        if self.is_static_node:
            self.Wo = np.zeros((self.c.Ny, self.c.Nx+1))
        else:
            self.Wo = np.zeros((self.c.Ny, self.c.Nx))
    
    def run(self, input, output=np.empty(0),NTinput=None, n_reset=None):
        """
        NTinput:入力により駆動される区間. 通常は全期間で入力が駆動. 時系列予測の場合に切り替え 
        """
        NT = len(input) # num of time steps
        if NTinput == None: NTinput = NT
        
        self.X = np.zeros((NT, self.c.Nx))
        self.Y = np.zeros((NT, self.c.Ny))
        
        x = np.zeros(self.c.Nx) # TODO ランダムな初期値
        y = np.zeros(self.c.Ny)

        for n in range(NT):
            if n<NTinput:
                u = input[n, :]# input driven
            else:
                u = self.Y[n-1,:] # output driven (時系列予測の予測期間)
            
            if self.is_feedback:
                x = np.tanh(self.Wi @ u + self.Wr @ x + self.Wb @ y)
            else:
                x = np.tanh(self.Wi @ u + self.Wr @ x)

            if self.is_static_node:
                x1 = np.concatenate([x, [1.0]])
                y = self.Wo @ x1
            else:
                y = self.Wo @ x
                
            self.X[n,:] = x
            self.Y[n,:] = y
        
        return self.Y
    
    def fit(self, input, output, is_run=True):
        if is_run:
            assert len(input)==len(output)
            self.run(input)

        assert len(self.X)==len(output)

        if self.is_static_node:
            ones = np.ones((self.X.shape[0], 1))
            X1 = np.hstack((self.X,ones))
        else:
            X1 = self.X

        if self.c.lambda_ridge != None:
            lambda_ridge = self.c.lambda_ridge
        if hasattr(self.c,"log10_lambda_ridge"):
            lambda_ridge = np.power(10,float(self.c.log10_lambda_ridge))

        self.Wo = compute_ridge_regression(X1[self.c.NTtrans:], output[self.c.NTtrans:],lambda_ridge)
        self.Y = X1 @ self.Wo.T
        return self.Y
    
        