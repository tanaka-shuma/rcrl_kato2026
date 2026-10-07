# Copyright (c) 2023 Katori lab. All Rights Reserved
# leaky integrator: 物理レザバーの実装例として、連続時間モデルを実装する。
# TODO 時間遅れ
import numpy as np
from models.matrix_generator import *
from models.utilities import *
from .labbench import StimulatorPulse,ProbeAverage,DelayLine
#from tqdm import tqdm

class PhysicalReservoir:
    def __init__(self,c=None):
        if c!= None: 
            self.c=c
            self.initialize(c)
        
    def initialize(self,c=None):
        if c!= None: self.c=c

        # 外部系の初期化
        self.is_feedback = False
        self.is_static_node = True
        self.Wi = generate_random_matrix(self.c.Nx, self.c.Nu, self.c.alpha_i, self.c.beta_i, distribution="one", normalization="none")
        self.Wb = generate_random_matrix(self.c.Nx, self.c.Ny, self.c.alpha_b, self.c.beta_b, distribution="one", normalization="none")
        
        if self.is_static_node:
            self.Wo = np.zeros((self.c.Ny, self.c.Nx+1))
        else:
            self.Wo = np.zeros((self.c.Ny, self.c.Nx))

        ### 物理系の初期化
        self.ds = self.c.ds # 外部系時間ステップ
        self.dt = self.c.dt # 物理系時間ステップ
        self.delay = self.c.delay
        self.sampling_steps = int(self.ds/self.dt) # サンプリング間隔(時間ステップ数)

        self.tau = self.c.tau
        self.Wr = generate_random_matrix(self.c.Nx, self.c.Nx, self.c.alpha_r, self.c.beta_r, distribution="uniform", normalization="sr")
    
        self.delay_line = DelayLine(depth=int(self.delay/self.dt), N=self.c.Nx)
        self.stimulator = StimulatorPulse(dt=self.c.dt,pulse_width=1.0,gain=1.0,N=self.c.Nx)
        self.probe = ProbeAverage(dt=self.dt,gain=1.0,N=self.c.Nx)
    
    def step_physics(self,input):
        # 物理系のODEをEular差分方程式で記述
        self.v = self.v + (-self.v + np.tanh(self.Wr @ self.v )+input)*self.dt/self.tau
        self.r = np.tanh(self.v)

    def step(self,input):
        # 外部系の単位時間ステップ
        self.stimulator.set(self.t,input)
        self.probe.set(self.t)
        for _ in range(self.sampling_steps):
            stim0 = self.stimulator.get(self.t)
            stim = self.delay_line(stim0)
            self.step_physics(stim)
            self.probe.measure(self.v)
            self.t += self.dt
        
        self.x = self.probe.get()
    
    def reset(self):
        # 物理系のリセット
        self.v = np.zeros(self.c.Nx)
        self.t = 0

    def run(self, input, output=np.empty(0),NTinput=None, n_reset=None):
        """
        NTinput:入力により駆動される区間. 通常は全期間で入力が駆動. 時系列予測の場合に切り替え 
        (時刻カウンタ、時間間隔、長さ)
        n,NT,ds 外部系の時刻カウンタ、時間ステップ数、時間間隔
        t,nt,dt 物理系の時刻カウンタ、時間ステップ数、時間間隔
        """

        ### 外部系の初期化
        NT = len(input)                       # 外部系の時間ステップ数(Number of time steps) 
        if NTinput == None: NTinput = NT
        self.X = np.zeros((NT, self.c.Nx))
        self.Y = np.zeros((NT, self.c.Ny))
        self.x = np.zeros(self.c.Nx) 
        self.y = np.zeros(self.c.Ny)

        ### 物理系の初期化
        nt = NT * self.sampling_steps  # 物理系シミュレーションの時間（ステップ数）
        time = np.arange(nt) * self.dt # 物理系シミュレーションの時間（描画用時間ベクトル）
        self.reset()

        for n in range(NT):
            u = input[n, :]
            sum = self.Wi @ u
            
            self.step(sum)

            if self.is_static_node:
                x1 = np.concatenate([self.x, [1.0]])
                self.y = self.Wo @ x1
            else:
                self.y = self.Wo @ self.x
            
            self.X[n,:] = self.x
            self.Y[n,:] = self.y
        
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
    
        