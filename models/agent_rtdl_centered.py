# Copyright (c) 2023-2025 Katori Lab. All Rights Reserved
# 強化学習エージェント: レザバーTD学習(RTDL) 連続状態空間・離散行動空間タスク用

import numpy as np
import matplotlib.pyplot as plt
from .matrix_generator import *
import common_configurator as common

class Agent:
    def __init__(self,c):
        self.c = c
        self.Nx = c.Nx
        self.Ny = c.Ny
        self.Nu = c.Nu
        self.reservoir = common.generate_instance(c,module=c.rc_module,class_=c.rc_class)#リザバーのインスタンスを生成

    def eta0(self):
        # 学習率：エピソードの経過 ・平均報酬に応じて変化.
        # eta_initからeta_finalにむけてtau_etaの時定数で指数減衰する
        eta = self.c.eta_final + (self.c.eta_init-self.c.eta_final) * np.exp(-self.episode/self.c.eta_tau -self.mean_reward/self.c.eta_decay)
        return eta
    
    def eta1(self):
        # 学習率：エピソードの経過に応じて減衰させる. 
        # eta_initからeta_finalにむけてtau_etaの時定数で指数減衰する
        eta = self.c.eta_final + (self.c.eta_init-self.c.eta_final) * np.exp(-self.episode/self.c.eta_tau)
        return eta
    
    def eta2(self):
        # 学習率：平均報酬に応じて変化. 
        eta = self.c.eta_final + (self.c.eta_init-self.c.eta_final) * np.exp(-self.mean_reward/self.c.eta_decay)
        return eta

    def epsilon0(self):
        # epsilon（ランダム行動をとる確率）：エピソードの経過・平均報酬に応じて変化.
        eps = self.c.eps_final + (self.c.eps_init-self.c.eps_final) * np.exp(-self.episode/self.c.eps_tau -self.mean_reward/self.c.eps_decay)
        return eps
    
    def epsilon1(self):
        # epsilon（ランダム行動をとる確率）：エピソードの経過に応じて減衰.
        eps = self.c.eps_final + (self.c.eps_init-self.c.eps_final) * np.exp(-self.episode/self.c.eps_tau)
        return eps

    def epsilon2(self):
        # epsilon（ランダム行動をとる確率）：平均報酬に応じて変化.
        eps = self.c.eps_final + (self.c.eps_init-self.c.eps_final) * np.exp(-self.mean_reward/self.c.eps_decay)
        return eps

    def initialize(self):
        self.n = 0 #　
        self.t = 0 # 
        self.Wi = generate_random_matrix(self.Nx, self.Nu, self.c.alpha_i, self.c.beta_i, distribution="one", normalization="none")
        self.Wb = generate_random_matrix(self.Nx, self.Ny, self.c.alpha_b, self.c.beta_b, distribution="one", normalization="none")
        self.Wo = np.zeros((self.Ny,self.Nx))
        self.list_reward = []
        self.mean_reward = 0

    def reset(self,episode):
        #各エピソードの始めに行う初期化
        self.reservoir.reset()
        self.n = 0 #time step
        self.episode = episode 
        self.s = np.zeros(self.Ny)
        self.q = np.zeros(self.Ny)
        self.a = np.random.randint(0,3)
        self.sum_reward = 0 
        if episode>20:
            self.mean_reward = sum(self.list_reward[episode-19:])/20
        self.X = np.zeros((1,self.Nx))
        self.R = np.zeros((1,self.Nx))
        self.Q = np.zeros((1,self.Ny))
        self.U = np.zeros((1,self.Nu))

    def get_action(self, state, reward, done):
        # action = self.step(state,reward)
        action = self.step(state, reward, done=done)
        self.sum_reward += reward
        #print("n:",self.n)
        if done:
            self.list_reward.append(self.sum_reward)
            #print("done n:",self.n)
            self.reservoir.done()
            
        return action

    # def step(self, u, reward):
    def step(self, u, reward, done=False):
        sum = np.zeros(self.Nx)
        sum += self.Wi @ u
        self.reservoir.step(sum) # リザバーの状態更新
        # E-neuron calcium features, population-mean centered
        features = self.reservoir.r.copy()
        features[:480] -= np.mean(features[:480])

        q_next = self.Wo @ features
        a_next = np.argmax(q_next) # 行動の決定（qが最大の行動を選択）

        target = reward

        if not done:
            target += self.c.gamma * q_next[a_next]

        td_error = target - self.q[self.a]

        ### training output conection
        Wo_next = self.Wo
        # Wo_next[self.a] = self.Wo[self.a] + self.eta2()*np.tanh(reward +self.c.gamma*q_next[a_next]-self.q[self.a])*self.reservoir.r#
        Wo_next[self.a] = (
            self.Wo[self.a]
            + self.eta2()
            * np.tanh(td_error)
            * features
        )

        ### epsilon greedy: 確率epsilonでランダム行動を選択
        if self.c.eps_greedy == True:
            epsilon = self.epsilon2()
            if epsilon > np.random.uniform(0, 1): a_next = np.random.choice(self.Ny) 

        ### Update
        self.q, self.Wo, self.a = q_next, Wo_next, a_next
        self.n += 1
        self.t += 1

        ### Record
        #self.X = np.append(self.X,self.reservoir.x.reshape(1,self.Nx), axis=0)
        self.R = np.append(self.R,self.reservoir.r.reshape(1,self.Nx), axis=0)
        self.Q = np.append(self.Q,self.q.reshape(1,self.Ny), axis=0)
        self.U = np.append(self.U,u.reshape(1,self.Nu), axis=0)

        return self.a
    