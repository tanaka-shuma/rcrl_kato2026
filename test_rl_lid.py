# Copyright (c) 2018-2025 Katori lab. All Rights Reserved
# 強化学習(ninerooms)+LIDモデルのテスト

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import copy

from explorer import common
from explorer import gridsearch as gs
from explorer import visualization as vs
from explorer import randomsearch as rs
from explorer import optimization as opt

### 共通設定
common.load_config("config.config_rl_ninerooms_rtdl_lid")# 設定ファイルの読込
common.prefix  = "data%s_rl_lid" % common.string_now() # 実験名（ファイルの接頭辞）
common.dir_path= "data/"+common.prefix # 実験データの出力先パス
common.exe     = "python main_rl.py " # 実行されるプログラム
common.columns =['seed','id','alpha_i', 'alpha_b', 'beta_i', 'beta_b', 'gamma', 'eta_init', 'eta_final', 'eta_tau', 'eta_decay', 'eps_init', 'eps_final', 'eps_tau', 'eps_decay', 'Nx', 'alpha_r', 'beta_r', 'tau_x','total_reward','mean_reward']
common.parallel= 14 # 並列実行の数
common.setup()
common.report_common()
common.report_config()

### 単体実行
def exe3():# 
    cnf=copy.copy(common.config)
    cnf.seed=2
    gs.execute(config=cnf)
#exe3()

### ランダムサーチと散布図行列
def rs1():
    rs.clear()
    rs.append("alpha_r",min=0,max=1.2)
    rs.append("alpha_i",min=0,max=1)
    rs.random(num=200,samples=2)
    df = common.load_dataframe() # 直前に保存されたcsvファイルをデータフレーム(df)に読み込む
    df = df[['alpha_r','alpha_i','NMSEtrain','NMSEtest']] # 指定した列のみでデータフレームを構成する
    #df = df[(df['y1']<=10.0)] # 条件を満たすデータについてデータフレームを構成する。
    #print(df)
    scatter_matrix(df, alpha=0.8, figsize=(6, 6), diagonal='kde') # 散布図行列
    vs.savefig()
#rs1()

### 最適化
def func(row):# 関数funcでtargetを指定する。
    return row['y1'] + 0.3*row['y2']

def optimize():
    opt.clear()#設定をクリアする
    # 変数の追加([変数名],[基本値],[下端],[上端],[まるめの桁数])
    #opt.append("beta_r",value=0.01,min=0.,max=1,round=3)
    #opt.append("beta_i",value=0.01,min=0.,max=1,round=3)
    #opt.append("alpha_i",value=0.5,min=0.,max=1,round=2)
    #opt.append("alpha_b",value=0.5,min=0.,max=1,round=2)
    #opt.append("alpha_r",value=0.8,min=0.,max=2,round=2)
    #opt.append("tau_x",value=2.0,min=1,max=10,round=2)
    #opt.append("gamma",value=0.9,min=0.8,max=1,round=3)
    opt.append("eta_init" ,value=0.01,min=0.,max=0.1,round=3)
    opt.append("eta_final",value=0.0,min=0.,max=0.1,round=4)
    opt.append("eta_tau"  ,value=10000,min=0.,max=10000,round=0)
    opt.append("eta_decay",value=4,min=0.,max=100,round=1)

    opt.append("eps_init" ,value=0.1,min=0.,max=0.2,round=3)
    opt.append("eps_final",value=0.02,min=0.,max=0.1,round=4)
    opt.append("eps_tau"  ,value=1000,min=0.,max=10000,round=0)
    opt.append("eps_decay",value=10,min=0.,max=100,round=1)

    opt.maximize(target="total_reward",iteration=10,population=21,samples=5)
    #opt.minimize(TARGET=func,iteration=5,population=10,samples=4)
    common.config = opt.best_config # 最適化で得られた設定を基本設定とする
optimize()

### グリッドサーチ
def gridsearch(X1,min=0,max=1,num=41,samples=10):
    # 変数(X1)についてグリッドサーチ(GS)を行い、評価基準の変化をまとめてプロット
    gs.scan1ds(X1,min=min,max=max,num=num,samples=samples)# GSの実行
    df = common.load_dataframe()# 直前のGSの結果をデータフレームに読込
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6,4))
    vs.vline(X1)
    vs.plot_stat_summary(df,X=X1,Y="total_reward",label="total reward",color=cmap(1))# 平均、標準偏差、最大値、最小値をプロット
    vs.plot_stat_summary(df,X=X1,Y="mean_reward",label="mean reward",color=cmap(2))
    plt.grid(linestyle="dotted")
    plt.ylabel("Reward")
    #plt.yscale('log')
    #plt.ylim([0,1]) # y軸の範囲
    plt.legend()
    plt.xlabel(X1)
    vs.plt_output()

def gs1():
    ns=5
    #gridsearch("Nx",min=100,max=1000,num=41,samples=ns)
    gridsearch("alpha_i",min=0.,max=1.5,num=21,samples=ns)
    gridsearch("alpha_b",min=0.,max=1,num=21,samples=ns)
    gridsearch("alpha_r",min=0.,max=1.5,num=21,samples=ns)
    gridsearch("tau_x",min=1,max=10,num=21,samples=ns)
    gridsearch("gamma",min=0.8,max=1,num=21,samples=ns)

    gridsearch("eta_init",min=0.0,max=0.1,num=21,samples=ns)
    gridsearch("eta_final",min=0.0,max=0.1,num=21,samples=ns)
    gridsearch("eta_tau",min=0.0,max=10000,num=21,samples=ns)
    gridsearch("eta_decay",min=0.0,max=100,num=21,samples=ns)
    
    gridsearch("eps_init",min=0.0,max=0.2,num=21,samples=ns)
    gridsearch("eps_final",min=0.0,max=0.1,num=21,samples=ns)
    gridsearch("eps_tau",min=0.0,max=10000,num=21,samples=ns)
    gridsearch("eps_decay",min=0.0,max=100,num=21,samples=ns)

gs1()
