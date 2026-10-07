# Copyright (c) 2018-2025 Katori lab. All Rights Reserved
# 強化学習(ninerooms)+SNNモデルのテスト

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
common.load_config("config.config_rl_ninerooms_rtdl_snn")# 設定ファイルの読込
common.prefix  = "data%s_rl_snn" % common.string_now() # 実験名（ファイルの接頭辞）
common.dir_path= "data/"+common.prefix # 実験データの出力先パス
common.exe     = "python main_rl.py " # 実行されるプログラム
common.columns = ['seed','id','alpha_i','alpha_b','beta_i','beta_b','gamma','eta_init','eta_final','eta_tau','eta_decay','eps_init','eps_final','eps_tau','eps_decay',\
                  'Nx','m_base','p_intra','p_inter','g_EE','g_EI','g_IE','g_II','ge_tau','gi_tau','ou_mu_e','ou_mu_i','ou_sigma_e','ou_sigma_i','ou_tau_e','ou_tau_i',\
                    'total_reward','mean_reward','avg_spike_rate','spike_rate','E_spike_rate','I_spike_rate',\
                        'mean_lv','mean_erank','ex_max','ex_tau','tca','tr','td','ds','dt','const', 'ca_mu','ca_sigma'] # データフレームの列名

common.parallel= 16 # 並列実行の数
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
    return row['total_reward'] - 20 * np.abs(row['avg_spike_rate']-20)

def optimize():
    opt.clear()#設定をクリアする
    # 変数の追加([変数名],[基本値],[下端],[上端],[まるめの桁数])
    #opt.append("beta_r",value=0.01,min=0.,max=1,round=3)
    #opt.append("beta_i",value=0.01,min=0.,max=1,round=3)
    opt.append("alpha_i",value=0.5,min=0.,max=1,round=2)
    #opt.append("alpha_b",value=0.5,min=0.,max=1,round=2)
    #opt.append("alpha_r",value=0.8,min=0.,max=2,round=2)
    #opt.append("tau_x",value=2.0,min=1,max=10,round=2)
    #opt.append("gamma",value=0.9,min=0.8,max=1,round=3)

    #opt.append("eta_init" ,value=0.01,min=0.,max=0.1,round=3)
    #opt.append("eta_final",value=0.0,min=0.,max=0.1,round=4)
    #opt.append("eta_tau"  ,value=10000,min=0.,max=10000,round=0)
    #opt.append("eta_decay",value=4,min=0.,max=100,round=1)

    #opt.append("eps_init" ,value=0.1,min=0.,max=0.2,round=3)
    #opt.append("eps_final",value=0.02,min=0.,max=0.1,round=4)
    #opt.append("eps_tau"  ,value=1000,min=0.,max=10000,round=0)
    #opt.append("eps_decay",value=10,min=0.,max=100,round=1)

    opt.append("g_EE", value=3.0, min=0.0, max=10.0, round=2)
    opt.append("g_EI", value=3.0, min=0.0, max=10.0, round=2)
    opt.append("g_IE", value=6.0, min=0.0, max=10.0, round=2)
    opt.append("g_II", value=6.0, min=0.0, max=10.0, round=2)
    opt.append("ge_tau", value=2, min=1, max=10, round=1)
    opt.append("gi_tau", value=5, min=1, max=10, round=1)
    opt.append("ou_mu_e", value=0.40, min=0.0, max=1.0, round=2)
    opt.append("ou_mu_i", value=0.30, min=0.0, max=1.0, round=2)
    opt.append("ou_sigma_e", value=0.2, min=0.0, max=1.0, round=2)
    opt.append("ou_sigma_i", value=0.1, min=0.0, max=1.0, round=2)
    #opt.append("ou_tau_e", value=2.0, min=0.0, max=10.0, round=1)
    #opt.append("ou_tau_i", value=5.0, min=0.0, max=10.0, round=1)
    opt.append("ex_tau", value=200, min=1, max=1000, round=1)

    #opt.maximize(target="total_reward",iteration=10,population=21,samples=5)
    opt.maximize(TARGET=func,iteration=10,population=21,samples=5)
    common.config = opt.best_config # 最適化で得られた設定を基本設定とする
# optimize()

### グリッドサーチ
def gridsearch(X1,min=0,max=1,num=41,samples=10):
    # 変数(X1)についてグリッドサーチ(GS)を行い、評価基準の変化をまとめてプロット
    gs.scan1ds(X1,min=min,max=max,num=num,samples=samples)# GSの実行
    df = common.load_dataframe()# 直前のGSの結果をデータフレームに読込
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(6,4))
    vs.vline(X1)
    vs.plot_stat_summary(df,X=X1,Y="total_reward",label="total reward",color=cmap(1))# 平均、標準偏差、最大値、最小値をプロット
    # vs.plot_stat_summary(df,X=X1,Y="mean_reward",label="mean reward",color=cmap(2))
    plt.grid(linestyle="dotted")
    plt.ylabel("Reward")
    #plt.yscale('log')
    #plt.ylim([0,1]) # y軸の範囲
    plt.legend()
    plt.xlabel(X1)
    vs.plt_output()

def gs1():
    gs.scan1ds("tca", min=100, max=2700, num=8, samples=2)
gs1()

def gs2b():
    ns = 20
    np = 11

    gs.scan2d("m_base", "Nx", min1=1, max1=5, num1=5,dtype1='int64', array2=[100,200,300,400,500,600,700,800,900,1000],samples=30)
    vs.plot2ds_pcolor("m_base","Nx","total_reward",fig="m_base_Nx_total")
    vs.plot2ds_pcolor("m_base","Nx","mean_reward",fig="m_base_Nx_mean")
    vs.plot2ds_pcolor("m_base","Nx","avg_spike_rate",fig="m_base_Nx_sr")
    vs.plot2ds_pcolor("m_base","Nx","mean_lv",fig="m_base_Nx_lv")
    vs.plot2ds_pcolor("m_base","Nx","mean_erank",fig="m_base_Nx_erank")

# gs2b()