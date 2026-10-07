# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
# NARMAタスク+esn1モデルのテスト
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from explorer import common
from explorer import gridsearch as gs
from explorer import visualization as vs
from explorer import randomsearch as rs
from explorer import optimization as opt

### 共通設定
common.load_config("config.config_narma_esn1")# 設定ファイルの読込
common.prefix  = "data%s_narma_esn1" % common.string_now() # 実験名（ファイルの接頭辞）
common.dir_path= "data/"+common.prefix # 実験データの出力先パス
common.exe     = "python main_narma.py " # 実行されるプログラム
common.columns =['seed','id','Nx','alpha_i','alpha_r','alpha_b','beta_i','beta_r','beta_b','log10_lambda_ridge','delay',"RMSEtrain","NMSEtrain","RMSEtest","NMSEtest"]
common.parallel= 32 # 並列実行の数
common.setup()
common.report_common()
common.report_config()

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
rs1()

### 最適化
def func(row):# 関数funcでtargetを指定する。
    return row['y1'] + 0.3*row['y2']

def optimize():
    opt.clear()#設定をクリアする
    # 変数の追加([変数名],[基本値],[下端],[上端],[まるめの桁数])
    opt.append("beta_r",value=0.01,min=0.,max=1,round=3)
    opt.append("beta_i",value=0.01,min=0.,max=1,round=3)
    opt.append("alpha_i",value=0.2,min=0.,max=1,round=3)
    opt.append("alpha_r",value=1,min=0.,max=1,round=2)
    opt.append("log10_lambda_ridge",value=-4,min=-10,max=2,round=3)
    opt.minimize(target="NMSEtest",iteration=10,population=16,samples=4)
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
    vs.plot_stat_summary(df,X=X1,Y="NMSEtest",label="NMSE(test)",color=cmap(1))# 平均、標準偏差、最大値、最小値をプロット
    vs.plot_stat_summary(df,X=X1,Y="NMSEtrain",label="NMSE(train)",color=cmap(2))
    plt.grid(linestyle="dotted")
    plt.ylabel("NMSE")
    plt.yscale('log')
    #plt.ylim([0,1]) # y軸の範囲
    plt.legend()
    plt.xlabel(X1)
    vs.plt_output()

def gs1():
    ns=10
    #gridsearch("Nx",min=100,max=1000,num=41,samples=ns)
    gridsearch("alpha_i",min=0.,max=1,num=41,samples=ns)
    gridsearch("alpha_r",min=0.,max=1.5,num=41,samples=ns)
    gridsearch("beta_i",min=0.0,max=1,num=41,samples=ns)
    gridsearch("beta_r",min=0.0,max=1,num=41,samples=ns)
    gridsearch("log10_lambda_ridge",min=-10,max=2,num=41,samples=ns)
gs1()
