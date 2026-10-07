# Copyright (c) 2022-2023 Katori lab. All Rights Reserved
# NARMAタスク
# TODO narmaの生成部分を確認
import argparse
import numpy as np
import matplotlib.pyplot as plt
import common_configurator as common
from models.metrics import *
from models.plotting import plot_rc
from datasets.dataset_narma import generate_narma

def execute(c):
    (NTtrain,NTtest) = (1000,1000)
    U,D = generate_narma(NTtrain+NTtest,order=10,seed=1)# NARMA10タスク(order=10)
    Utrain,Dtrain,Utest,Dtest = U[:NTtrain],D[:NTtrain],U[NTtrain:],D[NTtrain:]
    print("dataset shape: Utrain,Dtrain,Utest,Dtest",Utrain.shape,Dtrain.shape,Utest.shape,Dtest.shape)
    model = common.generate_model(c) # モデルのインスタンスを生成
    model.initialize() # 初期化
    Ytrain = model.fit(input=Utrain,output=Dtrain) # 訓練
    Ytest = model.run(input=Utest) # テスト
    c.RMSEtrain,c.NMSEtrain = evaluate(Dtrain[c.NTtrans:],Ytrain[c.NTtrans:]) # 評価(訓練)
    c.RMSEtest,c.NMSEtest = evaluate(Dtest[c.NTtrans:],Ytest[c.NTtrans:]) # 評価(テスト)
    print("RMSE(train):{:.3f} NMSE(train):{:.3f} RMSE(test):{:.3f} NMSE(test):{:.3f}"
          .format(c.RMSEtrain,c.NMSEtrain,c.RMSEtest,c.NMSEtest))
    if c.plot: 
        plot_rc(U=Utest[c.NTtrans:],X=model.X[c.NTtrans:],Y=Ytest[c.NTtrans:],D=Dtest[c.NTtrans:])

def evaluate(D,Y):
    RMSE = rmse(D,Y)
    NMSE = nmse(D,Y)
    return RMSE,NMSE

if __name__ == '__main__':
    c = common.load_config("config.config_narma_esn1")
    a = common.get_arguments()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
