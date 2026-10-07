# Copyright (c) 2018-2023 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: optimization.pyと同じ機能を簡単な記述で実装する。未完成

import subprocess
import sys
import re
import numpy as np
import copy
import os
import pandas as pd
import datetime
import copy
import scipy.optimize

#import time
from . import common

listx=[] # list of configuration of x
best_config = None
samples = None
config_opt = None
target = None
target_function = None
operation = None

def clear():
    listx.clear()

def append(name,value=0,min=-1,max=1,round=8):
    listx.append({'type':"f", 'name':name, 'value':value, 'min':min, 'max':max, 'variable':1,'round':round})

def execute_df(config,df0):
    """
    configの設定を基にdf0の設定で
    """
    ### Error message
    if len(df0.columns) != len(listx)+2: # XXX
        print("Error: size of given dataframe does not match listx. dataframe has %d columns, and listx has %d elements" % (len(df0.columns),len(listx)) )
        return 1

    print("df0:(execute_df)\n",df0)
    ### prepare dataframe
    id=0
    df1 = pd.DataFrame(index=[],columns=common.columns)
    for i in range(len(df0.index)):
        cnf = copy.copy(config)
        s0 = df0.iloc[i]
        #print(config)
        for j in range(len(df0.columns)):
            setattr(cnf,df0.columns[j],s0[j])
        s1 = common.config2series(cnf)
        #df1 = df1.append(s1,ignore_index=True)
        df1 = pd.concat([df1, pd.DataFrame([s1])],ignore_index=True,axis=0)

    ## execute
    print("df1:(execute_df)A\n",df1)
    df1 = common.execute(cnf,df1) # XXX cnfを渡すのはNG?おそらく問題ない
    pd.set_option('display.max_rows', None)# 表示の設定：多数の行のとき、表示を省略しない
    print("df1:(execute_df)B\n",df1)
    return df1

def count_key_value(list,key,value):
    count=0
    for c in list:
        if c[key]==value:
            count+=1
    return count

def func1(row):
    return row['x1']+row['x2']

def function(X,iteration):
    """
    最適化の対象となる関数。受け取った変数Xについて、関数の出力Yを返す。
    内部でcommon.execute()を呼び出して返り値を取得する。
    変数Xとともに乱数seedを設定してコードを実行し、返り値の平均値を返す。
    引数：
    X(numpy array): 行数はデータ点数、列数は最適化対象の変数の数
    iteration(int): 最適化のイテレーション番号
    返り値：
    Y(numpy array): 関数の返り値
    """

    """
    最適化のプロセスでは、CSVファイルに計算結果が逐次書き込まれていく。
    この際、CSVのデータからどの反復で書き込まれた計算結果なのかを判別する必要がある。
    そこで反復番号(iteration)とデータ点の番号からidを設定し、これをCSVに保存する。
    """
    
    ### コラム名のリスト
    xnames = ["id", "seed"] + [cx['name'] for cx in listx]
    #print("xmanes",xnames)

    ### df1: X をデータフレームに変換(id,seedのコラムを追加)
    df1 = pd.DataFrame(index=[],columns=xnames)
    for irow,x in enumerate(X):
        id = iteration*10000+irow # 最適化のiteration番号とデータ点の行番号でidを設定する  
        vx=[]
        vx.append(int(id))# id
        vx.append(0)# seed
        for i,cx in enumerate(listx): 
            vx.append(x[i])
        s1 = pd.Series(vx, index=xnames)
        df1 = pd.concat([df1, pd.DataFrame([s1])],ignore_index=True,axis=0)
        #print("s1\n",s1)
    #print("df1:\n",df1)
    #breakpoint()

    # id と seed の値を持つリストを生成
    ids = [iteration * 10000 + irow for irow in range(len(X))]
    seeds = [0] * len(X)
    data = list(zip(ids, seeds, *X.T))
    df1 = pd.DataFrame(data, columns=xnames)
    print("df1:\n",df1)
    #breakpoint()

    ### df2: 異なる乱数seed値について多重化
    df2 = pd.DataFrame(index=[],columns=xnames)
    for id in range(len(df1.index)): # dfの行数分を繰り返す
        s2 = df1.iloc[id]
        for i_sample in range(samples):
            s2[1] = i_sample
            #df2 = df2.append(s2,ignore_index=True)
            df2 = pd.concat([df2, pd.DataFrame([s2])],ignore_index=True,axis=0)
    #print("df2:\n",df2)

    ### df3: 全コラムのデータフレームを構成する
    df3 = pd.DataFrame(index=[],columns=common.columns)
    for i in range(len(df2.index)):
        cnf = copy.copy(config_opt)
        s0 = df2.iloc[i]
        #print(config)
        for j in range(len(df2.columns)):
            setattr(cnf,df2.columns[j],s0[j])
        s1 = common.config2series(cnf)
        df3 = pd.concat([df3, pd.DataFrame([s1])],ignore_index=True,axis=0)
    #print("df3:A\n",df3)
    cnf = copy.copy(config_opt)
    df3 = common.execute(cnf,df3) # XXX cnfを渡すのはNG?おそらく問題ない
    #pd.set_option('display.max_rows', None)# 表示の設定：多数の行のとき、表示を省略しない
    
    df4 = df3[df3['id']//10000 == iteration] # idから現在のiterationに対応するデータを抽出
    #print("df4:\n",df4)

    ### 最適化の評価値(TARGET)の計算とソート
    if target != None:
        #df4['TARGET'] = df4[target]
        df4.loc[:, 'TARGET'] = df4[target]
    if target_function != None:
        df4['TARGET']=df4.apply(target_function,axis=1)
    # ソート（不要）
    if operation == "max":#maximize
        df4 = df4.sort_values('TARGET',ascending=False)
    if operation == "min":#minimize
        df4 = df4.sort_values('TARGET',ascending=True)
    #print("df4 sorted\n",df4)

    ### df5: id による集約（異なるseed値について平均をとる）
    df5 = df4.groupby(df4['id']).mean()

    Y = df5['TARGET'].to_numpy()
    print("df5:\n",df5)
    #print("Y:\n",Y)

    return Y

def maximize(csv=None,config=None,target=None,TARGET=None,population=10,iteration=10,samples=1):
    optimize("max",csv,config,target,TARGET,population,iteration,samples)

def minimize(csv=None,config=None,target=None,TARGET=None,population=10,iteration=10,samples=1):
    optimize("min",csv,config,target,TARGET,population,iteration,samples)


def particle(fun=None,x0=None,bounds=None):
    return 0

def optimize(operation_,csv,config,target_,target_function_,num_population,num_iteration,samples_):
    """
    この関数の内部では、データフレームを使わない。
    """
    global config_opt
    global operation
    global target
    global target_function
    global samples

    ### setup
    if config == None:
        config_opt = copy.copy(common.config)
    else:
        config_opt = config
    if csv==None:
        #csv=common.name_file(common.prefix+"_opt.csv")
        filename=common.prefix+"_opt.csv"
        csv=common.name_file(filename,path=None)
    #config.csv=csv

    #filename=common.prefix+"_opt"+str(im)+".csv"
    filename=common.prefix+"_opt.csv"
    csv_tmp=common.name_file(filename,path=None)
    config_opt.csv=csv_tmp

    if hasattr(config_opt,'plot'): setattr(config_opt,'plot',False)
    if hasattr(config_opt,'show'): setattr(config_opt,'show',False)
    if hasattr(config_opt,'savefig'): setattr(config_opt,'savefig',False)

    operation = operation_
    target = target_
    target_function = target_function_
    samples = samples_

    ### TODO error message
    if target == None and target_function == None :
        print("Error: target/TARGET not specified"); return 1

    ### レポート

    text="### Optimization \n"
    text+="Configuration:  \n"
    text += "operation: %s \n" % operation
    text+="```\n"
    for j,cx in enumerate(listx):
        if cx['type']=='f':
            text += "{:8s}:{:9.6f}[{:9.6f},{:9.6f}]({:d})\n".format(cx['name'],cx['value'],cx['min'],cx['max'],cx['round'])
        #print("xxx",cx['min'])

    if target != None:
        text += "target: %s \n" % target
    if target_function != None:
        text += "TARGET: %s \n" % target_function

    text += "iteration: %s \n" % num_iteration
    text += "population: %s \n" % num_population
    text += "samples: %s \n" % samples

    #text += "target=%s \n" % op['target']
    text += "```\n"
    text+= "Start:" + common.string_now() + "  \n"
    common.report(text)

    ### 

    x0=[]
    bounds=[]
    for cx in listx:
        x0.append(cx['value'])
        bounds.append((cx['min'],cx['max']))

    #res = scipy.optimize.minimize(fun=function,x0=x0,bounds=bounds,method="nelder-mead",options={'maxiter':10})

    ### particle

    num_shrink = int(num_population/2)
    num_reflect = num_population - num_shrink

    ### prepare dataframe (df0) with random values
    ### df0: ランダムに決めた変数を初期値とする。１行目は設定された値
    #print("listx:",listx)

    x0 = np.zeros(0)
    for cx in listx:
        x0 = np.append(x0,cx['value'])

    X0 = x0
    for i in range(num_population-1):
        x = np.zeros(len(x0))
        for j,cx in enumerate(listx):
            x[j] = np.random.uniform(cx['min'],cx['max'])
            if "round" in cx: x[j] = np.round(x[j],cx['round'])
        X0 = np.vstack([X0,x])
    #print("X0:\n",X0)

    ### 最適化のメインループ
    tbest = -1e99 # 暫定最適な返り値、大きな値がより良い
    xbest = X0[0]

    im=0
    while im < num_iteration:
        print("Iteration:",im+1,"/",num_iteration)
        ### df1
        if im==0 : # for the first iteration, set df1 random values(df0)
            X1 = X0
        else:
            #Xprev = copy.copy(X1)
            X1 = np.zeros((0,len(x)))
            for k in range(num_shrink):# crossover
                x = (xbest + X2[k+1])/2.0
                X1 = np.vstack([X1,x])
            for k in range(num_reflect):# mutation
                x = xbest + (xbest - X2[k+1])*1.5
                X1 = np.vstack([X1,x])

            #print("X1",X1)
            for i in range(num_population):
                x = X1[i]
                for j,cx in enumerate(listx):
                    if "round" in cx: x[j] = round(x[j],cx['round'])
                    if x[j] > cx['max']: x[j] = cx['max']
                    if x[j] < cx['min']: x[j] = cx['min']

        #Y = np.zeros(0)
        #for i,x in enumerate(X1):
        #    y = function(x)
        #    Y = np.append(Y,y)
        #    #print(x,y)

        Y = function(X1,iteration=im)

        if operation == "max":
            asY=np.argsort(Y)[::-1]
            sY =np.sort(Y)[::-1]
        if operation == "min":
            asY=np.argsort(Y)
            sY =np.sort(Y)

        X2 = np.zeros_like(X1)
        for i1,i2 in enumerate(asY):
            X2[i1] = X1[i2]

        if im == 0:
            tbest = sY[0]
            xbest = X2[0]
        else:
            if operation == "max" and sY[0] > tbest:
                tbest = sY[0]
                xbest = X2[0]
            if operation == "min" and sY[0] < tbest:
                tbest = sY[0]
                xbest = X2[0]

        print("X1,Y:")
        for i in range(len(X1)): print(X1[i], Y[i])
        print("X2,sY:")
        for i in range(len(X2)): print(X2[i], sY[i])
        print("asY",asY)
        print("xbest tbest:",xbest,tbest)

        im+=1
    ### 最適化のメインループここまで

    global best_config
    best_config = config_opt
    for i,cx in enumerate(listx):
        setattr(best_config,cx['name'],xbest[i])

    ### print result
    common.report("Done :" + common.string_now() + "  \n")
    text="Optimization result:  \n"
    text+="```\n"
    for i,cx in enumerate(listx):
        text += "{:8s}:{:9.6f}\n".format(cx['name'],xbest[i])
    text += "terget: %s \n" % tbest
    text += "```\n"
    common.report(text)
