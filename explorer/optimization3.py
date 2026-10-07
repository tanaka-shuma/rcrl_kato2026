# Copyright (c) 2018-2019 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: optimization.py Bayesian optimization

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
from bayes_opt import BayesianOptimization
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

def execute_df(config,df):
    ### Error message
    if len(df.columns) != len(listx)+2:
        print("Error: size of given dataframe does not match listx. dataframe has %d columns, and listx has %d elements" % (len(df.columns),len(listx)) )
        return 1

    #print("df:\n",df)
    ### prepare dataframe
    id=0
    df1 = pd.DataFrame(index=[],columns=common.columns)
    for i in range(len(df.index)):
        cnf = copy.copy(config)
        s0 = df.iloc[i]
        #print(config)
        for j in range(len(df.columns)):
            setattr(cnf,df.columns[j],s0[j])
        s1 = common.config2series(cnf)
        df1 = df1.append(s1,ignore_index=True)

    ## execute
    df1 = common.execute(cnf,df1)
    #print("df1:\n",df1)
    return df1

def count_key_value(list,key,value):
    count=0
    for c in list:
        if c[key]==value:
            count+=1
    return count

def func1(row):
    return row['x1']+row['x2']

def function(x):
    """
    受け取った変数xについて、seedを設定してコードを実行し、targetの平均値を返す。
    id は変数（パラメータ）に対応する。この関数では常に0。
    seed 乱数のシード
    """
    ###
    xnames=[]
    xnames.append("id")
    xnames.append("seed")
    for cx in listx:
        xnames.append(cx['name'])
    #print("xmanes",xnames)

    vx=[]
    vx.append(0)# id
    vx.append(0)# seed
    #for cx in listx:
    for i,cx in enumerate(listx):
        #x.append(cx['value'])
        vx.append(x[i])
    s1 = pd.Series(vx, index=xnames)
    #print("s1\n",s1)

    ### df2: multiply df1 with different seeds
    df2 = pd.DataFrame(index=[],columns=xnames)
    for i_sample in range(samples):
        s2 = s1.copy()
        s2[1] = i_sample
        df2 = df2.append(s2,ignore_index=True)
    #print("df2:\n",df2)

    ### execute, df2のパラメータ値についてプログラムを実行
    df3 = execute_df(config_opt,df2)
    #print("df3:\n",df3)

    ### 最適化の評価値(TARGET)の計算とソート
    if target != None:
        df3['TARGET'] = df3[target]
    if target_function != None:
        df3['TARGET']=df3.apply(target_function,axis=1)
    # ソート（不要）
    if operation == "max":#maximize
        df3 = df3.sort_values('TARGET',ascending=False)
    if operation == "min":#minimize
        df3 = df3.sort_values('TARGET',ascending=True)
    #print("df3 sorted\n",df3)

    ### id による集約（異なるseed値について平均をとる）
    df4 = df3.groupby(df3['id']).mean()

    t0 = df4['TARGET'].iloc[0] #
    #print("df4:\n",df4)

    return t0

def particle(fun=None,x0=None,bounds=None):
    return 0

def fun_x( # BayesianOptimization用の関数のラッパー
    x1,x2=None,x3=None,x4=None,x5=None,x6=None,x7=None,x8=None,x9=None,x10=None,
    x11=None,x12=None,x13=None,x14=None,x15=None,x16=None,x17=None,x18=None,x19=None,x20=None):
    x=[]
    if x1 != None: x.append(x1)
    if x2 != None: x.append(x2)
    if x3 != None: x.append(x3)
    if x4 != None: x.append(x4)
    if x5 != None: x.append(x5)
    if x6 != None: x.append(x6)
    if x7 != None: x.append(x7)
    if x8 != None: x.append(x8)
    if x9 != None: x.append(x9)
    if x10 != None: x.append(x10)
    if x11 != None: x.append(x11)
    if x12 != None: x.append(x12)
    if x13 != None: x.append(x13)
    if x14 != None: x.append(x14)
    if x15 != None: x.append(x15)
    if x16 != None: x.append(x16)
    if x17 != None: x.append(x17)
    if x18 != None: x.append(x18)
    if x19 != None: x.append(x19)
    if x20 != None: x.append(x20)

    if operation=="min": f = -function(x)
    if operation=="max": f =  function(x)

    return f

def maximize(csv=None,config=None,method="nelder-mead",target=None,TARGET=None,population=10,iteration=10,samples=1):
    optimize("max",csv,config,method,target,TARGET,population,iteration,samples)

def minimize(csv=None,config=None,method="nelder-mead",target=None,TARGET=None,population=10,iteration=10,samples=1):
    optimize("min",csv,config,method,target,TARGET,population,iteration,samples)

def optimize(operation_,csv,config,method_,target_,target_function_,num_population,iteration,samples_):
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
    method = method_

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

    text += "iteration: %s \n" % iteration
    text += "population: %s \n" % num_population
    text += "samples: %s \n" % samples

    #text += "target=%s \n" % op['target']
    text += "```\n"
    text+= "Start:" + common.string_now() + "  \n"
    common.report(text)

    ### 初期値とバウンドをリストに格納する。
    x0=[]
    bounds=[]
    for cx in listx:
        x0.append(cx['value'])
        bounds.append((cx['min'],cx['max']))


    #method="nelder-mead"
    #method="bayesian"

    if method == "nelder-mead":
        ### initial simplex
        n=len(x0)
        simplex=np.zeros((n+1,n))
        for i in range(n+1):
            simplex[i,:]=x0
        for i in range(n):
            #print("asdf",bounds[i][0],bounds[i][1])
            simplex[i,i]=(bounds[i][0]+bounds[i][1])/2
            
        option={'maxiter':iteration,'initial_simplex':simplex,'adaptive':True,'return_all':False}
        res = scipy.optimize.minimize(fun=function,x0=x0,method="nelder-mead",options=option)
        print(res)
        tbest = res.fun
        xbest = []
        for i,x in enumerate(res.x): xbest.append(x)

    if method == "bayesian":
        #初期値、境界の辞書を作成
        init_x = dict()
        pbounds_x = dict()
        for i in range(len(x0)):
            name = "x%d" % (i+1)
            pbounds_x[name] = (bounds[i])
            init_x[name] = x0[i]
        #print("pb x:",pbounds_x)

        bo = BayesianOptimization(f=fun_x,pbounds=pbounds_x,verbose=2,random_state=1)
        bo.probe(params=init_x,lazy=True)
        init_points = int(iteration*0.1)
        n_iter = iteration-init_points
        bo.maximize(init_points=init_points,n_iter=n_iter)
        #print(bo.max)
        if operation=="min": tbest =-bo.max['target']
        if operation=="max": tbest = bo.max['target']
        xbest = []
        for i,(key,x) in enumerate(bo.max['params'].items()): xbest.append(x)

    if method == "bayesian2":
        #初期値、境界の辞書を作成
        init_x = dict()
        pbounds_x = dict()
        for i in range(len(x0)):
            name = "x%d" % (i+1)
            pbounds_x[name] = (bounds[i])
            init_x[name] = x0[i]
        #print("pb x:",pbounds_x)

        bo = BayesianOptimization(f=fun_x,pbounds=pbounds_x,verbose=2,random_state=1)
        bo.probe(params=init_x,lazy=True)
        init_points = int(iteration*0.1)
        n_iter = iteration-init_points
        bo.maximize(init_points=init_points,n_iter=n_iter)
        #print(bo.max)
        if operation=="min": tbest =-bo.max['target']
        if operation=="max": tbest = bo.max['target']
        xbest = []
        for i,(key,x) in enumerate(bo.max['params'].items()): xbest.append(x)

    ### 最適化後の処理
    #print("xbest:",xbest)
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
