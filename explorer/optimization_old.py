# Copyright (c) 2018-2019 Katori Lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: optimization.py

import subprocess
import sys
import re
import numpy as np
import copy
import os
import pandas as pd
import datetime
import copy
#import time
from . import common

listx=[] # list of configuration of x
best_config = None

def clear():
    listx.clear()

def appendid():
    listx.append({'type':"id",'name':"id",'value':0})

def append(name,value=0,min=-1,max=1,round=8):
    listx.append({'type':"f", 'name':name, 'value':value, 'min':min, 'max':max, 'variable':1,'round':round})

def appendseed(name="seed"):
    listx.append({'type':"seed",'name':name,'value':0})

def execute_df(config,df):
    ### Error message
    if len(df.columns) != len(listx):
        print("Error: size of given dataframe does not match listx. dataframe has %d columns, and listx has %d elements" % (len(df.columns),len(listx)) )
        return 1

    ### prepare dataframe
    id=0
    df1=pd.DataFrame(index=[],columns=common.columns)
    for i in range(len(df.index)): # dfの行数分を繰り返す
        cnf=copy.copy(config)
        s0 = df.iloc[i]
        for j in range(len(listx)):
            setattr(cnf,df.columns[j],s0[j])
        s1 = common.config2series(cnf)
        #df1 = df1.append(s1,ignore_index=True)# Pandas1.4以降で非推奨
        df1 = pd.concat([df1, pd.DataFrame([s1])],ignore_index=True,axis=0)

    ## execute
    df1 = common.execute(cnf,df1)
    return df1

def count_key_value(list,key,value):
    count=0
    for c in list:
        if c[key]==value:
            count+=1
    return count

def func1(row):
    return row['x1']+row['x2']

def maximize(csv=None,config=None,target=None,TARGET=None,population=10,iteration=10,samples=1):
    optimize(+1,csv,config,target,TARGET,population,iteration,samples)

def minimize(csv=None,config=None,target=None,TARGET=None,population=10,iteration=10,samples=1):
    optimize(-1,csv,config,target,TARGET,population,iteration,samples)

def optimize(minmax,csv,config,target,target_function,num_population,num_iteration,num_samples):
    """
    関数の最小値・最大値を探す。

    """
    ### setup
    if config==None:
        config=copy.copy(common.config)
    if csv==None:
        #csv=common.name_file(common.prefix+"_opt.csv")
        filename=common.prefix+"_opt.csv"
        csv=common.name_file(filename,path=None)
    #config.csv=csv

    if hasattr(config,'plot'): setattr(config,'plot',False)
    if hasattr(config,'show'): setattr(config,'show',False)
    if hasattr(config,'savefig'): setattr(config,'savefig',False)

    num_shrink = int(num_population/2)
    num_reflect = num_population - num_shrink

    ### TODO エラーメッセージの表示

    if target == None and target_function == None :
        print("Error: target/TARGET not specified"); return 1

    count_seed=count_key_value(listx,'type','seed')
    if count_seed==0 and num_samples>1: print("Error: seed not specified"); return 1

    count_id=count_key_value(listx,'type','id')
    if count_id==0: print("Error: id not appended"); return 1

    ### レポート
    
    text="### Optimization \n"
    text+="Configuration:  \n"
    text+="```\n"
    for j,cx in enumerate(listx):
        if cx['type']=='id':
            text += "{:8s}:{:9.6f}\n".format(cx['name'],cx['value'])
        if cx['type']=='seed':
            text += "{:8s}:\n".format(cx['name'])
        if cx['type']=='f':
            text += "{:8s}:{:9.6f}[{:9.6f},{:9.6f}]({:d})\n".format(cx['name'],cx['value'],cx['min'],cx['max'],cx['round'])
        #print("xxx",cx['min'])

    if target != None:
        text += "target: %s \n" % target
    if target_function != None:
        text += "TARGET: %s \n" % target_function

        text += "iteration: %s \n" % num_iteration
        text += "population: %s \n" % num_population
        text += "samples: %s \n" % num_samples

    #text += "target=%s \n" % op['target']
    text += "```\n"
    text+= "Start:" + common.string_now() + "  \n"
    common.report(text)

    for cx in listx:
        print(cx)

    ### 最適化プロセスの初期値を設定

    ### prepare dataframe (df0) with random values
    ### df0: ランダムに決めた変数を初期値とする。１行目は設定された値
    xnames=[]
    for cx in listx:
        xnames.append(cx['name'])
    vx=[]
    for cx in listx:
        vx.append(cx['value'])
    s0 = pd.Series(vx, index=xnames)
    #print("s0\n",s0)

    df0 = pd.DataFrame(index=[],columns=xnames)
    #df0 = df0.append(s0,ignore_index=True) # Pandas1.4以降で非推奨
    df0 = pd.concat([df0, pd.DataFrame([s0])],ignore_index=True,axis=0)
    id = 1
    for i in range(num_population-1):
        s1 = s0
        #for j in range(len(listx)):
        for j,cx in enumerate(listx):
            #cx=listx[j]
            if cx['type']=='id' :
                s1[j] = id

            if cx['type']=='f' and cx['variable'] :
                x=np.random.uniform(cx['min'],cx['max'])
                if "round" in cx: x = round(x,cx['round'])
                s1[j] = x

            if cx['type']=='i' and cx['variable'] :
                x=np.random.randint(cx['min'],cx['max']+1)
                s1[j] = x

        #df0 = df0.append(s1,ignore_index=True)# Pandas1.4以降で非推奨
        df0 = pd.concat([df0, pd.DataFrame([s1])],ignore_index=True,axis=0)
        id = id + 1
    #print("df0:\n",df0)

    ### 最適化のメインループ
    tbest = -1e99 # 暫定最適な返り値、大きな値がより良い
    sbest = df0.iloc[0] # 暫定最適値を与える引数
    im=0
    while im < num_iteration:
        print("Iteration:",im+1,"/",num_iteration)
        ### df1
        if im==0 : # for the first iteration, set df1 random values(df0)
            df1 = df0
        else:
            df1previous = df1
            df1 = pd.DataFrame(index=[],columns=xnames)
            id=0
            for k in range(num_shrink):# crossover
                s1 = s0
                #s1[0] = id
                for j,cx in enumerate(listx):
                    if cx['type']=='id':
                        s1[j] = id
                    if cx['type']=='f' and cx['variable']:
                        x = (sbest[j] + df1previous.iloc[k+1,j])/2.0
                        if "round" in cx: x = round(x,cx['round'])
                        if x>cx['max']: x=cx['max']
                        if x<cx['min']: x=cx['min']
                        s1[j] = x
                    if cx['type']=='i' and cx['variable'] :
                        x = int( (sbest[j] + df1previous.iloc[k+1,j])/2.0 )
                        s1[j] = x
                #df1 = df1.append(s1,ignore_index=True)# Pandas1.4以降で非推奨
                df1 = pd.concat([df1, pd.DataFrame([s1])],ignore_index=True,axis=0)
                id=id+1
            for k in range(num_reflect):# mutation
                s1 = s0
                for j,cx in enumerate(listx):
                    cx=listx[j]
                    if cx['type']=='id':
                        s1[j] = id
                    if cx['type']=='f' and cx['variable']:
                        x = sbest[j] + (sbest[j] - df1previous.iloc[k+1,j])*1.5
                        if "round" in cx: x = round(x,cx['round'])
                        if x>cx['max']: x=cx['max']
                        if x<cx['min']: x=cx['min']
                        s1[j] = x
                    if cx['type']=='i' and cx['variable'] :
                        x = int( sbest[j] + (sbest[j] - df1previous.iloc[k+1,j])*1.5 )
                        s1[j] = x
                #df1 = df1.append(s1,ignore_index=True)# Pandas1.4以降で非推奨
                df1 = pd.concat([df1, pd.DataFrame([s1])],ignore_index=True,axis=0)
                id=id+1

        df1['id'] = df1['id'].astype("int64")
        #print("df1:\n",df1)

        ### df2: multiply df1 with different seeds
        df2 = pd.DataFrame(index=[],columns=xnames)
        for i in range(num_population):
            s1 = df1.iloc[i]
            #print(s1)
            for i_sample in range(num_samples):
                s2 = s1.copy()
                for j in range(len(listx)):
                    if listx[j]['type']=='seed':
                        s2[j] = i_sample
                #print(s2)
                #df2 = df2.append(s2,ignore_index=True)# Pandas1.4以降で非推奨
                df2 = pd.concat([df2, pd.DataFrame([s2])],ignore_index=True,axis=0)
                
        #print("df2:\n",df2)

        ### execute, df2のパラメータ値についてプログラムを実行

        filename=common.prefix+"_opt"+str(im)+".csv"
        csv_tmp=common.name_file(filename,path=None)
        config.csv=csv_tmp

        df3 = execute_df(config,df2)
        df3['id'] = df3['id'].astype("int64")
        #print("df3:\n",df3)

        ### 最適化の評価値(TARGET)の計算とソート
        if target != None:
            df3['TARGET'] = df3[target]
        if target_function != None:
            df3['TARGET']=df3.apply(target_function,axis=1)
        if minmax==+1:#maximize
            df3 = df3.sort_values('TARGET',ascending=False)
        if minmax==-1:#minimize
            df3 = df3.sort_values('TARGET',ascending=True)
        #print("df3 sorted",df3)

        if im == 0:
            df5=df3
        else:
            df5=pd.concat([df5, df3], axis=0)
        #print("df5:\n",df5)
        common.save_dataframe(df5,csv)

        ### id による集約（異なるseed値について平均をとる）とソート
        df4=df3.groupby(df3['id']).mean()
        if minmax==+1:#maximize
            df4 = df4.sort_values('TARGET',ascending=False)
        if minmax==-1:#minimize
            df4 = df4.sort_values('TARGET',ascending=True)
        i0 = int(df4.index[0]) # 最適値を含む行のインデックス
        t0 = df4['TARGET'].iloc[0] #
        #print("df4:\n",df4)

        if num_samples>1:
            df4b=df4.drop('seed',axis=1)
            print(df4b)

        if im == 0:
            tbest = t0
            sbest = df1.iloc[i0]
            xbest = df4.iloc[0]
        else:
            if minmax == +1 and t0 > tbest:
                tbest = t0
                sbest = df1.iloc[i0]
                xbest = df4.iloc[0]
            if minmax == -1 and t0 < tbest:
                tbest = t0
                sbest = df1.iloc[i0]
                xbest = df4.iloc[0]
        #print("sbest:\n",sbest)
        #print("xbest:\n",xbest)
        im+=1
    ### 最適化のメインループここまで
    global best_config
    best_config = common.series2config(config,xbest,xbest.index)

    ### print result
    common.report("Done :" + common.string_now() + "  \n")
    text="Optimization result:  \n"
    text+="```\n"
    for i in range(len(xbest)):
        text += "{:8s}:{:9.6f}\n".format(df4.columns[i],xbest[i])
    text += "```\n"
    common.report(text)
