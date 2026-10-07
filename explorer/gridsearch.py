# Copyright (c) 2018-2019 Katori lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: サーチツール
import os
import copy
import numpy as np
import pandas as pd
from . import common

def execute(csv=None,config=None):
    """
    単体で実行し、結果をCSVフィアルに保存する。
    csv: 出力するCSVファイルの名前（デフォルト値：None,自動で名前つけ）
    config: 基準となる設定(デフォルト値：None,共通設定を使用する)
    """
    ### setup
    if config==None:
        config=common.config
    if csv==None:
        filename=common.prefix+"_execute1.csv"
        csv=common.name_file(filename,path=None)
        # NOTE: このcsvはパスを含まないファイル名

    ### report (start)
    s = "### Execute \n"
    s += "Data:**%s**  \n" % (csv)
    common.report(s)

    ### prepare dataframe
    df = pd.DataFrame(index=[],columns=common.columns)
    cnf= copy.copy(config)
    cnf.csv=csv # XXX
    x  = common.config2series(cnf)
    #df = df.append(x,ignore_index=True)# Pandas1.4以降で非推奨
    df = pd.concat([df, pd.DataFrame([x])],ignore_index=True,axis=0)
    print(df)

    ### execute and save data
    df = common.execute(cnf,df)
    common.save_dataframe(df,csv)
    print(df)

    ### report (done)
    x = df.iloc[0]
    s = ""
    s+="Results:  \n"
    s+="```\n"
    for i,c in enumerate(common.columns):
        s += "%s: %s\n" % (common.columns[i],str(x[i]))
    s+="```\n"
    common.report(s)

def scan1d(X1,
array=np.empty(0),min=0,max=1,num=None,step=1,dtype='float64',
samples=1,csv=None,config=None):
    """
    １変数でグリッドサーチを行い、結果をCSVフィアルに保存する。
    min,maxで指定した範囲を等間隔で調べる。num, stepのどちらかを指定する。
    arrayを指定する場合は、arrayで指定した点を調べる。
    X1     :グリッドサーチを行う変数名
    array  :
    min,max:グリッドサーチを行う範囲、
    num    :グリッドサーチを行う点の数
    step   :グリッドサーチを行う点の間隔
    dtype  :変数の型 （デフォルト値：float64）、整数型を使う場合は int64 を指定する。
    samples:各点で指定した数の異なる乱数のシードについて調べる。サンプル数(デフォルト値：１)
    csv    :出力するCSVファイルの名前（デフォルト値：None,自動で名前つけ）
    config :基準となる設定(デフォルト値：None,共通設定を使用する)
    """
    ### setup
    if config==None:
        config=common.config
    if csv==None:
        filename=common.prefix+"_scan1d_%s.csv" % (X1)
        csv=common.name_file(filename,path=None)

    if hasattr(config,'plot'): setattr(config,'plot',False)
    if hasattr(config,'show'): setattr(config,'show',False)
    if hasattr(config,'savefig'): setattr(config,'savefig',False)

    #if array == None:
    if len(array) == 0:
        if step != None:
            array = np.arange(min,max,step,dtype=dtype)
        if num != None:
            array = np.linspace(min,max,num,dtype=dtype)
    # NOTE: arrayは list の場合、numpy arrayの場合がある。

    ### prepare dataframe
    df = pd.DataFrame(index=[],columns=common.columns)
    for seed in np.arange(samples,dtype='int64'):
        id=0
        for x1 in array:
            cnf=copy.copy(config)
            cnf.csv=csv
            setattr(cnf,'id',id)
            #if common.seed != None: # NOTE　なぜこ条件がある？
            #    setattr(cnf,common.seed,seed)
            setattr(cnf,'seed',seed)
            setattr(cnf,X1,x1)
            s = common.config2series(cnf)
            #df = df.append(s,ignore_index=True) # Pandas1.4以降で非推奨
            df = pd.concat([df, pd.DataFrame([s])],ignore_index=True,axis=0)
            id+=1
    #print(df)

    ### report
    s = "### Grid search 1D (" +X1+ ") \n"
    s+= "1D grid search on "\
    +X1+" from "+str(array[0])+" to "+str(array[-1])+" ("+str(len(array))+" points "
    if samples >=2:
        s+=str(samples)+" samples"
    s+= ")\n\n"
    s+= "Data:**%s**  \n" % (csv)
    common.report(s)

    ### execute and save data
    common.report_time_start()
    df = common.execute(cnf,df)
    common.report_time_done()
    common.save_dataframe(df,csv)
    #print(df)

def scan1ds(X1,array=np.empty(0),min=0,max=1,num=None,step=1,dtype='float64',samples=1,csv=None,config=None):
    scan1d(X1,array=array,min=min,max=max,num=num,step=step,dtype=dtype,samples=samples,csv=csv,config=config)

def scan2d(X1,X2,
array1=None,min1=0,max1=1,num1=None,step1=1,dtype1='float64',
array2=None,min2=0,max2=1,num2=None,step2=1,dtype2='float64',
samples=1,csv=None,config=None):
    ### setup
    if config==None:
        config=common.config
    if csv==None:
        filename=common.prefix+"_scan2d_%s_%s.csv" % (X1,X2)
        csv=common.name_file(filename,path=None)

    if hasattr(config,'plot'): setattr(config,'plot',False)
    if hasattr(config,'show'): setattr(config,'show',False)
    if hasattr(config,'savefig'): setattr(config,'savefig',False)

    if array1 == None:
        if step1 != None:
            array1 = np.arange(min1,max1,step1,dtype=dtype1)
        if num1 != None:
            array1 = np.linspace(min1,max1,num1,dtype=dtype1)
    if array2 == None:
        if step2 != None:
            array2 = np.arange(min2,max2,step2,dtype=dtype2)
        if num2 != None:
            array2 = np.linspace(min2,max2,num2,dtype=dtype2)

    ### prepare dataframe
    df=pd.DataFrame(index=[],columns=common.columns)
    for seed in np.arange(samples):
        id=0
        for x1 in array1:
            for x2 in array2:
                cnf=copy.copy(config)
                cnf.csv=csv
                setattr(cnf,'id',id)
                #if common.seed != None:
                #    setattr(cnf,common.seed,seed)
                setattr(cnf,'seed',seed)
                setattr(cnf,X1,x1)
                setattr(cnf,X2,x2)
                s = common.config2series(cnf)
                #df = df.append(s,ignore_index=True) # Pandas1.4以降で非推奨
                df = pd.concat([df, pd.DataFrame([s])],ignore_index=True,axis=0)
                id+=1

    ### report
    s = "### Grid search (" +X1+ " " +X2+ ") \n"
    s+= "2D grid search on "\
    +X1+" from "+str(array1[0])+" to "+str(array1[-1])+" and "\
    +X2+" from "+str(array2[0])+" to "+str(array2[-1])+" "\
    +" ("+str(len(array1))+" * "+str(len(array2))+" points "
    if samples >=2:
        s+=str(samples)+" samples"
    s+=")\n"
    s += "Data:** %s **  \n" % (csv)
    common.report(s)
    common.report_time_start()
    df = common.execute(cnf,df)
    common.report_time_done()
    common.save_dataframe(df,csv)
