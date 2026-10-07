# Copyright (c) 2018-2023 Katori Lab. All Rights Reserved
# 最適化 BayesianOptimizationを含む
# optimization2.pyをコピーしてBayesianOptimizationを無効化したものをoptimization.pyとする。
# 更新はoptimization2.py で行う。

import numpy as np
import copy
import pandas as pd
import copy
import scipy.optimize
#from bayes_opt import BayesianOptimization
from . import common

listx=[] # list of configuration of x
best_config = None
samples = None
config_opt = None
target = None
target_function = None
operation = None
episode = 0
is_print_detail = False

def clear():
    listx.clear()

def appendid():
    # NOTE 不要な関数だが、古いテストコードとの互換性を確保するため残す。
    #listx.append({'type':"id",'name':"id",'value':0})
    return None

def appendseed(name="seed"):
    # NOTE 不要な関数だが、古いテストコードとの互換性を確保するため残す。
    #listx.append({'type':"seed",'name':name,'value':0})
    return None

def append(name,value=0,min=-1,max=1,round=8):
    listx.append({'type':"f", 'name':name, 'value':value, 'min':min, 'max':max, 'variable':1,'round':round})

def count_key_value(list,key,value):
    count=0
    for c in list:
        if c[key]==value:
            count+=1
    return count

def func1(row):
    return row['x1']+row['x2']

def function(X):
    """
    最適化の対象となる関数。受け取った変数Xについて、関数の出力Yを返す。
    内部でcommon.execute()を呼び出してメインコードから返り値を取得する。
    変数Xとともに乱数seedを設定してメインコードを実行し、返り値の平均値を返す。
    引数：
    X(numpy array): 関数の引数. 行数はデータ点数、列数は最適化対象の変数の数
    返り値：
    Y(numpy array): 関数の返り値、要素数はテータ点数と同じ
    """

    """
    最適化のプロセスでは、CSVファイルに計算結果が逐次書き込まれていく。
    この際、CSVのデータからどの反復で書き込まれた計算結果なのかを判別する必要がある。
    そこで反復番号(episode)とデータ点の番号からidを設定し、これをCSVに保存する。
    """
    
    global episode,is_print_detail

    ### コラム名のリスト
    xnames = ["id", "seed"] + [cx['name'] for cx in listx]
    #print("xmanes",xnames)

    ### df1: X をデータフレームに変換(id,seedのコラムを追加)
    df1 = pd.DataFrame(index=[],columns=xnames)
    for irow,x in enumerate(X):
        id = episode * 10000 + irow # NOTE: 最適化のepisode番号とデータ点の行番号でidを設定する  
        vx = [int(id), 0] + list(x)
        s1 = pd.Series(vx, index=xnames)
        df1 = pd.concat([df1, pd.DataFrame([s1])],ignore_index=True,axis=0)
    #print("df1:\n",df1)

    ### df2: 異なる乱数seed値について多重化
    df2 = pd.DataFrame(index=[],columns=xnames)
    for id in range(len(df1.index)): # df1の行数分を繰り返す
        s2 = df1.iloc[id]
        for i_sample in range(samples):
            s2[1] = i_sample
            df2 = pd.concat([df2, pd.DataFrame([s2])],ignore_index=True,axis=0)
    #print("df2:\n",df2)

    ### df3: 全コラムのデータフレームを構成する
    df3 = pd.DataFrame(index=[],columns=common.columns)
    for i in range(len(df2.index)):
        cnf = copy.copy(config_opt)
        s0 = df2.iloc[i]
        for j in range(len(df2.columns)):
            setattr(cnf,df2.columns[j],s0[j])
        s1 = common.config2series(cnf)
        df3 = pd.concat([df3, pd.DataFrame([s1])],ignore_index=True,axis=0)
    #print("df3:A\n",df3)

    ### 
    cnf = copy.copy(config_opt)
    df3 = common.execute(cnf,df3) # cnfを渡すのはNG?おそらく問題ない

    #pd.set_option('display.max_rows', None)# 表示の設定：多数の行のとき、表示を省略しない
    
    df4 = df3[df3['id']//10000 == episode] # idから現在のepisodeに対応するデータを抽出
    #print("df4:\n",df4)

    ### 最適化の評価値(TARGET)の計算とソート
    df4 = df4.copy()
    if target != None:
        #df4['TARGET'] = df4[target]
        df4.loc[:, 'TARGET'] = df4[target]
    if target_function != None:
        df4['TARGET']=df4.apply(target_function,axis=1)
    
    ### df5: id による集約（異なるseed値について平均をとる）
    df5 = df4.groupby(df4['id']).mean()
    Y = df5['TARGET'].to_numpy()
    #print("df5:\n",df5)

    ### df6: ソート(結果の表示用)
    if operation == "max":
        df6 = df5.sort_values('TARGET',ascending=False)
    if operation == "min":
        df6 = df5.sort_values('TARGET',ascending=True)
    
    if is_print_detail:
        print(df6)

    episode += 1

    return Y

def maximize(csv=None,config=None,method="pnm",target=None,TARGET=None,population=10,iteration=10,samples=1):
    optimize("max",csv,config,method,target,TARGET,population,iteration,samples)

def minimize(csv=None,config=None,method="pnm",target=None,TARGET=None,population=10,iteration=10,samples=1):
    optimize("min",csv,config,method,target,TARGET,population,iteration,samples)

def particle(fun=None,x0=None,bounds=None):
    return 0

def optimize(operation_,csv,config,method_,target_,target_function_,population_,iterations_,samples_):
    """
    最適化を実行する
    注意：この関数の内部では、データフレームを使わない。
    """
    global config_opt
    global operation
    global target
    global target_function
    global population
    global iterations
    global samples

    operation = operation_
    target = target_
    target_function = target_function_
    population = population_
    iterations = iterations_
    samples = samples_
    method = method_

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

    ### TODO error message
    if target == None and target_function == None :
        print("Error: target/TARGET not specified"); return 1

    ### レポート(最適化の設定)

    text = "### Optimization \n"
    text += "Configuration:  \n"
    text += "operation: %s \n" % operation
    text += "```\n"
    for j,cx in enumerate(listx):
        if cx['type']=='f':
            text += "{:8s}:{:9.6f}[{:9.6f},{:9.6f}]({:d})\n".format(cx['name'],cx['value'],cx['min'],cx['max'],cx['round'])
        #print("xxx",cx['min'])

    if target != None:
        text += "target: %s \n" % target
    if target_function != None:
        text += "TARGET: %s \n" % target_function

    text += "iteration: %s \n" % iterations
    text += "population: %s \n" % population
    text += "samples: %s \n" % samples

    #text += "target=%s \n" % op['target']
    text += "```\n"
    text+= "Start:" + common.string_now() + "  \n"
    common.report(text)

    ### 初期値とバウンド（変数の可動範囲）のリストを作成
    x0=[]
    bounds=[]
    for cx in listx:
        x0.append(cx['value'])
        bounds.append((cx['min'],cx['max']))

    #res = scipy.optimize.minimize(fun=function,x0=x0,bounds=bounds,method="nelder-mead",options={'maxiter':10})

    if method=="pnm":
        tbest,xbest = optimize_pnm()
    #elif method=="bayesian":
    #    tbest,xbest = optimize_bayesian(x0,bounds)
    else:
        raise ValueError(f"Method '{method}' is not supported.")
    
    # TODO nelder-meadを実装する。

    ### 最適化された変数をconfigに設定

    global best_config
    best_config = config_opt
    for i,cx in enumerate(listx):
        setattr(best_config,cx['name'],xbest[i])

    ### レポート（最適化の結果）
    
    common.report("Done :" + common.string_now() + "  \n")
    text="Optimization result:  \n"
    text+="```\n"
    for i,cx in enumerate(listx):
        text += "{:8s}:{:9.6f}\n".format(cx['name'],xbest[i])
    text += "terget: %s \n" % tbest
    text += "```\n"
    common.report(text)

def optimize_bayesian(x0,bounds):
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
    init_points = 2 #int(iterations*0.1)
    n_iter = iterations-init_points
    if operation=="min":
        print("NOTE: Following [target] shows the negative value of the objective function.")
    bo.maximize(init_points=init_points,n_iter=n_iter)
    if operation=="min": tbest =-bo.max['target']
    if operation=="max": tbest = bo.max['target']
    xbest = []
    for i,(key,x) in enumerate(bo.max['params'].items()): xbest.append(x)

    return tbest,xbest

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

    X = np.array(x).reshape(1, -1)

    if operation=="min": f = -function(X)
    if operation=="max": f =  function(X)
    # NOTE BayesianOptimizationは最大化(maximize)のみサポートなので、
    # 最小化は対象の関数に負号をつけてその最大化を行う。

    return f[0]

def optimize_pnm():
    ### 初期化
    global is_print_detail; is_print_detail = True

    ### X0: ランダムに決めた変数を初期値とする。１行目は設定された値
    x0 = np.zeros(0)
    for cx in listx:
        x0 = np.append(x0,cx['value'])

    X0 = x0
    for i in range(population-1):
        x = np.zeros(len(x0))
        for j,cx in enumerate(listx):
            x[j] = np.random.uniform(cx['min'],cx['max'])
            if "round" in cx: x[j] = np.round(x[j],cx['round'])
        X0 = np.vstack([X0,x])
    #print("X0:\n",X0)

    ### 最適化のメインループ
    num_shrink = int(population/2)
    num_reflect = population - num_shrink
    tbest = -1e99 # 暫定最適な返り値、大きな値がより良い
    xbest = X0[0]
    im=0
    while im < iterations:
        print("Iteration:",im+1,"/",iterations)
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
            for i in range(population):
                x = X1[i]
                for j,cx in enumerate(listx):
                    if "round" in cx: x[j] = round(x[j],cx['round'])
                    if x[j] > cx['max']: x[j] = cx['max']
                    if x[j] < cx['min']: x[j] = cx['min']

        Y = function(X1)
        #Ycol= Y[:, np.newaxis]
        #print(Ycol,X1)
        #print(np.hstack((Ycol,X1)))

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

        #print("X1,Y:")
        #for i in range(len(X1)): print(X1[i], Y[i])
        #print("X2,sY:")
        #for i in range(len(X2)): print(X2[i], sY[i])
        #print("asY",asY)
        print("current optimal: x, target:",xbest,tbest)

        im+=1
    ### 最適化のメインループここまで

    return tbest,xbest




