# Copyright (c) 2018-2019 Katori lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: サーチツール
import os
import numpy as np
import pandas as pd
import copy
from . import common

### random search
listx=[]
def clear():
    listx=[]
def append(name,min,max):
    listx.append({'type':'f', 'name':name,'min':min, 'max':max})
def appendint(name,min,max):
    listx.append({'type':'i', 'name':name,'min':min, 'max':max})

def random(num=1000,samples=1,csv=None,config=None):
    ### setup
    if config==None:
        config=common.config
    if csv==None:
        filename=common.prefix+"_random.csv"
        csv=common.name_file(filename,path=None)

    if hasattr(config,'plot'): setattr(config,'plot',False)
    if hasattr(config,'show'): setattr(config,'show',False)
    if hasattr(config,'savefig'): setattr(config,'savefig',False)

    ### report
    s = "### Random search (random) \n"
    s += "%d points search on  \n" % (num)
    for j,cx in enumerate(listx):
        s += "%s: (min:%f max:%f)  \n" % (cx['name'],cx['min'],cx['max'])
    s += "Exe: `%s`  \n" % common.exe
    s += "Data: **%s**  \n" % (csv)
    common.report(s)

    ### prepare dataframe
    id=0
    df=pd.DataFrame(index=[],columns=common.columns)
    for i in np.arange(num):
        ex1=''
        cnf=copy.copy(config)
        cnf.csv=csv
        for j,cx in enumerate(listx):
            if cx['type']=='f' :
                x=np.random.uniform(cx['min'],cx['max'])
                setattr(cnf,cx['name'],x)

            if cx['type']=='i' :
                x=np.random.randint(cx['min'],cx['max'])
                setattr(cnf,cx['name'],x)

        for seed in np.arange(samples):
            setattr(cnf,'seed',seed)
            setattr(cnf,'id',id)
            s = common.config2series(cnf)
            #df = df.append(s,ignore_index=True)# Pandas1.4以降で非推奨
            df = pd.concat([df, pd.DataFrame([s])],ignore_index=True,axis=0)
        id+=1
    #print(df)
    common.report_time_start()
    df = common.execute(cnf,df)
    common.report_time_done()
    common.save_dataframe(df,csv)
    print(df)
