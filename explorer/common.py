# Copyright (c) 2018-2019 Katori lab. All Rights Reserved
# Author: Yuichi Katori (yuichi.katori@gmail.com)
# NOTE: サーチツール

import os
import shutil
import subprocess
import importlib
import sys
import numpy as np
import pandas as pd
import itertools
import datetime
import time
import psutil
from tqdm import tqdm
from pprint import pprint

#import matplotlib as mpl
#mpl.use('Agg')# リモート・サーバ上で図を出力するための設定
#import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix

config = None
parallel=4
exe=""
columns=[]
dir_path =  os.getcwd() # 結果を出力するディレクトリのパス，デフォルトはカレントディレクトリ
prefix = "asdf"
file_md = ""
last_csv = ""
test = None # テストコードのファイル名
dryrun = False # Trueにするとexecute()とsave_dataframe()を実行しない
seed = None
cleanup = False

### main側から呼び出す関数
#def load_config(a):
#    c = pd.read_pickle(a.config)
#    return c

def load_config(a,class_="Config"):
    """
    pickle形式または.py形式のconfigを読み込む.
    引数：
    a:argparseでの解析結果、またはモジュール名
    """
    global config # TODO 動作チェック
    if hasattr(a, 'config'):# aをargparseの結果と解釈する
        if a.config.endswith('.pkl'):
            c = pandas.read_pickle(a.config)
        else:
            #c = importlib.import_module(a.config).Config()
            module = importlib.import_module(a.config)
            c = getattr(module, class_)()
    else:# aをモジュール名と解釈する
        module = importlib.import_module(a)
        c = getattr(module, class_)()
    config = c
    return c

def save_config(c):
    # pd.to_pickle(c,a.config)
    # csv
    vx=[]
    for col in c.columns:
        vx.append(getattr(c,col))

    ss = pd.Series(vx, index=c.columns)
    df = pd.DataFrame(index=[],columns=c.columns)
    #df = df.append(ss,ignore_index=True)
    df = pd.concat([df, pd.DataFrame([ss])],ignore_index=True,axis=0)
    df.to_csv(c.csv, index=False, mode='a', header=False)

### 初期設定
def setup():
    # レポート（マークダウン）ファイルの準備
    global file_md
    file_md = prefix+".md"

    # 作業ディレクトリの準備
    global dir_path
    prepare_directory(dir_path)

    global columns
    #columns = config.columns

    # seedの設定：columnsに"seed"が含まれれば、それをseedとして扱う
    for c in columns:
        if c=="seed":
            seed="seed"
    #print("seed:",seed)

    # columnsとconfig属性の一致を確認

    is_ok=True
    for c in columns:
        is_column = False
        for k, v in config.__dict__.items():
            if c == k:
                is_column = True
        if is_column == False:
            is_ok = False
    if is_ok == False:
        print("WARNING: columns and cofig are non match.")

### ファイルとディレクトリの操作
def prepare_directory(path):
    """
    # 指定されたディレクトリがなければ作る
    """
    # ディレクトリの存在有無の確認
    if os.path.isdir(path): # ディレクトリの存在の確認
        print("Exist Directory: %s" % (path))
        pass
    else:
        print("Create Directory: %s" % (path))
        os.makedirs(path) # ディレクトリの作成

    #common.dir_path = path
    #dir_path = path

def set_path(path):
    """
    # 結果を出力するディレクトリのパスを指定する
    """
    global dir_path
    prepare_directory(path)
    dir_path = path

def string_now():
    t1=datetime.datetime.now()
    s=t1.strftime('%Y%m%d_%H%M%S')
    return s

def string_today():
    t1=datetime.datetime.now()
    s=t1.strftime('%Y%m%d')
    return s

def create_directory(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

def load_dataframe(csv=None,path=None):
    global last_csv

    if csv == None:
        csv = last_csv
        #CSVファイルが指定されない場合は、直前に保存されたCSVファイルを読み込む
    else:
        last_csv = csv
        #CSVファイルが指定された場合は、これをlast_csvに設定する。
        #このlast_csvは画像を保存する際に使用される。

    if path == None:
        file_csv_path = os.path.join(dir_path, csv)
    else:
        file_csv_path = os.path.join(path, csv)

    #print(file_csv_path)
    #print(columns)
    #print(set(columns))
    #print(len(columns),len(set(columns)))
    df = pd.read_csv(file_csv_path, sep=",", names=columns)
    #print(df)
    return df

def save_dataframe(df,file_csv,path=None):
    """
    Dataframeの内容をCSVに保存する。
    """
    global last_csv
    global dryrun

    if path == None:
        file_csv_path = os.path.join(dir_path, file_csv)
    else:
        file_csv_path = os.path.join(path, file_csv)

    last_csv = file_csv
    if dryrun:
        return None

    df.to_csv(file_csv_path, index=False, mode='a', header=False)

### レポートツール

def report(str):
    print(str)
    file_md_path = os.path.join(dir_path, file_md)
    if file_md_path != "":
        f=open(file_md_path,"a")
        f.write(str+"")
        f.close()

def report_common():
    s = "## %s\n" % (prefix)
    s+= "### Common config\n"
    s+= "```\n"
    s+= "hostname: %s\n" % (os.uname()[1])
    s+= "dir_path: %s\n" % (dir_path)
    s+= "Report  : %s\n" % (file_md)
    s+= "Test    : %s\n" % (test)
    s+= "Exe     : %s\n" % (exe)
    s+= "parallel: %s\n" % (parallel)
    s+= "```\n"
    report(s)

def report_figure(fig):
    """
    画像ファイルについてのレポート
    fig:ファイル名(パス付きでも、パスなし、どちらでも良い)
    """
    dir=os.path.split(fig)[0]
    fig=os.path.split(fig)[1]
    #print("DEBUG dir:",dir," filenme:",os.path.split(fig)[1])
    s = "Figure:** %s **  \n" % (fig)
    s+= "![](%s)  \n" % (fig)
    report(s)

def report_config(c=None):
    global config # TODO 動作確認
    if c != None:
        config = c
    
    s= "### Current Config\n"
    s+="```\n"
    for k, v in config.__dict__.items():
        s+="%8s:%s\n" % (k,v)
    s+="```\n"
    report(s)

def report_time_start():
    s= "Start:" + string_now() + "  \n"
    report(s)

def report_time_done():
    s= "Done :" + string_now() + "  \n"
    report(s)

###
def config2series(config):
    """
    config の設定をpandasのseriesに変換する。
    """
    # TODO 型の情報を保ったまま変換する。
    value=[]
    for i,column in enumerate(columns):
        value.append( getattr(config,column) )
    s = pd.Series(value,index=columns)
    return s

def series2config(cnf,s,columns):
    #print("columns in series2config:",s.columns)
    for i,col in enumerate(columns): # XXX
        #setattr(cnf,col,s[col])
        #print("debug:series2config:",col)
        if(hasattr(cnf,col)):
            a=getattr(cnf,col)
            v=s[col]
            if type(a) == type(0):# int
                v=int(v)
            if type(a) == type(0.):#float
                v=float(v)

        setattr(cnf,col,v)
    return cnf

def name_file(filename,path=True,directory=None):
    """
    作業ディレクトリで、ファイル(filename)の存在をチェックし、
    すでにファイルが存在する場合は、番号を付与して名前をつける。
    path:True 作業ディレクトリ(dir_path)へのパスをつけて返す。
    path:None 作業ディレクトリ(dir_path)へのパスをつけずに返す。
    """

    id=0
    is_need_rename = True
    while is_need_rename:

        # ファイル名を名前と拡張子に分割する。
        name= os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]

        if id==0:
            file = "%s%s" % (name,ext)
        else:
            file = "%s[%d]%s" % (name,id,ext)

        #ファイル名にpathを付与する。
        if directory == None:
            file_path = os.path.join(dir_path,file)
        else:
            file_path = os.path.join(directory,file)
        #print(file_path)

        # ファイルの存在を確認する。
        if os.path.isfile(file_path):
            #print("exist")
            id+=1 # ファイルが存在すれば、付与する番号を大きくする。
        else:
            is_need_rename = False
            #print("not exist")
        
        if dryrun:
            is_need_rename = False
        
    if path == True:
        return file_path
    else:
        return file


### Execute
def execute(cnf,df):
    """
    ベースになるconfig(cnf)を読み込み、dataframe(df)の内容を反映させて、並列実行する。
    計算結果を含むdataframeを返す。
    """
    global parallel
    global file_sh_path
    global dryrun
    global columns
    global file_tmp_path

    if dryrun:
        return None

    ### シェルスクリプト
    #file_sh = "cmd.sh"
    #file_sh_path = os.path.join(dir_path, file_sh)

    ### config と csv を保存するディレクトリを準備

    file_tmp_path = os.path.join(dir_path,"tmp")
    create_directory(file_tmp_path)

    ### 計算結果を一時的に保存するCSVファイル
    #file_csv_path = os.path.join(file_tmp_path,"tmp.csv")
    csv=name_file(cnf.csv,path=file_tmp_path)
    file_csv_path = os.path.join(file_tmp_path,csv)

    cnf.csv = file_csv_path
    cnf.columns = columns
    cnf.show = False

    ### dfの各行をconfigファイルに書き出し
    commands=[]
    for i in range(len(df)):
        s=df.iloc[i]
        #file_config="tmp/config%04d.pkl" % (i)
        file_config = os.path.join(file_tmp_path,"config%04d.pkl" % (i) )

        commands.append(exe+"-config %s" % (file_config) )
        cnf1 = series2config(cnf,s,df.columns)

        #print("x1",cnf1.x1,type(cnf1.x1))#XXX
        #print("x2",cnf1.x2,type(cnf1.x2))#XXX

        #print("cnf1.csv",cnf1.csv)
        pd.to_pickle(cnf1,file_config)

    # コマンドを[parallel]個に分割して、新たにリストを作る。
    n = parallel
    if parallel == 0:
        n=1
    commands = [commands[idx:idx + n] for idx in range(0,len(commands), n)]
    #print(len(commands))
    #print(commands)

    ### コマンドをシェルスクリプトに書き出し、それを実行する。
    ### 実行結果はconfigファイルに書き込まれる　（※これは古い設定）
    ### 実行結果はcsvファイルに書き込まれる
    #for cmd in tqdm(commands):
    for i,cmd in enumerate(commands):
        #print(command)
        procs=[]
        for c in cmd:
            proc=subprocess.Popen([c],shell=True)
            procs.append(proc)
        for proc in procs:
            proc.communicate()

    """
    ### configファイルの読み込み
    #print("load config...")
    for i in range(len(df)):
        file_config = os.path.join(file_tmp_path,"config%04d.pkl" % (i) )
        cnf=pd.read_pickle(file_config)
        s=config2series(cnf) # TODO この変換に時間を要している。時間を短縮したい。
        df.iloc[i]=s
    """
    ### csv ファイルの読み込み
    df1 = pd.read_csv(file_csv_path, sep=",", names=df.columns)
    #print("df1:(in common.execute)\n",df1)

    ### 不要なファイルの削除
    """
    ## TODO XXX: ファイルを削除するときに
    ## BlockingIOError: [Errno 11] Resource temporarily unavailable:
    ## のエラーが出て停止することがある。
    ## owncloud(davfsで接続されている)でファイル削除が禁止されることに起因する。
    """

    if cleanup:
        #print("remove...")
        if os.path.exists(file_csv_path): os.remove(file_csv_path)
        if os.path.exists(file_sh_path): os.remove(file_sh_path)
        for i in range(len(df)):
            file_config = os.path.join(file_tmp_path,"config%04d.pkl" % (i) )
            if os.path.exists(file_config):
                os.remove(file_config)

    #shutil.rmtree(file_tmp_path) # tmp ディレクトリを削除

    return df1

def clean():
    if cleanup:
        shutil.rmtree(file_tmp_path) # tmp ディレクトリを削除

### others
def wait_cpu_free(percent=5,sec=60,interval=60,first=1):
    """
    CPU負荷がpercentより大きいときには待つ。
    CPU負荷がintervalの期間percentを下回れば終了して、次の処理を実行。
    """
    
    if sec!=60:
        interval=sec # 互換性を保つためにSecによる設定を残す。いずれ削除

    cpu_load = psutil.cpu_percent(interval=first)
    if cpu_load > percent:
        t1=datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        #print("cpu_load:",cpu_load)
        print("wait_cpu_free: wait cpu load get lower than ",percent,"% for",interval,"seconds. (",t1,")")
    else:
        return None

    while True:
        cpu_load = psutil.cpu_percent(interval=interval)
        #print("cpu_load:",cpu_load,"interval:",interval)
        if cpu_load < percent:
            return None
