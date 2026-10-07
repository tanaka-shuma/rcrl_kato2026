# Copyright (c) 2021-2023 Katori lab. All Rights Reserved
"""
TI46音声データからcochleagramの時系列とクラス分類のターゲット(one-hot-vector)時系列を生成する。
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import wave 
from lyon.calc import LyonCalc
from tqdm import tqdm
from datasets.config_speech5x import *

WAVEDIR = "./datasets/ti46wave/" # TI46の音声データを配置するディレクトリ
IMAGEDIR ="./datasets/images/"
DATADIR = "./datasets/"

### wave

def cut(wave):
    """
    waveデータから発話期間を切り出して返す。
    12500Hzのサンプリングを想定する。10000ステップが0.8secに対応する
    振幅の絶対値がthresholdを超える期間がcount回あった時点を発話開始とし、
    そこからlookbackステップ戻った時点をデータの起点とする。
    """
    len_output=10000 # 出力波形の長さ
    count = 10
    threshold = 200
    lookback = 2500

    start = 0
    for i in range(len(wave)):
        if np.fabs(wave[i]) > threshold:
            count -= 1
            if count==0:
                start = i-lookback
                start = max(start,0)
                end = min(len_output+start,wave.shape[0])
                break 

    #output = np.zeros(min(len_output, end-start))
    #output[:] = wave[start:end]
    output = np.zeros(len_output)
    output[:end-start] = wave[start:end]
    return output

def save_wave_image(wave,filename):
    """
    waveデータの時系列をプロットして、画像として保存する.
    """
    if not os.path.isdir(IMAGEDIR):os.makedirs(IMAGEDIR)
    file = os.path.join(IMAGEDIR, f"wave_{filename}.png")
    plt.plot(wave)
    plt.savefig(file)
    plt.clf()
    plt.close()

def load_wave(filename):
    """
    waveデータを読み込み、発話期間を切り出して返す。
    """
    file = WAVEDIR + filename + ".wav"
    with wave.open(file,mode='r') as W:
        W.rewind()
        buf = W.readframes(-1)  # read all
        wa = np.frombuffer(buf,dtype=np.int16).astype(np.float64)
        wa = cut(wa)
    return wa

def load_waves(dataset,is_save_image=False):
    """
    datasetに含まれるwaveデータを読み込み、連結して返す。
    """
    waves = np.empty((0,0))
    for d in tqdm(dataset):
        filename = d[4]
        w = load_wave(filename)
        waves = np.append(waves,w,axis=0) if not waves.shape == (0,0) else w

        if is_save_image:
            save_wave_image(w,filename)
    return waves

### cochleagram

def save_coch_image(c,filename):
    """
    コクリアグラムの時系列をプロットして、画像として保存する.
    """
    if not os.path.isdir(IMAGEDIR):os.makedirs(IMAGEDIR)
    file = os.path.join(IMAGEDIR, f"coch_{filename}.png")
    plt.imshow(c.T)
    plt.savefig(file)
    plt.clf()
    plt.close()

def generate_cochleagram(dataset,cl,is_save_image=False):
    """
    waveデータを読み込み、コクリアグラムに変換して返す
    """
    calc = LyonCalc()
    coch = np.empty((0,0))
    for d in tqdm(dataset):
        filename = d[4]
        wave = load_wave(filename)
        c = calc.lyon_passive_ear(wave, sample_rate=cl["sample_rate"], decimation_factor=cl["decimation_factor"], ear_q=cl["ear_q"], step_factor=cl["step_factor"], tau_factor=cl["tau_factor"])      
        coch = np.append(coch,c,axis=0) if not coch.shape == (0,0) else c
        #print(coch.shape)
        if is_save_image:
            save_coch_image(c,filename)
    return coch,c.shape # 連結したcochleagramと、データ１つ分のcochleagramのshapeを返す

### target

def generate_target(dataset,num_class,num_step):
    """
    分類クラスのone-hot-vector時系列を生成する。
    num_class:時系列の次元、
    num_step:1データあたりの時系列の長さ
    """
    num_data = len(dataset)
    target = np.zeros((num_data*num_step,num_class))
    start = 0
    for d in dataset:
        target[start:start + num_step,d[0]] = 1
        start += num_step            
    return target

def save_config(filename,dataset_train,dataset_test,shape,config_lyon):
    """
    データセット(読み込むデータのリスト)と設定パラメータをcsvに保存する。
    """
    f = open(filename,"w")
    #(len(dataset_train),len(dataset_test),c_shape[0],c_shape[1],num_class)
    ### 訓練データの数、テストデータの数、コクリアグラムの時間ステップ数、周波数成分の数、ターゲットの次元
    f.write("len(dataset_train),{},\n".format(shape[0]))
    f.write("len(dataset_test),{},\n".format(shape[1]))
    f.write("num_step,{},\n".format(shape[2]))
    f.write("frequency components,{},\n".format(shape[3]))
    f.write("num_class,{},\n".format(shape[4]))
    ### lyon filter
    f.write("sample_rate,{},\n".format(config_lyon["sample_rate"]))
    f.write("decimation_factor,{},\n".format(config_lyon["decimation_factor"]))
    f.write("ear_q,{},\n".format(config_lyon["ear_q"]))
    f.write("step_factor,{},\n".format(config_lyon["step_factor"]))
    f.write("tau_factor,{},\n".format(config_lyon["tau_factor"]))

    for d in dataset_train:
        line = "train,{},{},{},{},{},\n".format(d[0],d[1],d[2],d[3],d[4])
        f.write(line)

    for d in dataset_test:
        line = "test,{},{},{},{},{},\n".format(d[0],d[1],d[2],d[3],d[4])
        f.write(line)
    
    f.close()

def generate_dataset(data):
    """
    データセット（データのリスト）を作成し、waveファイルを読み込み、コクリアグラムに変換して保存する。
    data:データの名前、保存先のディレクトリ名として使われる.
    """
    print(f"generating dataset: {data}")
    ### datasetの設定
    if data not in DATASETS:
        raise ValueError("data must be in {}".format(DATASETS))
    
    config_func = eval(f"config_dataset_{data}")
    dataset_train, dataset_test, num_class, config_lyon = config_func()
    # NOTE: config_speech5x.pyに含まれるデータセット生成用関数を呼び出している

    ### waveデータの読み込み
    print(f"loading waves:")
    waves_train = load_waves(dataset_train,is_save_image=False)
    waves_test = load_waves(dataset_test)
    
    ### コクリアグラムの生成
    print(f"generating cochleagram:")
    c_shape=(0,0)
    coch_train,c_shape = generate_cochleagram(dataset_train,config_lyon,is_save_image=False)
    coch_test,c_shape = generate_cochleagram(dataset_test,config_lyon)
    
    ### クラス分類のターゲット(one-hot-vector)時系列を生成
    target_train = generate_target(dataset_train,num_class,c_shape[0])
    target_test = generate_target(dataset_test,num_class,c_shape[0])

    shape = (len(dataset_train),len(dataset_test),c_shape[0],c_shape[1],num_class)    
    print(shape)#訓練データの数、テストデータの数、コクリアグラムの時間ステップ数、周波数成分の数、ターゲットの次元
    
    ### データの出力・保存

    # 保存先のディレクトリを確認
    path = os.path.join(DATADIR, f"{data}")
    if not os.path.isdir(path):os.makedirs(path)

    # データセットの概要をCSVに保存
    filename = os.path.join(DATADIR, f"{data}/data.csv")
    save_config(filename,dataset_train,dataset_test,shape,config_lyon)

    # コクリアグラムとターゲットの時系列データをnpzに保存
    filename = os.path.join(DATADIR, f"{data}/data.npz")
    np.savez(filename,c_train = coch_train, c_test = coch_test, t_train = target_train, t_test = target_test,shape=shape)
    
    load_dataset(data)

def load_dataset(data,is_debug=False):
    """
    保存されているコクリアグラムの時系列データを読み出して返す
    """
    if data not in DATASETS:
        raise ValueError("data must be in {}".format(DATASETS))
    
    filename = os.path.join(DATADIR, f"{data}/data.npz")
    with np.load(filename) as f:
        U1 = np.array(f['c_train'])
        U2 = np.array(f['c_test'])
        D1 = np.array(f['t_train'])
        D2 = np.array(f['t_test'])
        shape = np.array(f['shape'])

    if is_debug:
        print(U1.shape,shape)
        plt.plot(U1)
        plt.show()

    return U1,U2,D1,D2,shape
