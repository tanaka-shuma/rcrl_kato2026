# Copyright (c) 2021-2022 Katori lab. All Rights Reserved
import numpy as np 
import sympy as sp
import pickle
import os
import matplotlib.pyplot as plt 
import scipy.stats as st

DATASETS = ["Legendre","Hermite","Chebyshev","Laguerre"]
DATADIR = "./datasets/prediction/"

def generate_lorentz(T,tau1=1):
    """
    Lorentz system
    T:length of data
    """
    x,y,z = 10,0,0
    sigma,r,b = 10, 28, 8.0/3.0
    dt = 0.01
    sampling_interval = 2
    T0 = 1000

    D = np.zeros((T+T0,3))
    i = 0
    n = 0
    while n < T+T0:
        _x = x + (-sigma*x + sigma*y)*dt/tau1
        _y = y + (-x*z + r*x-y)*dt/tau1
        _z = z + (x*y - b*z)*dt/tau1
        # update
        x,y,z = _x,_y,_z
        if i % sampling_interval == 0:
            D[n, :] = x, y, z
            n += 1
        i += 1
    return D[T0:]

def generate_dataset(T=10000,data="lorentz",mode="load",is_save=True):
    """
    データセットの生成(保存)・読込
    データのファイルが
    引数:
    T: 生成する時系列の長さ
    data:データの種類: {"lorentz"}
    mode:{load,generate}
        load: 
    """
    filename = os.path.join(DATADIR, f"{data}.pickle")
    is_file = os.path.isfile(filename)

    if mode=="generate" or not is_file:
        print(f"generating...{data}")

        if data=="lorentz":
            d = generate_lorentz(T)

        if is_save:
            save_dataset(data,d)

    elif mode=="load":
        d = load_dataset(data)

    else:
        raise ValueError("'mode' shoule be {load,generate}")

    return d

def save_dataset(data,d,is_debug=False):
    if not os.path.isdir(DATADIR):os.makedirs(DATADIR)
    filename = os.path.join(DATADIR, f"{data}.pickle")
    with open(filename, mode='wb') as f:
        pickle.dump(d,f)
    
def load_dataset(data,is_debug=False):
    filename = os.path.join(DATADIR, f"{data}.pickle")
    with open(filename, mode='rb') as f:
        d = pickle.load(f)
    return d

if __name__=="__main__":
    d = generate_dataset(T=300000,mode="generate")
    print(d)
    print(d.shape)
    plt.plot(d)
    plt.show()
