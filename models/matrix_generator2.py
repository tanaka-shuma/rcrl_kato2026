# Copyright (c) 2022 Katori lab. All Rights Reserved
import numpy as np

def assign_matrix_with_1d_array(W: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    行列Wに配列Vの要素を配置する。
    
    """
    k=0
    for j in range(W.shape[1]):
        for i in range(W.shape[0]):
            if k < len(V):
                W[i,j] = V[k]
                k += 1
    return W

def generate_random_matrix(Nrow,Ncol,alpha,beta,distribution="one",normalization="sd",row_sum_zero=True,diagnal=True,is_debug=False):
    """
    ランダムに行列を生成する。結合率（beta）の割合で非ゼロの値をランダムに割り当て、
    規格化の後、係数(alpha)をかけて出力する。
    Nrow:行数
    Ncol:列数
    alpha:スケールパラメータ
    beta: 結合率
    dist: ランダムに割り当てる値の分布 {one, normal, uniform}
        one: １または−１
        normal: 正規分布
        uniform: 一様分布
    normalization: 正規化{none,sr,sd}
        none: なにもしない。
        sr: スペクトル半径（最大固有値）で規格化
        sd: 列方向の和の分散が１になるように規格化
    diagnal: 
        Falseの場合は対角要素を0にする
    """
    assert beta <= 1.0

    W = np.zeros((Nrow,Ncol))
    Nc = int(Nrow * Ncol * beta) # num of connections
    Nc2 = (Nc-(Nc%2))//2 # Ncを超えない最大の偶数を2で割った値

    if distribution == "one":    
        #V = np.ones(Nc)
        V = np.random.choice([-1, 1], size=(Nc))
        var = 1

    if distribution == "normal":
        V = np.random.normal(0,1, Nc)
        var = 1

    if distribution == "uniform":
        V = np.random.uniform(-1,1, Nc)
        var = 1/3
    
    if row_sum_zero:# 各行の和が0になるよう配置
        W = assign_matrix_with_1d_array(W,V[:Nc2])
        W = np.roll(W, Nc2//Nrow+1, axis=1)# 列方向にシフト
        W = assign_matrix_with_1d_array(W,-V[:Nc2])
    else:
        W = assign_matrix_with_1d_array(W,V)

    if is_debug:
        print(f"Nrow:{Nrow} Ncol:{Ncol} Nc:{Nc} Nc2:{Nc2}")
        print(W)

    np.apply_along_axis(np.random.shuffle, axis=1, arr=W)#各行内で要素をシャッフル
    np.random.shuffle(W)# 各行をシャッフル

    if not diagnal:
        for i in range(Nrow):
            if W[i,i] != 0: # 対角要素が0でない場合
                zero_indices = np.argwhere(W[i] == 0) #行iで値が0の要素のインデックスを抽出 
                j = np.random.choice(zero_indices.T[0]) # 値が0の要素をランダムに選択
                W[i,j],W[i,i] = W[i,i],W[i,j] # スワップする

    #print(W)
    # spectral radium (sr) 最大固有値による規格化
    if normalization == "sr":
        assert Nrow == Ncol, "Nrow and Ncol should be same for spectral radius normalization"
        done = False
        if not done:
            v = np.linalg.eigvals(W)
            lambda_max = max(abs(v))
            if lambda_max>0:
                W = W / lambda_max
                done = True
            else:
                #NOTE 最大固有値が０の場合はシャッフルして再計算
                np.random.shuffle(W)

    # standard deviation (sd) 標準偏差による規格化
    # NOTE: 分散 var の分布から取得したm個のサンプルの和の分散は m * var となる。
    # 列方向の和の分散が１になるように規格化
    if normalization == "sd":
        W = W / np.sqrt(var * beta * Ncol)

    W = W * alpha

    return W

if __name__ == '__main__':
    print("asdf")
    W=generate_random_matrix(Nrow=10,Ncol=10,alpha=1,beta=0.3,distribution="one",normalization="sd",
                             row_sum_zero=True,diagnal=False,is_debug=True)
    #print(W)