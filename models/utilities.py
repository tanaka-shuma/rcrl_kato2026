# Copyright (c) 2022 Katori lab. All Rights Reserved
# リザバー用ツール、Ridge回帰、ラベル設定
import numpy as np

### 
def compute_ridge_regression(X, D,lambda_):
    """
    Ridge回帰で出力結合の重み行列Woutを求めて返す.lambda_=0の場合は疑似逆行列
    引数:
    X:リザバー状態 (NT,Nx)
    D:目標信号 (NT,Ny)
    lambda_: 正則化パラメータ
    返り値：
    Wo: 出力結合の重み行列(Ny,Nx)
    """
    assert len(X)==len(D)
    if lambda_ == 0: # pseudo-inverse
        Wo = D.T @ np.linalg.pinv(X).T
    else: # ridge regression
        E = np.identity(X.shape[1])
        Wo = D.T @ X @ np.linalg.inv(X.T @ X + lambda_ * E)
    return Wo

###
def compute_label(Y,D,Ndata,NTdata,Ny,method="sum"):
    """
    時系列のクラス分類：
    リザバーの出力信号（目標信号）をラベルの予測値（目標値）のone-hot-vector(OHV)に変換して返す.
    引数：
    Y: リザバーの出力（連続値の系列）、
    D: 目標信号（OHVの系列）、
    Ndata: データの数(発話の数)、
    NTdata: １データの時間ステップ数、
    Ny: 目標信号のラベル数（クラス数）、
    method: {sum,max,count_max}
    返り値：
    YC: ラベルの予測値（リザバーの出力）OHVの系列、
    DC: ラベルの目標値（正解データ）OHVの系列
    """
    assert Y.shape == (Ndata*NTdata,Ny)
    assert D.shape == (Ndata*NTdata,Ny)

    YC = np.zeros((Ndata,Ny)) # クラスのOHVを初期化
    for i in range(Ndata):
        Y1 = Y[i*NTdata:(i+1)*NTdata,:] # 出力時系列から１データ分を切り出す
        if method=="sum":
            score = np.sum(Y1,axis=0)
        elif method=="max":
            score = np.max(Y1,axis=0)
        elif method=="count_max":
            max_index = np.argmax(Y1, axis=1) # 逐次最大値をとるノード番号を取得（長さNTdataの時系列）
            score = np.bincount(max_index)    # 最大出力ノード番号のヒストグラム
        else:
            print(f"Invalid scoring method: {method}. Please choose from 'sum', 'max', or 'count_max'.")


        c_index = np.argmax(score) # スコアが最大のインデックスを抽出
        YC[i][c_index] = 1         # ラベルの予測値、OHV

    DC = D[::NTdata]# 各発話のラベルを抽出

    assert YC.shape == (Ndata,Ny)
    assert YC.shape == (Ndata,Ny)

    return YC,DC

