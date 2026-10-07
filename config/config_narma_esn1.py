# Copyright (c) 2023 Katori lab. All Rights Reserved
# NARMAタスク ESN
from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    ### フィールド（クラス変数）の定義。各フィールドは型注釈を持つ
    id: int = None
    plot: bool = True # 図の出力
    show: bool = True # 図の表示
    savefig: bool = True # 図の保存
    fig1: str = "fig1.png" # 画像ファイル名

    ### model:esn1
    model_module: str = "models.esn1"
    model_class: str = "ESN"
    
    seed: int = 1 # 乱数生成のためのシード
    NT: int = 1000 # 時間ステップ数
    NTtrans: int = 200 # 過渡状態の時間ステップ数（学習・評価の際にはこの期間のデータを捨てる）
    
    Nu: int = 1 # ノード数（入力）
    Nx: int = 200 # ノード数（レザバー）
    Ny: int = 1 # ノード数（出力）
    
    alpha_i: float = 0.2 # 結合強度（入力）
    alpha_r: float = 0.8 # 結合強度（レザバーのリカレント結合）
    alpha_b: float = 0. # 結合強度（フィードバック）

    beta_i: float = 0.95 # 結合率（入力）
    beta_r: float = 0.15 # 結合率（レザバーのリカレント結合）
    beta_b: float = 0. # 結合率（フィードバック）

    log10_lambda_ridge = -8
    lambda_ridge: float = np.power(10,0.5) # Ridge回帰の正則化パラメータ

    ### task:narma
    delay: int = 9

    ### Results
    RMSEtrain: float = None
    RMSEtest: float = None
    NMSEtrain: float = None
    NMSEtest: float = None
