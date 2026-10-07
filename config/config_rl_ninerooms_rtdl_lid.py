# Copyright (c) 2023 Katori lab. All Rights Reserved
# 強化学習タスク: 9つの部屋(rl_ninerooms) 
# エージェント: レザバーTD学習(RTDL)
# レザバー: 離散時間 Leaky integrator(lid) 
from dataclasses import dataclass
#import numpy as np

@dataclass
class Config:
    ### 共通
    id: int = None
    plot: bool = True # 図の出力
    show: bool = True # 図の表示
    savefig: bool = True # 図の保存
    fig1: str = "fig1.png" # 画像ファイル名

    num_threads = 4 # 使用するスレッド数の上限(0に設定すると無制限)
    seed: int = 1 # 乱数生成のシード

    ### agent: レザバーTD学習(RTDL)

    model_module: str = "models.agent_rtdl"
    model_class: str = "Agent"
    
    Nu: int = 8 # ノード数（入力・状態の数）
    Ny: int = 3 # ノード数（出力・行動の数）
    alpha_i: float = 1.0 # 結合強度（入力）
    alpha_b: float = 0.8 # 結合強度（フィードバック）
    beta_i: float = 0.05 # 結合率（入力）
    beta_b: float = 0.05 # 結合率（フィードバック）

    gamma: float = 0.9 # TD学習の割引率

    # eta1() eta2()
    eta_init: float  = 0.02 # 学習率の初期値
    eta_final: float = 0.0 # 学習率の最小値
    eta_tau: float = 10000 # 学習率の時間減衰時定数
    eta_decay: float = 6 # 学習率の報酬減衰定数

    # epsilon1() epsilon2()
    eps_init: float = 0.1 
    eps_final: float = 0.02
    eps_tau: float = 1000
    eps_decay: float = 20

    ### reservoir: 離散時間 Leaky integrator(lid) 

    rc_module: str = "models.leaky_integrator_discrete" # lid
    rc_class: str = "LeakyIntegrator"

    Nx: int = 500 # ノード数（レザバー）
    alpha_r: float = 0.8 # 結合強度（レザバーのリカレント結合）
    beta_r: float = 0.05 # 結合率（レザバーのリカレント結合）
    tau_x: float = 5.0

    ### env:rl_nineroom

    env_module: str = "tasks.rl_ninerooms"
    env_class: str = "RobotRoom"

    is_render: bool = False # ゲーム画面の描画
    num_episodes: int = 300 # エピソードの回数
    max_timesteps: int = 200 # 1エピソード中の最大時間ステップ数

    ### Results

    total_reward: float = None
    mean_reward: float = None
