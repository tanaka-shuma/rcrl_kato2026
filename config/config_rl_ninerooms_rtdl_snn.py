# Copyright (c) 2025 Katori lab. All Rights Reserved
# 強化学習タスク: 9つの部屋(rl_ninerooms) 
# エージェント: レザバーTD学習(RTDL)
# レザバー: SpikingNeuralNetwork
from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    ### 共通
    id: int = None
    plot: bool = True # 図の出力
    show: bool = True # 図の表示
    savefig: bool = True # 図の保存
    fig1: str = "fig1.png" # 画像ファイル名
    seed: int = 1 # 乱数生成のシード
    episode_dict: bool = False # エピソードごとの結果を保存するかどうか (False:保存しない, True:保存する)
    plot_dynamics: bool = False # リザバー状態を描画するかどうか (False:描画しない, True:描画する)
    plot_orbit: bool = False #ロボット軌道を描画するかどうか (False:描画しない, True:描画する)

    ### agent: レザバーTD学習(RTDL)

    model_module: str = "models.agent_rtdl"
    model_class: str = "Agent"
    
    Nu: int = 8 # ノード数（入力・状態の数）
    Ny: int = 3 # ノード数（出力・行動の数）
    alpha_i: float = 0.5 # 結合強度（入力）
    alpha_b: float = 0.5 # 結合強度（フィードバック）
    beta_i: float = 0.05 # 結合率（入力）
    beta_b: float = 0.05 # 結合率（フィードバック）

    gamma: float = 0.9 # TD学習の割引率

    eps_greedy = True # epsilon-greedy (False: 使用しない, True: 使用する)

    # eta1() eta2()
    eta_init: float  = 0.01 # 学習率の初期値
    eta_final: float = 0.00 # 学習率の最小値
    eta_tau: float = 1000 # 学習率の時間減衰時定数
    eta_decay: float = 4 # 学習率の報酬減衰定数

    # epsilon1() epsilon2()
    eps_init: float = 0.1 
    eps_final: float = 0.01
    eps_tau: float = 100
    eps_decay: float = 1000

    ### env:rl_nineroom
    env_module: str = "tasks.rl_ninerooms"
    env_class: str = "RobotRoom"

    is_render: bool = False # ゲーム画面の描画
    num_episodes: int = 300  # エピソードの回数 (300)
    max_timesteps: int = 200 # 1エピソード中の最大時間ステップ数

    ### reservoir: Spiking Neural Network リザバー
    rc_module: str = "models.snn_reservoir" 
    rc_class: str = "SpikingNeuralNetwork"

    # 時刻
    dt: float = 0.2            # 物理系の単位時間ステップ（ms）
    ds: float = 50             # リザバー単位時間ステップ（ms）

    ### ニューロン
    Nx: int = 500              # ニューロン数
    Irate: float = 0.2         # 抑制性ニューロンの割合
    tca: int = 200             # 細胞内Caイオン濃度の時定数 (ms) 
    const = 2                  # 入力電流乗算定数

    ### シナプス
    # ネットワーク構造
    m_base: int = 2            # モジュールの行数（列数）
    p_intra: float = 0.25      # モジュール内の結合確率
    p_inter: float = 0.01      # モジュール間の結合確率

    # 時間遅れの範囲 (ms)
    delay_e_min: float = 1.0   # 興奮性結合 最小遅延
    delay_e_max: float = 5.0   # 興奮性結合 最大遅延
    delay_i_min: float = 0.6   # 抑制性結合 最小遅延
    delay_i_max: float = 1.0   # 抑制性結合 最大遅延
    delay_e_inter_min: float = 0.6  # 興奮性・モジュール間結合 最小遅延
    delay_e_inter_max: float = 1.4  # 興奮性・モジュール間結合 最大遅延
    delay_i_inter_min: float = 0.2  # 抑制性・モジュール間結合 最小遅延
    delay_i_inter_max: float = 0.6  # 抑制性・モジュール間結合 最大遅延
    delay_e_intra_min: float = 0.0  # 興奮性・モジュール内結合 最小遅延
    delay_e_intra_max: float = 0.6  # 興奮性・モジュール内結合 最大遅延
    delay_i_intra_min: float = 0.0  # 抑制性・モジュール内結合 最小遅延
    delay_i_intra_max: float = 0.2  # 抑制性・モジュール内結合 最大遅延

    # シナプスコンダクタンス（単位は μS）
    g_EE: float = 0.003      # プレが興奮性でポストが興奮性のシナプスコンダクタンス最大値
    g_EI: float = 0.003      # プレが興奮性でポストが抑制性のシナプスコンダクタンス最大値
    g_IE: float = 0.006      # プレが抑制性でポストが興奮性のシナプスコンダクタンス最大値
    g_II: float = 0.006      # プレが抑制性でポストが抑制性のシナプスコンダクタンス最大値
    ge_tau: float = 2        # 興奮性ニューロンの時定数
    gi_tau: float = 5        # 抑制性ニューロンの時定数

    # OUノイズ（単位は nA）
    ou_mu_e: float = 0.35      # 興奮性定常電流
    ou_mu_i: float = 0.25      # 抑制性定常電流
    ou_sigma_e: float = 0.2    # 興奮性ニューロンのノイズ強度
    ou_sigma_i: float = 0.1    # 抑制性ニューロンのノイズ強度
    ou_tau_e: float = 5.3      # 興奮性ニューロンの時定数
    ou_tau_i: float = 5.3      # 抑制性ニューロンの時定数

    ### 指数関数フィルター
    filter: str = "Single" # {Single, Double}
    ex_max: float = 1.0   # 指数関数フィルターの上がり幅（1で固定）
    # NOTE Exp: 指数関数フィルター
    ex_tau: float = 200   # 指数関数フィルターの時定数
    # NOTE DoubleExp: 二重指数関数フィルター
    tr: float = 10    # 立ち上がり時定数
    td: float = 100   # 減衰時定数

    ca_mu =0
    ca_sigma = 0

    ### Results
    total_reward: float = None
    mean_reward: float = None
    avg_spike_rate: float = None
    spike_rate= None
    I_spike_rate = None
    E_spike_rate = None
    mean_lv = None
    mean_erank = None
    erank_bin_size = 20