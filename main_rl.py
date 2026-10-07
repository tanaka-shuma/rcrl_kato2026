# Copyright (c) 2023-2025 Katori Lab. All Rights Reserved
# 強化学習タスク
import os

### 使用するCPU(スレッド数)の上限
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

### サーバー上で実行する場合
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import numpy as np
import matplotlib.pyplot as plt
import logging as log
import common_configurator as common

import someplot
from models.metrics import episode_lv_from_spike_times, episode_erank_from_spike_times
import pickle
import datetime
import uuid

c = common.load_config("config.config_rl_ninerooms_rtdl_snn")
a = common.get_arguments()# 引数で設定
if a.config: c=common.load_config(a)

log.basicConfig(level=log.INFO)
np.random.seed(int(c.seed))

# エージェントと環境のインスタンスを生成
agent = common.generate_instance(c,module=c.model_module,class_=c.model_class)
env = common.generate_instance(c,module=c.env_module,class_=c.env_class)

agent.initialize()
total_reward = 0
list_reward = []
today=datetime.datetime.now()
yd = today.strftime("%Y-%m-%d")
hm = today.strftime("%H%M")

### 結果記録用 (初期設定)
x_orbit_list = []   # 軌道のx座標
y_orbit_list = []   # 軌道のy座標
episode_list = []   # エピソード番号
total_reward_list = []  # 報酬総和
spike_rate_list = []    # 全ニューロンの発火率
I_spike_rate_list = []  # 抑制性ニューロンの発火率
E_spike_rate_list = []  # 興奮性ニューロンの発火率
goal_count = 0      # ゴールに到達した回数

### 全データをまとめるリスト (pkl保存用)
evaluation_episodes_data = []   # 各エピソードごとの評価結果
lv_episode_list = []            # 各エピソードごとのLV
erank_episode_list = []         # 各エピソードごとのeffective rank

for episode in range(c.num_episodes):
    agent.reset(episode)
    state, reward, done, info = env.reset()
    action = 0
    sum_reward = 0

    ### 記録用
    action_counts = np.zeros(c.Ny)  # 各行動の選択回数
    spikeE, spikeI = 0, 0  # 興奮性・抑制性のスパイク数

    ### Plot記録用
    v_array = np.zeros((c.Nx, c.max_timesteps))
    r_array = np.zeros((c.Nx, c.max_timesteps))
    spike_array = np.zeros((c.Nx, c.max_timesteps))
    q_array = np.zeros((c.Ny, c.max_timesteps))
    ca_array = np.zeros((c.Nx, c.max_timesteps))
    I_input_array = np.zeros((c.Nx, c.max_timesteps))
    I_ext_array = np.zeros((c.Nx, c.max_timesteps))

    for t in range(c.max_timesteps):
        state, reward, done, info = env.step(action)
        action = agent.get_action(state,reward,done)
        sum_reward += reward

        action_counts[action] += 1

        if c.plot_dynamics == True:
            v_array[:,t] = agent.reservoir.neurons.v
            r_array[:,t] = agent.reservoir.r
            I_ext_array[:,t] = agent.reservoir.I_ext
            q_array[:,t] = agent.q
            ca_array[:,t] = agent.reservoir.neurons.c
            spike_array[:,t] = agent.reservoir.neurons.spike
            I_input_array[:,t] = agent.reservoir.I_input

        spikeE, spikeI = agent.reservoir.sum_spike(spikeE=spikeE, spikeI=spikeI)

        if c.is_render:
            env.render() # 環境（ゲーム画面）の描画

        log.debug("ep:%4d t:%4d done:%d r:%5.2f R:%5.2f a:%d q:%s s:%s" %
        (episode,agent.t,done,reward,sum_reward,action,agent.q,agent.s))

        if done:
            total_reward += sum_reward
            list_reward.append(sum_reward)
            log.info(f"episode:{episode:4d} sum_reward: {sum_reward:6.3f} total_reward: {total_reward:7.3f}")
            episode_list.append(episode)
            total_reward_list.append(total_reward)
            break

        env.handle_event()
    
    ### 評価&プロット
    # ロボットの軌道を保存する(全てのエピソードを保存) #
    if c.plot_orbit == True:
        x_orbit_list.append(env.x_oribit)
        y_orbit_list.append(env.y_oribit)
    
    ### 記録を更新
    spike_rate, I_spike_rate, E_spike_rate = agent.reservoir.spike_rate(c, spikeE, spikeI, t)

    c.spike_rate= spike_rate
    c.I_spike_rate = I_spike_rate
    c.E_spike_rate = E_spike_rate

    if c.plot_dynamics == True:
        E_spike_rate_list.append(E_spike_rate)
        I_spike_rate_list.append(I_spike_rate)
        spike_rate_list.append(spike_rate)

    actual_steps = t + 1    # 実際のステップ数 (RL側のステップ数: doneで抜けた時点のt+1が実際の長さ)
    episode_lv = episode_lv_from_spike_times(agent.reservoir.spike_times)   # ラスターからLVを計算
    ### ラスターからeffective rankを計算 (20msごとにbinningして行列を作成)
    # TODO(2026-07-15): ラスターではなくCa系列でerankを計算
    episode_erank = episode_erank_from_spike_times(
        agent.reservoir.spike_times,
        total_time_ms=actual_steps * c.ds,
        bin_size=getattr(c, "erank_bin_size", 20)
    )
    lv_episode_list.append(episode_lv)
    erank_episode_list.append(episode_erank)

    ### エピソードごとの評価結果
    if c.episode_dict == True:
        evaluation_episode_dict = {
            'episode_id': int(episode),
            'steps': int(actual_steps),
            'sum_reward': float(sum_reward),
            'forward_count': int(action_counts[2]),
            'right_count': int(action_counts[1]),
            'left_count': int(action_counts[0]),
            'goal_reached': bool(env.goal),
            'lv': float(episode_lv) if not np.isnan(episode_lv) else np.nan,
            'erank': float(episode_erank) if not np.isnan(episode_erank) else np.nan
        }
        
        evaluation_episodes_data.append(evaluation_episode_dict)
    
    if c.plot_dynamics == True:
        nt = np.arange(c.max_timesteps)
        someplot.plot_dynamics(c, t, episode, spike_rate, I_spike_rate, E_spike_rate, I_input_array, 
                      I_ext_array, v_array, r_array, q_array, sum_reward, agent.reservoir.spike_times, 
                      reservoir=agent.reservoir, lv=episode_lv, erank=episode_erank)

if c.plot_orbit == True:
    someplot.plot_orbit(c, x_orbit_list, y_orbit_list)

if c.plot_dynamics == True:
    def nanmean_or_nan(values):
        values = np.asarray(values, dtype=float)
        if values.size == 0 or np.all(np.isnan(values)):
            return np.nan
        return float(np.nanmean(values))
    
    summary_dir = 'reservoir_plot/' + yd + '-' + hm
    os.makedirs(summary_dir, exist_ok=True)
    summary_filename = summary_dir + '/summary.txt'
    with open(summary_filename, 'w') as f:
        f.write('Simulation summary\n')
        f.write('==================\n')
        f.write(f'ou_mu_e: {c.ou_mu_e}\n')
        f.write(f'ou_mu_i: {c.ou_mu_i}\n')
        f.write(f'ou_sigma_e: {c.ou_sigma_e}\n')
        f.write(f'ou_sigma_i: {c.ou_sigma_i}\n \n')
        f.write(f'goal_count: {goal_count}\n')
        f.write(f'total_reward: {total_reward}\n')
        f.write(f'mean_spike_rate_all: {nanmean_or_nan(spike_rate_list)}\n')
        f.write(f'mean_spike_rate_E: {nanmean_or_nan(E_spike_rate_list)}\n')
        f.write(f'mean_spike_rate_I: {nanmean_or_nan(I_spike_rate_list)}\n')
        f.write(f'mean_LV: {nanmean_or_nan(lv_episode_list)}\n')
        f.write(f'mean_erank: {nanmean_or_nan(erank_episode_list)}\n')

env.close()
c.total_reward = total_reward # 全エピソードの報酬の総和
c.mean_reward = sum(list_reward[-20:]) / 20 # 終盤20エピソードの平均報酬
c.mean_lv = float(np.nanmean(lv_episode_list)) if len(lv_episode_list) > 0 else np.nan  # 300ep平均LV
c.mean_erank = float(np.nanmean(erank_episode_list)) if len(erank_episode_list) > 0 else np.nan     # 300ep平均effective rank
if hasattr(c, "avg_spike_rate"): print("sasdf:",c.avg_spike_rate)

if a.config: common.save_config(c)

### pklに保存
if c.episode_dict == True:
    goal_count = sum(ep['goal_reached'] for ep in evaluation_episodes_data)
    goal_rate = goal_count / len(evaluation_episodes_data) if evaluation_episodes_data else np.nan

    evaluation_data_to_save = {
        'config_seed': int(c.seed),
        'num_episodes': int(c.num_episodes),
        'tca': c.tca,
        'total_reward': float(total_reward),
        'mean_lv': c.mean_lv,
        'lv_episodes': lv_episode_list,
        'mean_erank': c.mean_erank,
        'erank_episodes': erank_episode_list,
        'goal_count': int(goal_count),
        'goal_rate': float(goal_rate),
        'action_mapping': {
            0: 'left',
            1: 'right',
            2: 'forward'
        },
        'episodes': evaluation_episodes_data
    }

    save_dir = os.path.join("sim_result", yd, str(c.eps_greedy))
    os.makedirs(save_dir, exist_ok=True)
    random_str = uuid.uuid4().hex[:8]
    filename = f'sim_result_seed{c.seed}_{yd}_{hm}_{random_str}.pkl'
    filepath = os.path.join(save_dir, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(evaluation_data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
