### いろいろとプロットするもの

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

import datetime
today=datetime.datetime.now()
yd = today.strftime("%Y-%m-%d")
hm = today.strftime("%H%M")

def trace_matrix(trace):
    if trace is None or len(trace) == 0:
        return None
    return np.asarray(trace).T

def time_axis(n, step):
    return np.arange(n) * step

def dt_trace_or_fallback(c, reservoir, trace_name, fallback_array, t):
    trace = trace_matrix(getattr(reservoir, trace_name, None))
    if trace is not None:
        n = min(trace.shape[1], int(round((t + 1) * c.ds / c.dt)))
        return time_axis(n, c.dt), trace[:, :n], c.dt

    n = min(t + 1, fallback_array.shape[1])
    return time_axis(n, c.ds), fallback_array[:, :n], c.ds

def ds_series(c, array, t):
    n = min(t + 1, array.shape[-1])
    return time_axis(n, c.ds), array[..., :n]

def format_metric(value):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"

def plot_orbit(c, x_orbit, y_orbit):
    plt.figure(figsize=(10,10))
    plt.axis('off')

    if c.env_module == "tasks.rl_ninerooms":
        walls = []
        walls.append([[50,50],[950,50],[950,950],[50,950],[50,50]])
        walls.append([[50,50],[130,50],[210,50],[210,130],[130,130],[130,210],[50,210],[50,130],[50,50]])
        walls.append([[610,50],[690,50],[690,130],[610,130],[610,50]])
        walls.append([[850,50],[950,50],[950,150],[850,150],[850,50]])
        walls.append([[320,200],[380,200],[380,320],[450,320],[450,380],[380,380],[380,390],[320,390],[320,380],[250,380],[250,320],[320,320],[320,200]])
        walls.append([[620,240],[680,240],[680,320],[780,320],[780,380],[680,380],[680,430],[620,430],[620,380],[620,320],[620,240]])
        walls.append([[890,320],[950,320],[950,380],[890,380],[890,320]])
        walls.append([[50,380],[110,380],[110,620],[160,620],[160,680],[110,680],[50,680],[50,620],[50,380]])
        walls.append([[320,590],[380,590],[380,620],[380,680],[380,740],[320,740],[320,680],[320,620],[320,590]])
        walls.append([[620,620],[680,620],[830,620],[830,680],[680,680],[680,740],[620,740],[620,680],[520,680],[520,620],[620,620]])
        walls.append([[50,850],[150,850],[150,950],[50,950],[50,850]])
        walls.append([[620,840],[680,840],[680,890],[950,890],[950,950],[680,950],[620,950],[620,890],[620,840]])
        a = 850; d = 870; e = 380; b = 320
        walls.append([[b,a],[e,a],[e,d],[b,d],[b,a]])

        # 座標の範囲を設定
        plt.xlim(0, 1000)
        plt.ylim(1000, 0)

        task_name = 'nineroom'

    if c.env_module == "tasks.rl_tmaze":
        walls = []
        walls.append([[50,50],[650,50],[650,1500],[50,1500], [50,50]])          #壁 0
        walls.append([[50,350],[250,350],[250,1500],[50,1500], [50,350]])       #壁 1
        walls.append([[450,350],[650,350],[650,1500],[450,1500], [450,350]])    #壁 2

        # 座標の範囲を設定
        plt.xlim(0, 1550)
        plt.ylim(1550, 0)

        task_name = 'tmaze'

    # 壁の描画
    for wall in walls:
        x, y = zip(*wall)
        plt.plot(x, y, color='black', linewidth=2)

    # 軌道の描画
    for i in range(c.num_episodes):
        # 実験回数に応じて色を緑→赤に変化させる
        # (num_experiments - 1) が0にならないようにmaxで保護
        denominator = max(1, c.num_episodes - 1)
        r = i / denominator
        g = 1.0 - r
        b = 0.0
        color = (r, g, b)
        
        plt.plot(x_orbit[i], y_orbit[i], color=color)

    make_dir_path = 'reservoir_plot/' + yd + '-' + hm
    
    if os.path.isdir(make_dir_path):
        pass
    else:
        os.makedirs(make_dir_path)

    plt.savefig(make_dir_path + '/orbit_'+task_name+'_'+yd + '-' + hm+'.eps')
    plt.close()

def plot_dynamics(c, t, episode, spike_rate, I_spike_rate, E_spike_rate, I_input_array, I_ext_array, v_array, 
         r_array, q_array, sum_reward_, spike_times, reservoir=None, lv=None, erank=None):
    spike_rate = round(spike_rate, 4) if np.isfinite(spike_rate) else np.nan
    I_spike_rate = round(I_spike_rate, 4) if np.isfinite(I_spike_rate) else np.nan
    E_spike_rate = round(E_spike_rate, 4) if np.isfinite(E_spike_rate) else np.nan
    sum_reward_ = round(sum_reward_, 3)
    plt.figure(figsize=(7,12))
    max_time_ms = (t + 1) * c.ds
    metric_title = ""
    if lv is not None or erank is not None:
        metric_title = ', \n LV = '+format_metric(lv)+', erank = '+format_metric(erank)

    plt.subplot(6,1,1)
    plt.title('episode : '+str(episode+1)+', sum_reward : '+str(sum_reward_)+metric_title+',\n spike_rate = '+str(spike_rate) +', I_spike_rate = '+str(I_spike_rate)+ ', E_spike_rate = '+str(E_spike_rate))
    I_input_time, I_input_plot = ds_series(c, I_input_array, t)
    plt.plot(I_input_time, I_input_plot.T)
    plt.xlim(0, max_time_ms)
    # plt.xlabel('Time (ms)')
    plt.ylabel('Input(t)')

    plt.subplot(6,1,2)
    I_ext_time, I_ext_plot, _ = dt_trace_or_fallback(c, reservoir, "Iext_trace", I_ext_array, t)
    plt.plot(I_ext_time, I_ext_plot.T)
    plt.xlim(0, max_time_ms)
    # plt.xlabel('Time (ms)')
    plt.ylabel('I_ext(t)')

    plt.subplot(6,1,3)
    v_time, v_plot, _ = dt_trace_or_fallback(c, reservoir, "v_trace", v_array, t)
    plt.plot(v_time, v_plot.T)
    plt.xlim(0, max_time_ms)
    # plt.xlabel('Time (ms)')
    plt.ylabel('Membrane \n potential (mV)')

    plt.subplot(6,1,4)
    plt.eventplot(spike_times)
    plt.xlim(0, max_time_ms)
    # plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')

    plt.subplot(6,1,5)
    r_time, r_plot, _ = dt_trace_or_fallback(c, reservoir, "c_trace", r_array, t)
    plt.plot(r_time, r_plot.T)
    plt.xlim(0, max_time_ms)
    # plt.xlabel('Time (ms)')
    plt.ylabel('Calcium ion\n concentration')

    plt.subplot(6,1,6)
    q_time, q_plot = ds_series(c, q_array, t)
    plt.plot(q_time, q_plot.T)
    plt.xlim(0, max_time_ms)
    plt.xlabel('Time (ms)')
    plt.ylabel('Output q(t)')

    
    make_dir_path = 'reservoir_plot/' + yd + '-' + hm

    if os.path.isdir(make_dir_path):
        pass
    else:
        os.makedirs(make_dir_path)

    title = 'episode:'+str(episode+1)+'-sum_reward:'+str(sum_reward_)

    plt.savefig(make_dir_path +'/'+ str(episode+1)+'.png')
    plt.close()