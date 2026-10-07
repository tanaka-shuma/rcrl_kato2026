from . import snn_components
from . import exfilter

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class SpikingNeuralNetwork:
    def __init__(self, c):
        self.c = c
        self.physical_time_steps_in_unit_reservoir_time_step = int(c.ds/c.dt) # 
        
        ### ニューロン
        self.NNi = int(c.Nx * c.Irate)
        self.inh_index = np.random.choice(range(int(c.Nx)), self.NNi, replace=False)   # 抑制性を選択//重複:×
        self.n_type = np.zeros(int(c.Nx)).astype(int); self.n_type[self.inh_index] = 1 # ニューロンのタイプ（興奮性or抑制性）抑制性1/興奮性0
        self.NNe = np.sum(self.n_type==0)                                              # 抑制性の数
        self.NNi = np.sum(self.n_type==1)                                              # 興奮性の数
        self.spike_times = [[] for _ in range(c.Nx)] 
        assert int(c.Nx) == self.NNe + self.NNi
        self.neurons = snn_components.frenchLIF(N = int(c.Nx), dt = c.dt, n_type = self.n_type, tca=c.tca)     

        ### シナプス
        self.graph = self.gen_matrix(m_row = int(c.m_base), m_col = int(c.m_base), Nx = int(c.Nx), p_intra = c.p_intra, p_inter = c.p_inter)
        self.connect = np.array(self.graph.edges()).T                  # 各シナプス結合のプレ・ポストのニューロンのインデックス
        self.n_module = np.array(self.graph.nodes.data('block')).T[1]  # 各ニューロンの所属するモジュールのインデックス
        self.NC, self.NCe, self.NCi, self.ce_pre, self.ce_post, self.ce_delay, self.ce_g, self.ci_pre, self.ci_post, self.ci_delay, self.ci_g  = \
            self.gen_synapse(c = c,connect = self.connect, n_type = self.n_type, n_module = self.n_module)

        self.exc_synapses = snn_components.SingleExponentialSynapse(N = int(c.Nx), dt = c.dt, tau = c.ge_tau)                      # 興奮性シナプス
        self.inh_synapses = snn_components.SingleExponentialSynapse(N = int(c.Nx), dt = c.dt, tau = c.gi_tau)                      # 抑制性シナプス   
        self.delay_e = snn_components.DelayConnection(int(c.Nx), c.dt, self.ce_pre, self.ce_post, self.ce_delay, self.ce_g)        # 興奮性遅延
        self.delay_i = snn_components.DelayConnection(int(c.Nx), c.dt, self.ci_pre, self.ci_post, self.ci_delay, self.ci_g)      # 抑制性遅延

        ### OU Noise
        self.ou_mu = np.zeros(int(c.Nx))
        self.ou_mu[self.n_type==0] = c.ou_mu_e
        self.ou_mu[self.n_type==1] = c.ou_mu_i
        self.ou_sigma = np.zeros(int(c.Nx))
        self.ou_sigma[self.n_type==0] = c.ou_sigma_e
        self.ou_sigma[self.n_type==1] = c.ou_sigma_i
        self.ou_tau = np.zeros(int(c.Nx))
        self.ou_tau[self.n_type==0] = c.ou_tau_e
        self.ou_tau[self.n_type==1] = c.ou_tau_i
        self.noise = snn_components.OUNoise(N = int(c.Nx), mu = self.ou_mu, sigma = self.ou_sigma, tau = self.ou_tau, dt = c.dt)   # OUノイズ

        ### 読み出し用フィルター
        if c.filter == "Single":
            self.ex_filter = exfilter.ExponentialFilter(N = int(c.Nx), dt = c.dt, tau = c.ex_tau)                                  # 単一指数関数フィルター
        if c.filter == "Double":
            self.ex_filter = exfilter.DoubleExponentialFilter(N = int(c.Nx), dt = c.dt, tau_d = c.td, tau_r = c.tr)                      # 2重指数関数フィルター

        ### その他
        self.n = 0    # ステップ数のカウント
        self.t = 0    # 時刻(ms)
    
    def gen_matrix(self, m_row, m_col, Nx, p_intra, p_inter):
        '''
        格子状のモジュール構造をもつネットワークを生成する
        m_row : モジュールの行方向の数
        m_col : モジュールの列方向の数
        n : ニューロン数
        p_intra : モジュール内の結合確率 (ニューロン毎)
        p_inter : モジュール間の結合確率 (ニューロン毎)
        '''
        m_tot = m_row * m_col                                                   #モジュール数:2*2=4
        sizes = np.array([(Nx + i) // m_tot for i in range(m_tot)])             #モジュール毎のニューロン数: n/(m_row x m_col) 96/4 ≒ 24 (余ったニューロンは可能な限り均等に分配される) //はPythonの演算子
        probs = p_inter * nx.to_numpy_array(nx.grid_graph(dim = (m_col, m_row)))  #2次元格子(m_col,m_row)を隣接行列化(m_tot,m_tot)
        probs[np.arange(m_tot), np.arange(m_tot)] = p_intra                     #グループ内の結合密度を設定(対角成分の値を決める)
        G:nx.DiGraph = nx.generators.community.stochastic_block_model(sizes, probs, directed=True, selfloops=False) #組数/結合状況/有向:○/自己結合:×
        return G

    def gen_synapse(self, c, connect, n_type, n_module):
        """
        シナプス結合の時間遅れと結合強度（コンダクタンス）
        TODO: コンダクタンスの値を分布させる
        """
        ### すべてのシナプス結合を興奮性と抑制性に分割する 
        ce_pre  = np.array([connect[0][ic] for ic,i in enumerate(connect[0]) if n_type[i]==0]) #興奮性結合についてプレのインデックスを収集
        ce_post = np.array([connect[1][ic] for ic,i in enumerate(connect[0]) if n_type[i]==0]) #興奮性結合についてポストのインデックスを収集
        ci_pre  = np.array([connect[0][ic] for ic,i in enumerate(connect[0]) if n_type[i]==1]) #抑制性結合についてプレのインデックスを収集
        ci_post = np.array([connect[1][ic] for ic,i in enumerate(connect[0]) if n_type[i]==1]) #抑制性結合についてポストのインデックスを収集

        NC = len(connect[0])    # シナプス結合の総数
        NCe = len(ce_pre)       # 興奮性シナプス結合の数
        NCi = len(ci_pre)       # 抑制性シナプス結合の数
        assert NC == NCe+NCi

        ### シナプス結合の時間遅れと結合強度（コンダクタンス）
        #TODO: コンダクタンスの値を分布させる
        #TODO: 
        ce_delay = np.zeros(NCe); ce_g = np.zeros(NCe)
        for ic in range(NCe): # 興奮性結合（プレが興奮性ニューロン）について設定
            i_pre = ce_pre[ic]
            i_post= ce_post[ic]
            ## 時間遅れ
            delay = np.random.uniform(c.delay_e_min, c.delay_e_max)
            if n_module[i_pre] == n_module[i_post]: # モジュール内結合
                delay += np.random.uniform(c.delay_e_intra_min, c.delay_e_intra_max)
            if n_module[i_pre] != n_module[i_post]: # モジュール間結合
                delay += np.random.uniform(c.delay_e_inter_min, c.delay_e_inter_max)
            ce_delay[ic] = delay
            ## コンダクタンス
            if n_type[i_post] == 0: 
                ce_g[ic] = c.g_EE
            if n_type[i_post] == 1: 
                ce_g[ic] = c.g_EI

        ci_delay = np.zeros(NCi); ci_g = np.zeros(NCi)
        for ic in range(NCi): # 抑制性結合（プレが抑制性ニューロン）について設定。
            i_pre = ci_pre[ic]
            i_post= ci_post[ic]
            ## 時間遅れ
            delay = np.random.uniform(c.delay_i_min, c.delay_i_max)
            if n_module[i_pre] == n_module[i_post]: # モジュール内結合
                delay += np.random.uniform(c.delay_i_intra_min, c.delay_i_intra_max)
            if n_module[i_pre] != n_module[i_post]: # モジュール間結合
                delay += np.random.uniform(c.delay_i_inter_min, c.delay_i_inter_max)
            ci_delay[ic] = delay
            ## コンダクタンス 
            if n_type[i_post] == 0:  
                ci_g[ic] = c.g_IE
            if n_type[i_post] == 1:  
                ci_g[ic] = c.g_II

        return NC, NCe, NCi, ce_pre, ce_post, ce_delay, ce_g, ci_pre, ci_post, ci_delay, ci_g
    

    def reset(self):
        self.neurons.initialize_states()         # flenchLIFの初期化
        self.exc_synapses.initialize_states()    # 興奮性シナプスの初期化
        self.inh_synapses.initialize_states()    # 抑制性シナプスの初期化
        self.ex_filter.initialize_states()       # 指数フィルターの初期化
        self.n = 0    # ステップ数のカウント
        self.t = 0    # 時刻(ms)
        self.spike_times = [[] for _ in range(self.c.Nx)]
        
    def step_snn(self,Iinput):
        self.I_input = Iinput

        ### 現ステップに来る刺激の配列
        de = self.delay_e(self.neurons.spike)
        di = self.delay_i(self.neurons.spike)

        ### 外部入力（入力電流、再帰電流、ノイズ電流）
        self.I_ext = ( 0 - self.neurons.v) * self.exc_synapses(de) \
                    + (-80 - self.neurons.v) * self.inh_synapses(di) \
                    + self.noise() \
                    + self.I_input

        # リザバーの内部状態の更新
        self.neurons.step(self.I_ext)     # リザバーの更新
        # self.r = self.ex_filter(self.neurons.spike)   # リザバーを指数関数フィルターにかける
        self.r = self.neurons.c     # 読み出しは細胞内Ca2+濃度
        
        # 時間ステップ
        self.n += 1
        self.t += self.c.dt

        active_neurons = np.nonzero(self.neurons.spike)[0]# スパイクを出すニューロンのインデックスを取得
        for neuron_id in active_neurons:
            self.spike_times[neuron_id].append(self.t)# スパイクのタイミングを記録

    def step(self,input):
        #リザバーの単位時間ステップ
        Iin = input * self.c.const ### 感覚入力を電流に変換 (スケーリング：const μA)
        Iin = np.where(Iin < 0, 0, Iin)
        for i in range(self.physical_time_steps_in_unit_reservoir_time_step): # physical_time_steps_in_unit_reservoir_time_step
            self.step_snn(Iin)

    def done(self):
        # エピソードの終わりに実行される処理

        #print("done t:",self.t)
        # self.plot_spike()

        #平均発火率
        num_neurons = len(self.spike_times)
        total_time_ms = self.t
        total_spikes = sum(len(times) for times in self.spike_times)
        # total_time_ms を秒に変換して、各ニューロンあたりの平均スパイク数を秒単位で割る
        self.avg_spike_rate = total_spikes / (num_neurons * (total_time_ms / 1000.0))
        if hasattr(self.c, "avg_spike_rate"): 
            self.c.avg_spike_rate = self.avg_spike_rate
        #print(avg_spike_rate)
        return None

    def plot_spike(self):
        plt.eventplot(self.spike_times, colors='black')
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron ID")
        plt.title("Raster Plot of Spike Times")
        # plt.show()
    
    def sum_spike(self, spikeE, spikeI):
        # 平均発火率のみの計算
        idx_spk, = np.array(np.where(self.neurons.spike>0))
        for i in idx_spk:
            if self.n_type[i] == 1:
                spikeI += 1
            else:
                spikeE += 1

        return spikeE, spikeI
    
    def spike_rate(self, c, sumE, sumI, nt):
        ###発火率[sp/sec]のみの場合の確認#######
        Tsim = nt*c.dt
        sumN = sumE + sumI #全ニューロンの合計の発火数
        NNe = np.sum(self.n_type==0)             #抑制性の数
        NNi = np.sum(self.n_type==1)             #興奮性の数
        
        spike_rate = sumN/(Tsim/1000)/int(c.Nx)
        I_spike_rate = sumI/(Tsim/1000)/NNi
        E_spike_rate = sumE/(Tsim/1000)/NNe

        return spike_rate, I_spike_rate, E_spike_rate