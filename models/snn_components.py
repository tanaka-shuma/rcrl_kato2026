"""
* FrenchによるCa依存電流を含むLIFモデル
* 指数減衰シナプス
* FIFO(先入れ先出し)型のキューによるシナプス信号伝達の時間遅れ
* OUノイズ
"""

import numpy as np

class frenchLIF:
    """
    FrenchによるCa依存電流を含むLIFモデル
    """
    def __init__(self, N, n_type, tca=200, dt=0.2):
        ### 単位系： 時間:ms, 電圧:mV, 電流:nA, 抵抗:MΩ, コンダクタンス:μS, Caイオン濃度:μM
        self.N = N                      # ニューロン数
        self.neuron_type = n_type       # ニューロンのタイプ(興奮性・抑制性)
        self.dt = dt                    # time step
        self.Rine = 40                  # 入力抵抗（興奮性）
        self.Rinh = 50                  # 入力抵抗（抑制性）
        self.tref = 5                   # 絶対不応期
        self.tref2 = 12                 # 不応期電流時定数
        self.tmeme = 20                 # 細胞膜時定数（興奮性）
        self.tmemh = 10                 # 細胞膜時定数（抑制性）
        self.tca = tca                  # Ca時定数
        self.vreste = -74               # 静止膜電位（興奮性）
        self.vresth = -70               # 静止膜電位（抑制性）
        self.vreset = -60               # リセット電位
        self.vthe = -54                 # 閾値電位（興奮性）
        self.vthh = -50                 # 閾値電位（抑制性）
        self.vpeak = -30                # ピーク電流
        self.Ek = -75                   # K電流反転電位
        self.gk = 0.01                  # K電流コンダクタンス
        self.gref = 0.15                # 不応期電流コンダクタンス
        self.Cstep = 0.1                # Ca単位流入量

    def initialize_states(self, random_state=False): 
        ### 変数を初期化
        self.tcount = 0
        self.v = np.ones(self.N)*self.vreset
        self.c = np.zeros(self.N)
        self.spike = np.zeros(self.N)
        self.tlast = np.zeros(self.N)

        ### 膜電位の初期値をランダムに設定
        if random_state:
            self.v = self.vreset + np.random.rand(self.N)*(self.vth-self.vreset)

        ### パラメータを設定、１次元配列として設定している。
        nt = self.neuron_type
        self.vrest = np.zeros(self.N); self.vrest[nt==0] = self.vreste; self.vrest[nt==1] = self.vresth; 
        self.Rin = np.zeros(self.N); self.Rin[nt==0] = self.Rine; self.Rin[nt==1] = self.Rinh; 
        self.tmem = np.zeros(self.N); self.tmem[nt==0] = self.tmeme; self.tmem[nt==1] = self.tmemh; 
        self.vth = np.zeros(self.N); self.vth[nt==0] = self.vthe; self.vth[nt==1] = self.vthh 
        ### NOTE 必要あれば、各ニューロンのパラメータに分布を持たせる。
    
    def step(self, Iext):
        t = self.tcount*self.dt
        ### 発火の処理
        self.spike = 1*(self.v >= self.vth)                           # スパイク生成
        self.tlast = self.tlast*(1-self.spike) + t*self.spike         # 最終発火時刻を記録
        self.v = self.v*(1-self.spike) + self.vpeak*self.spike        # 発火したらvpeakまで膜電位を上げる
        self.v = self.v*(1-self.spike) + self.vreset*self.spike       # 発火したら膜電位をリセット
        self.c += self.Cstep*self.spike                               # 発火したらカルシウムイオン濃度を上げる
        
        ### 入力電流の定義
        self.Ica = self.gk*self.c*(self.Ek-self.v)                                                  # カルシウムイオン濃度に依存するカリウムイオン電流
        self.Iref = -self.gref/(1+(t - self.tlast)/self.tref2)*(self.v-self.vreset)*(self.tlast>0)  # 相対不応期電流
        self.Itot = Iext  + ( self.Iref + self.Ica )*(self.neuron_type==0)                          # 入力電流

        ### 膜電位の変化
        self.dv = (self.vrest - self.v + self.Rin*self.Itot) / self.tmem                # 膜電位の変化量
        self.v = self.v + (self.dt*self.tcount>(self.tlast+self.tref))*self.dv*self.dt  # 不応期でなければ膜電位を変化

        ### カルシウムイオン濃度の変化
        self.c -= self.c/self.tca*self.dt

        self.tcount += 1

        return self.spike


class SingleExponentialSynapse: 
    """
    指数減衰シナプス
    """
    def __init__(self, N, dt=0.2, tau=5.0):
        self.N = N
        self.dt = dt
        self.tau = tau
        self.r =np.zeros(N)
    def initialize_states(self):
        self.r=np.zeros(self.N)
    def __call__(self, spike):
        self.r = self.r*(1-self.dt/self.tau) + spike
        return self.r
    

class DoubleExponentialSynapse:
    def __init__(self, N, dt=0.2, td=5, tr=2):
        self.N = N
        self.dt = dt
        self.td = td
        self.tr = tr
        self.rmax = 1 / ((1/td) * (tr/td)**(tr/(td-tr)))
        self.r = np.zeros(N)
        self.hr = np.zeros(N)

    def initialize_states(self):
        self.r = np.zeros(self.N)
        self.hr = np.zeros(self.N)

    def __call__(self, spike, ex_max):
        self.r = self.r*(1 - self.dt/self.td) + self.hr * self.dt
        self.hr = self.hr * (1 - self.dt/self.tr) + (spike / (self.tr*self.td)) * self.rmax * ex_max

        return self.r

class DelayConnection:
    """
    FIFO(先入れ先出し)型のキューによるシナプス信号伝達の時間遅れ
    """
    ### TODO: FIFOのキュー(queue)で実装
    def __init__(self,N,dt,c_pre,c_post,c_delay,c_g):
        self.c_pre = c_pre.astype(int)                        # 各結合のプレニューロンのインデックス
        self.c_post = c_post#.astype(int)                     # 各結合のポストニューロンのインデックス
        self.c_delay = c_delay                                # 各結合の時間遅れ
        self.c_g = c_g                                        # 各結合の強度（コンダクタンス）
        max_delay = np.max(c_delay)                           # 時間遅れの最大値
        max_delay_step = int(np.round(max_delay/dt)) + 1      # 時間遅れの時間ステップ数の最大値
        self.state = np.zeros((N, max_delay_step))            # ポスト側にくる信号を保持するキュー 
        self.c_delay_step = np.round(c_delay/dt).astype(int)  # 各結合の時間遅れの時間ステップ

    def __call__(self,fire):
        out = self.state[:,0]                           # 1:1列目を出力
        self.state = np.roll(self.state, -1, axis=1)    # 2:列を左方向にずらす
        self.state[:,-1] = 0                            # 3:最終列の全要素を0にリセット(np.rollの後処理)

        fire_index, = np.where(fire > 0)                # 発火したプレニューロンのインデックスを取得
        for i_pre in fire_index:                        # 発火したプレニューロン毎のループ処理
            c_index, = np.where(self.c_pre == i_pre)    # 発火したプレニューロンの持つ結合のインデックスを取得 
            for ic in c_index:                          # 発火状態の結合についてループ処理
                self.state[self.c_post[ic], self.c_delay_step[ic]] += self.c_g[ic]  # キューに追加

        return out     

class OUNoise:
    """
    OUノイズ
    mu:平均値、tau:時定数、sigma:ノイズ強度, sd:標準偏差
    NOTE:sdの値を設定するとsigmaの値を上書きする。平均mu、標準偏差sdの正規分布に従う時系列を生成する。
    このsdは瞬時的に与えるノイズの強度ではないことに注意．
    """
    def __init__(self,N,mu,tau,sigma=None,sd=None,dt=0.2):
        
        self.x = np.zeros(N)
        self.N = N
        self.dt = dt
        self.sqrtdt = np.sqrt(dt)
        self.tau = tau
        self.mu = mu
        self.sigma = sigma
        if sd != None:
            self.sigma = sd * np.sqrt(2.0 / tau)
        
    def __call__(self):
        self.x += (self.mu - self.x)/self.tau*self.dt + self.sigma*np.random.randn(self.N)*self.sqrtdt
        return self.x