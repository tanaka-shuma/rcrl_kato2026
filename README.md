# Reservoir Computing based Reinforcement Learning
著者：香取勇一

このリポジトリは、リザバー強化学習の性能解析を行うためのPythonスクリプトです。

## 使い方

### 準備

Ubuntu上のAnaconda（3.x）での動作を想定しています。
強化学習タスクを使用する場合は pygame をインストールします。
```
$ pip install pygame --user
```
### 基本的な仕組み

メインのファイル(main_*.py)を実行します。実行の際には、諸々のパラメータ値を記述した設定ファイル(config_*.py)を読み込み、スクリプトを実行します。オプション `-c` で設定ファイルを指定することができます。オプションを設定しない場合はデフォルトの設定ファイルが読み込まれます。
メインの内部では、環境のインスタンス(env)とエージェントのインスタンス(agent)を生成します。agentの内部でさらにリザバーのインスタンス(reservoir)を生成します。各インスタンスは、設定ファイルにより初期化され、メインの内部のメインループが実行されます。


### 実行
次のようにスクリプトを実行します。

#### 強化学習 9つの部屋(rl_ninerooms) レザバーTD学習(RTDL) 離散時間 Leaky integrator(lid) 
```
$ python main_rl.py
```

#### 強化学習 9つの部屋(rl_ninerooms) レザバーTD学習(RTDL) 連続時間 Leaky integrator(lic) 
 
物理リザバー、ODEを差分方程式にして実装する例
```
$ python main_rl.py -c config/config_rl_ninerooms_rtdl_lic.py
```

#### 強化学習 9つの部屋(rl_ninerooms) レザバーTD学習(RTDL) Spiking neuron model(snn)

```
python main_rl.py -c config/config_rl_ninerooms_rtdl_snn.py
```

## ファイルとディレクトリの構成

### メインのスクリプト
設定情報は common_configurator モジュールを用いて管理し、
設定情報はそれぞれのスクリプトに対応した設定ファイルから読み込まれます。
スクリプトはコマンドラインから実行でき、設定ファイルを引数で指定することも可能です。

* common_configurator.py: 設定ファイルの保存・読込、モデルの保存・読込に関わる関数を提供します。
* main_rl.py: 強化学習タスク
* main_narma.py: NARMAタスクを行います。

### models/
モデル関連のファイルが格納されています。

* agent_rtdl.py 強化学習エージェント: レザバーTD学習(RTDL)
* esn1.py 基本的なESN (フィードバックなし)
* leaky_integrator_continuous.py 
リーク付き積分モデル（連続時間）、物理レザバー、連続時間モデル（ODE）で記述されるモデルを差分方程式で実装する例
* leaky_integrator_discrete.py リーク付き積分モデル（離散時間）

* matrix_generator.py ランダム行列の生成
* labbench.py 物理リザバーのためのモジュール、時間遅れ、パルスによる刺激、計測値の時間平均
* metrics.py 性能評価の指標
* plotting.py plotに関わる関数
* utilities.py リザバー用ツール、Ridge回帰、ラベル設定

### tasks/
タスク関連のファイルが格納されています。
* rl_ninerooms.py  強化学習９つの部屋タスク

### config/
設定ファイルが格納されています。

* config_rl_ninerooms_rtdl_lid.py 
強化学習タスク: 9つの部屋(rl_ninerooms) 
エージェント: レザバーTD学習(RTDL)
レザバー: 離散時間 Leaky integrator(lid) 

* config_rl_ninerooms_rtdl_lic.py 
強化学習タスク: 9つの部屋(rl_ninerooms) , 
エージェント: レザバーTD学習(RTDL), 
レザバー: 連続時間 Leaky integrator(lic) 

* config_narma_esn1.py NARMAタスク, ESN

* config_narma_leaky_integrator.py NARMAタスク, leaky_integrator モデル（物理レザバーのシミュレーション）

### テストスクリプト
* test_narma_esn1.py: NARMAタスク+esn1モデルのテストを行います。

### dataset/
タスクで使用するデータセットが格納されています。

## その他

#### NARMAタスク, ESNモデル
```
$ python main_narma.py
```
#### NARMAタスク, Leaky_integrator モデル
物理リザバー、ODEを差分方程式にして実装する例
```
$ python main_narma.py -c config/config_narma_lic.py 
```



