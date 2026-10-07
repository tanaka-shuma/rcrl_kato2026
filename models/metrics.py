# Copyright (c) 2023 Katori lab. All Rights Reserved
# 性能評価の指標
import numpy as np
from sklearn import metrics

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(metrics.mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return metrics.mean_absolute_error(y_true, y_pred)

def r2_score(y_true, y_pred):
    """R2 Score"""
    return metrics.r2_score(y_true, y_pred)

def nmse(y_true, y_pred):
    """Normalized Mean Squared Error"""
    mse = metrics.mean_squared_error(y_true, y_pred)
    var = np.var(y_true)
    return mse / var if var != 0 else 0

def wsr(Y,D):
    """
    Word Success Rate(WSR)を計算して返す。
    Y,Dはone-hot-vectorの系列
    """
    assert Y.shape == D.shape
    return 1 - np.sum(abs(Y-D)/2)/Y.shape[0] # Word Success Rate(WSR)

def ber(D, Y, th=0.5):
    """
    Evaluate the performance of a prediction using the Bit Error Rate (BER).
    Args:
        D (np.ndarray): The target output series.
        Y (np.ndarray): The predicted output series.
        th (float): The threshold for converting the series to binary format.
    Returns:
        BER (float): The Bit Error Rate.
    """
    assert D.shape == Y.shape, "The two series must have the same shape."

    Dbin = (D >= th).astype(int)
    Ybin = (Y >= th).astype(int)
    
    total_bits = Dbin.size
    error_bits = np.sum(Dbin != Ybin)
    BER = error_bits / total_bits
    
    return BER

def lv_from_isis(isis):
    """Local variation of consecutive ISIs, following Shinomoto et al. (2003)."""
    isis = np.asarray(isis, dtype=float)

    if isis.size < 2:
        return np.nan # ISIが2つ未満の場合、LVは定義されないためNaNを返す

    if np.any(isis <= 0):
        return np.nan # ISIが0以下の場合、LVは定義されないためNaNを返す

    return np.mean(3.0 * ((isis[:-1] - isis[1:])**2 / (isis[:-1] + isis[1:])**2))

def episode_lv_from_spike_times(spike_times_list):
    """Mean LV over neurons in one episode."""
    neuron_lvs = []
    for times in spike_times_list:
        if len(times) < 3:
            continue # ISIが2つ未満の場合、LVは定義されないためスキップ

        isis = np.diff(np.asarray(times, dtype=float)) # スパイク時刻をISIに変換
        lv = lv_from_isis(isis)  # LVを計算
        if not np.isnan(lv):
            neuron_lvs.append(lv)

    if len(neuron_lvs) == 0:
        return np.nan

    return float(np.nanmean(neuron_lvs))

def effective_rank_from_matrix(matrix):
    """Effective rank from singular values, following Roy and Vetterli (2007)."""
    matrix = np.asarray(matrix, dtype=float)
    # 行列0要素を除外
    if matrix.size == 0 or np.all(matrix == 0):
        return np.nan

    singular_values = np.linalg.svd(matrix, compute_uv=False)   # 特異値を計算
    singular_values = singular_values[singular_values > 0]      # 0 の特異値を除外
    sum_singular = np.sum(singular_values)  # 特異値の総和(l1ノルム)
    if sum_singular == 0:
        return np.nan

    p = singular_values / sum_singular  # 特異値分布を計算
    entropy = -np.sum(p * np.log(p))    # Shannon entropyを計算
    return float(np.exp(entropy))       # effective rankを返す

def episode_erank_from_spike_times(spike_times_list, total_time_ms, bin_size=20, center=True):
    """1エピソード中のスパイク時刻から、スパイク数行列を作成し、effective rankを計算"""
    if total_time_ms <= 0 or bin_size <= 0:
        return np.nan

    bins = np.arange(0, total_time_ms + bin_size, bin_size) # ビニングの境界を作成
    if len(bins) < 2:
        return np.nan

    spike_counts = np.zeros((len(spike_times_list), len(bins) - 1), dtype=float)    # スパイク数行列を初期化
    # スパイク時刻をビニングしてスパイク数行列を作成
    for i, times in enumerate(spike_times_list):
        if len(times) > 0:
            counts, _ = np.histogram(times, bins=bins)
            spike_counts[i, :] = counts

    # 各ニューロンのスパイク数を平均0にセンタリングする場合
    if center:
        spike_counts = spike_counts - np.mean(spike_counts, axis=1, keepdims=True) # 各ニューロンについて時間方向の平均スパイク数を引く

    return effective_rank_from_matrix(spike_counts) # 作成したスパイク数行列からeffective rankを計算