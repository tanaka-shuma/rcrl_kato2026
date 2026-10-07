# Copyright (c) 2021-2023 Katori lab. All Rights Reserved.
"""
NARMAタスク用時系列データの生成
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_narma(NT, seed=None, order=10):
    """
    Generates a Nonlinear AutoRegressive Moving Average (NARMA) sequence.
    
    :param NT: Number of timesteps
    :param seed: Seed for random number generation
    :param order: Order of the NARMA model (equivalent to delay + 1)
    :return: Tuple containing input sequence u and output sequence d

    The NARMA task is a benchmark problem for time-series prediction.
    This function generates sequences for the NARMA task, following the
    given parameters.

    Reference:
    Atiya, A. F., & Parlos, A. G. (2000). New results on recurrent network
    training: unifying the algorithms and accelerating convergence.
    IEEE Transactions on Neural Networks, 11(3), 697-709.
    """
    if seed is not None:
        np.random.seed(seed)

    # Ensure order is within the valid range
    if order < 1 or order >= NT:
        print("Invalid order value")
        return None, None

    # Constants
    NTtrans = 200
    NT_total = NT + NTtrans
    u = np.random.uniform(0, 0.5, (NT_total))

    # Generate NARMA sequence
    d = 1.0 * np.zeros((NT_total))
    for i in range(order - 1, NT_total - 1):
        d[i + 1] = 0.3 * d[i] + 0.05 * d[i] * np.sum(d[i - (order - 1):i + 1]) + 1.5 * u[i - (order - 1)] * u[i] + 0.1

    d = d[NTtrans:]
    u = u[NTtrans:]
    NT = NT - NTtrans

    # Check if values have diverged and regenerate sequence if necessary
    if np.isfinite(d).all():
        u = u.reshape((-1, 1))
        d = d.reshape((-1, 1))
        return u, d
    else:
        print("Sequence diverged, regenerating...")
        new_seed = seed + 1 if seed is not None else None
        return generate_narma(NT=NT, seed=new_seed, order=order)

if __name__ == '__main__':
    u, d = generate_narma(1000)
    if u is not None and d is not None:
        plt.plot(u)
        plt.plot(d)
        plt.show()
        print(u.shape, d.shape)
    else:
        print("Error generating sequence.")
