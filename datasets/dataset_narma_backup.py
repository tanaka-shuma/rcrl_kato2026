# Copyright (c) 2021-2023 Katori lab. All Rights Reserved.
"""
NARMAタスク用時系列データの生成
"""
import numpy as np
import matplotlib.pyplot as plt

def generate_narma(N,seed=None,delay=9):

    if seed != None : 
        np.random.seed(seed)
    
    """Generate NARMA sequence."""
    N1 = 200
    N = N + N1
    u = np.random.uniform(0,0.5,(N))

    # Generate NARMA sequence
    d = 1.0*np.zeros((N))
    
    for i in range(delay,N-1):
        d[i+1] = 0.3*d[i] + 0.05*d[i] * np.sum(d[i-delay:i+1]) + 1.5*u[i-delay]*u[i] + 0.1

    d = d[N1:]
    u = u[N1:]
    N = N-N1
    
    # NOTE: 乱数の実現値よっては値が発散する．その場合はseedを変更して再度時系列を生成する．
    if np.isfinite(d).all():
        u = u.reshape((N,1))
        d = d.reshape((N,1))
        return u,d
    else:
        print("again")
        return generate_narma(N=N,seed=seed+1)

if __name__ == '__main__':
    u,d = generate_narma(1000)
    plt.plot(u)
    plt.plot(d)
    plt.show()
    print(u.shape,d.shape)