# Copyright (c) 2018-2023 Katori lab. All Rights Reserved
# plotに関わる関数

import numpy as np
import matplotlib.pyplot as plt 
from cycler import cycler

def generate_color_cycle(cmap_name, length):
    """
    Given a colormap name and a length, generate a color cycle using matplotlib.

    Parameters:
    cmap_name (str): The name of the colormap to use.
    length (int): The number of colors to generate in the cycle.

    Returns:
    cycler.Cycler: A color cycler object for use with matplotlib.
    """
    cmap = plt.get_cmap(cmap_name)
    colors = []
    for i in range(length):
        colors.append(cmap(float(i) / length))
    cycle = cycler(color=colors)
    return cycle

def plot_rc(U,X,Y,D):
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(16, 10))
    
    (nrows,ncols) = (3,1)

    plt.subplot(nrows, ncols, 1)
    plt.plot(U)
    #plt.xlabel("time steps")
    plt.ylabel("Input (U)")

    plt.subplot(nrows, ncols, 2)
    plt.plot(X)
    #plt.xlabel("time steps")
    plt.ylabel("Reservoir (X)")

    plt.subplot(nrows, ncols, 3)
    plt.plot(Y)
    plt.plot(D)
    plt.xlabel("time steps")
    plt.ylabel("Output (Y,D)")
    
    #plt.savefig("./{}/{}".format(dir_name,fig_name))
    plt.show()
