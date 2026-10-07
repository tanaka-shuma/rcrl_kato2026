"""
https://physionet.org/content/santa-fe/1.0.0/

    The data are presented in text form and have been split into two sequential parts,
    b1.txt and b2.txt.Each line contains simultaneous samples of three parameters;
    the interval between samples in successive lines is 0.5 seconds. 
    The first column is the heart rate, the second is the chest volume (respiration force),
    and the third is the blood oxygen concentration (measured by ear oximetry). 
    The sampling frequency for each measurement is 2 Hz 
    (i.e., the time interval between measurements in successive rows is 0.5 seconds).

"""
import matplotlib.pyplot as plt 
import numpy as np

def generate_santafe():
    filename = "./santafeA.txt"
    filename = "./santafeA2.txt"
    with open(filename, 'r', encoding='UTF-8') as f:
        data = np.array(list(f)).astype(int)

    return data

if __name__ == '__main__':
    u = generate_santafe()
    plt.plot(u)
    #plt.plot(d)
    plt.show()
    
