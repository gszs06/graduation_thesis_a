from usefulfunction import function_adjust
from usefulfunction import function_plot
from usefulfunction import function_corr
from usefulfunction import read_data

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import PyEMD
from scipy import fftpack
from scipy.signal import find_peaks

data_band = np.load("F:/光谱数据/data_band_2.npy")
index = np.load("F:/光谱数据/index_1.npy")
fn_lai = "F:/光谱数据/LAI.txt"
lai = read_data.read_physiological_index(index,fn=fn_lai)
lai = np.array(lai,dtype=np.float32)
Wave = np.arange(350,1350)

def emd_plot(imfs):
    numb = len(imfs)
    fig,axes = plt.subplots(nrows=numb,ncols=2,figsize=(7,10))
    for i in range(numb):
        axes[i,0].plot(Wave,imfs[i])
        X_fft = fftpack.fft(imfs[i])
        axes[i,1].plot(np.abs(X_fft))
        print('第{:d}个的最大频率出现在{:d}，幅度为{:.2f}'.format(i,np.argmax(np.abs(X_fft)),np.max(np.abs(X_fft))))
    plt.show()

if __name__ == '__main__':

    n = 197
    print('选择了第{:d}个样本'.format(n))
    E_IMFs = PyEMD.EEMD(trials=50)
    E_IMFs.noise_seed(0)
    imfs = E_IMFs.eemd(data_band[n,:],Wave,max_imf=8)
    a = imfs[4] + imfs[5]
    b = imfs[6] + imfs[7]
    emd_plot(imfs)
