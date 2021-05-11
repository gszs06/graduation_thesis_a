from function import section2_function
from function import read_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import subprocess
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

from scipy.signal import find_peaks
np.set_printoptions(precision=3,suppress=True)


data_band = np.load("F:/光谱数据/data_band_2.npy")
index = np.load("F:/光谱数据/index_1.npy")

running = 'python {}'.format(os.path.abspath("script/eemd.py"))
data_input = "277"
data_input = bytes(data_input,encoding='utf8')
a = subprocess.run(args=running,input=data_input,capture_output=True)
eemd_data = np.load(os.path.abspath("data/{:s}_eemd.npy".format(data_input.decode())))
fft_data = section2_function.fft(eemd_data)
fftmax_index,fftmax_data = section2_function.find_max(fft_data)
section2_function.plot_eemd_fft(eemd_data,fft_data,fftmax_index,fftmax_data,path="F:/吟咏之间，吐纳珠玉之声/论文改2/2章/图片/1.png")

original_data = data_band[277,:]
nonoise_data = np.sum(eemd_data[4:],axis=0)
section2_function.plot_noise(original_data,nonoise_data,path="F:/吟咏之间，吐纳珠玉之声/论文改2/2章/图片/2.png")

peak_points,valley_points = section2_function.find_point(nonoise_data)
line_data = section2_function.make_line_data(nonoise_data,peak_points)
norm_data = nonoise_data/line_data
norm_data = np.where(norm_data>1,1,norm_data)

points = np.hstack((peak_points,valley_points))
points = np.sort(points)
section2_function.plot_normway(line_data,nonoise_data,norm_data,points,path="F:/吟咏之间，吐纳珠玉之声/论文改2/2章/图片/3.png")


