from function import read_data
import numpy as np
import matplotlib.pyplot as plt
import os

##设置显示数据保存三位数
np.set_printoptions(precision=3,suppress=True)
##读取基本数据
data_band = np.load("F:/光谱数据/data_band_2.npy")
index = np.load("F:/光谱数据/index_1.npy")


a = data_band[277,:]
b = np.load(os.path.abspath('data/277_eemd.npy'))
b = np.sum(b[4:],axis=0)
c = (a - b)**2
d = 10*np.log10((b**2)/c)

plt.plot(d)
plt.bar(x=list(range(350,1350)),height=d)





