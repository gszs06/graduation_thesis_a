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


def emd_plot(imfs):
    numb = len(imfs)
    fig,axes = plt.subplots(nrows=numb,ncols=2,figsize=(7,10))
    for i in range(numb):
        axes[i,0].plot(Wave,imfs[i])
        X_fft = fftpack.fft(imfs[i])
        axes[i,1].plot(np.abs(X_fft))
        print('第{:d}个的最大频率出现在{:d}，幅度为{:.2f}'.format(i,np.argmax(np.abs(X_fft)),np.max(np.abs(X_fft))))
    plt.show()


data_band = np.load("F:/光谱数据/data_band_2.npy")
index = np.load("F:/光谱数据/index_1.npy")
fn_lai = "F:/光谱数据/LAI.txt"
lai = read_data.read_physiological_index(index,fn=fn_lai)
lai = np.array(lai,dtype=np.float32)
Wave = np.arange(350,1350)
if __name__ == '__main__':

    n = np.random.randint(0,327)
    print('选择了第{:d}个样本'.format(n))
    E_IMFs = PyEMD.EEMD(trials=50)
    E_IMFs.noise_seed(0)
    imfs = E_IMFs.eemd(data_band[n,:],Wave,max_imf=8)
    print(len(imfs))
    emd_plot(imfs)






####尝试

#if __name__ == '__main__':
#
#    print('随机选择了20个样本测试极大值点个数')
#    for i in range(20):
#
#        n = np.random.randint(0,327)
#        print('选择了第{:d}个样本'.format(n))
#        E_IMFs = PyEMD.EEMD(trials=50)
#        E_IMFs.noise_seed(0)
#        imfs = E_IMFs.eemd(data_band[n,:],Wave,max_imf=8)
#        a = imfs[4] + imfs[5]
#        b = imfs[6] + imfs[7]
#        peak_points = find_peaks(a,width=50)[0]
#        print('余波5和6的极大值点个数为{:d}'.format(len(peak_points)))
#
#
#
#
#
#a = imfs[4] + imfs[5]
#b = imfs[6] + imfs[7]
#plt.plot(Wave,a)
#from scipy.signal import find_peaks
#peak_points = find_peaks(a,width=50)[0]
#
#valley_points = []
#for i in range(len(peak_points)-1):
#    valley_points.append(find_peaks(-a[peak_points[i]:peak_points[i+1]])[0][0] + peak_points[i])
#
#
#normalize = np.zeros((1000),dtype=np.float32)
#normalize[0:peak_points[0]] = a[0:peak_points[0]]
#normalize[peak_points[0]:peak_points[1]] = line(a,peak_points[0],peak_points[1])
#normalize[peak_points[1]:peak_points[2]] = line(a,peak_points[1],peak_points[2])
#normalize[peak_points[2]:peak_points[3]] = line(a,peak_points[2],peak_points[3])
#normalize[peak_points[3]:] = a[peak_points[3]:]
#
#
#fig = plt.figure(figsize=(5,4))
#ax0 = fig.add_subplot(111)
#ax0.plot(a)
#ax0.plot(normalize)
#plt.show()
#
#
#
#def line(relf,a,b):
#    x = np.arange(a,b)
#    x1 = a
#    y1 = relf[int(a)]
#    x2 = b
#    y2 = relf[int(b)]
#
#    y = (x-x2)*(y1-y2)/(x1-x2) + y2
#    return y
#
####进度条尝试
#import time
#scale = 50
#start = time.time()
#for i in range(scale+1):
#    a = "*" * i
#    b = "." * (scale-i)
#    c = (i / scale) * 100
#    dur = time.time() - start
#    print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c,a,b,dur),end = "")
#    time.sleep(0.1)


#######
data_band = np.load("F:/光谱数据/data_band_2.npy")
index = np.load("F:/光谱数据/index_1.npy")
fn_lai = "F:/光谱数据/LAI.txt"
lai = read_data.read_physiological_index(index,fn=fn_lai)
lai = np.array(lai,dtype=np.float32)
Wave = np.arange(350,1350)

mask = [0]*len(data_band)
mask[5] = 1
mask[40] = 1
mask[315] = 1
lai  = np.ma.array(lai,mask=mask)
lai.compressed()

data_sum = np.load("F:/光谱数据/data_sum.npy")

a = data_sum[:,0] + data_sum[:,1]
b = data_sum[:,2] + data_sum[:,3]
c = data_sum[:,4] + data_sum[:,5]


####第二次使用EEMD方法尝试
data_sum_Enorm = np.load("F:/光谱数据/data_sum_Enorm.npy")
for i in range(9):
    print(pearsonr(data_sum_Enorm[:,i],lai.compressed()))

#分别选择了第一个左右峰面积，第二三个深度为参数建模

third_data_area1_left = data_sum_Enorm[:,0]
third_data_area1_left = third_data_area1_left.reshape(-1,1)
from sklearn.linear_model import LinearRegression
model_line = LinearRegression()
random_index = [i for i in range(len(data_sum_Enorm))]
np.random.shuffle(random_index)
train_data_second = third_data_area1_left[random_index[0:262]]
train_label_second = lai[random_index[0:262]]
test_data_second = third_data_area1_left[random_index[262:]]
test_label_second = lai.compressed()[random_index[262:]]
model_line.fit(train_data_second,train_label_second)

print('建立的方程为:y = {:.2f}x+{:.2f}'.format(model_line.coef_[0],model_line.intercept_))
print('使用检验数据得到的R方为:{:.2f}'.format(model_line.score(test_data_second,test_label_second)))

##
third_data_area1_right = data_sum_Enorm[:,1]
third_data_area1_right = third_data_area1_right.reshape(-1,1)
from sklearn.linear_model import LinearRegression
model_line = LinearRegression()
random_index = [i for i in range(len(data_sum_Enorm))]
np.random.shuffle(random_index)
train_data_second = third_data_area1_right[random_index[0:262]]
train_label_second = lai.compressed()[random_index[0:262]]
test_data_second = third_data_area1_right[random_index[262:]]
test_label_second = lai.compressed()[random_index[262:]]
model_line.fit(train_data_second,train_label_second)

print('建立的方程为:y = {:.2f}x+{:.2f}'.format(model_line.coef_[0],model_line.intercept_))
print('使用检验数据得到的R方为:{:.2f}'.format(model_line.score(test_data_second,test_label_second)))

###
third_data_deep2 = data_sum_Enorm[:,-2]
third_data_deep2 = third_data_deep2.reshape(-1,1)
from sklearn.linear_model import LinearRegression
model_line = LinearRegression()
random_index = [i for i in range(len(data_sum_Enorm))]
np.random.shuffle(random_index)
train_data_second = third_data_deep2[random_index[0:262]]
train_label_second = lai.compressed()[random_index[0:262]]
test_data_second = third_data_deep2[random_index[262:]]
test_label_second = lai.compressed()[random_index[262:]]
model_line.fit(train_data_second,train_label_second)

print('建立的方程为:y = {:.2f}x+{:.2f}'.format(model_line.coef_[0],model_line.intercept_))
print('使用检验数据得到的R方为:{:.2f}'.format(model_line.score(test_data_second,test_label_second)))


###

data_sum_Enorm = np.load("F:/光谱数据/data_sum_orignorm.npy")
for i in range(9):
    print(pearsonr(data_sum_Enorm[:,i],lai.data))



