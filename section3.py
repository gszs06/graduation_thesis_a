import os
from function import section3_function
from function import read_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
np.set_printoptions(precision=3,suppress=True)



data_band = np.load("F:/光谱数据/data_band_2.npy")
index = np.load("F:/光谱数据/index_1.npy")
###每运行一次取数据再运行下面的
index_0 = index[read_data.select_data_2(index,day='14',year='2016')]
data_band_0 = data_band[read_data.select_data_2(index,day='14',year='2016')]
_ = section3_function.pre_statis_test(data_band_0,index_0)
###
index_1 = index[read_data.select_data_2(index,day='21',year='2016')]
data_band_1 = data_band[read_data.select_data_2(index,day='21',year='2016')]
_ = section3_function.pre_statis_test(data_band_1,index_1)

index_2 = index[read_data.select_data_2(index,day='28',year='2016')]
data_band_2 = data_band[read_data.select_data_2(index,day='28',year='2016')]
_ = section3_function.pre_statis_test(data_band_2,index_2)

index_3 = index[read_data.select_data_2(index,day='14',year='2017')]
data_band_3 = data_band[read_data.select_data_2(index,day='14',year='2017')]
_ = section3_function.pre_statis_test(data_band_3,index_3)

index_4 = index[read_data.select_data_2(index,day='21',year='2017')]
data_band_4 = data_band[read_data.select_data_2(index,day='21',year='2017')]
_ = section3_function.pre_statis_test(data_band_4,index_4)

index_5 = index[read_data.select_data_2(index,day='28',year='2017')]
data_band_5 = data_band[read_data.select_data_2(index,day='28',year='2017')]
_ = section3_function.pre_statis_test(data_band_5,index_5)


mean_data_16 = np.zeros((3,3,1000),dtype=np.float)
mean_data_16[0,0,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['ck1','ck2','ck3'],day='14',year='2016')],axis=0)
mean_data_16[0,1,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['ck1','ck2','ck3'],day='21',year='2016')],axis=0)
mean_data_16[0,2,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['ck1','ck2','ck3'],day='28',year='2016')],axis=0)
mean_data_16[1,0,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['p1','p2','p3'],day='14',year='2016')],axis=0)
mean_data_16[1,1,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['p1','p2','p3'],day='21',year='2016')],axis=0)
mean_data_16[1,2,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['p1','p2','p3'],day='28',year='2016')],axis=0)
mean_data_16[2,0,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['m1','m2','m3'],day='14',year='2016')],axis=0)
mean_data_16[2,1,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['m1','m2','m3'],day='21',year='2016')],axis=0)
mean_data_16[2,2,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['m1','m2','m3'],day='28',year='2016')],axis=0)
mean_data_16_d = np.zeros((3,3,80),dtype=np.float)
for i in range(3):
    for j in range(3):
        mean_data_16_d[i,j,:] = section3_function.derivative(mean_data_16[i,j,:])

mean_data_17 = np.zeros((3,3,1000),dtype=np.float)
mean_data_17[0,0,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['ck1','ck2','ck3'],day='14',year='2017')],axis=0)
mean_data_17[0,1,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['ck1','ck2','ck3'],day='21',year='2017')],axis=0)
mean_data_17[0,2,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['ck1','ck2','ck3'],day='28',year='2017')],axis=0)
mean_data_17[1,0,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['p1','p2','p3'],day='14',year='2017')],axis=0)
mean_data_17[1,1,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['p1','p2','p3'],day='21',year='2017')],axis=0)
mean_data_17[1,2,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['p1','p2','p3'],day='28',year='2017')],axis=0)
mean_data_17[2,0,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['m1','m2','m3'],day='14',year='2017')],axis=0)
mean_data_17[2,1,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['m1','m2','m3'],day='21',year='2017')],axis=0)
mean_data_17[2,2,:] = np.mean(data_band[read_data.select_data_2(index,strlist=['m1','m2','m3'],day='28',year='2017')],axis=0)
mean_data_17_d = np.zeros((3,3,80),dtype=np.float)
for i in range(3):
    for j in range(3):
        mean_data_17_d[i,j,:] = section3_function.derivative(mean_data_17[i,j,:])





