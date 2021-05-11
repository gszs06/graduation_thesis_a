###计算准备工作
##导包
from function import section5_function
from function import read_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import subprocess
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
##设置显示数据保存三位数
np.set_printoptions(precision=3,suppress=True)
##读取基本数据
data_band = np.load("F:/光谱数据/data_band_2.npy")
index = np.load("F:/光谱数据/index_1.npy")
fn_biomass = "F:/光谱数据/above_ground_biomass.txt"
biomass = read_data.read_physiological_index(index,fn=fn_biomass)
biomass = np.array(biomass,dtype=np.float32)
index_21 = index[read_data.select_data_2(index,day='21')]
data_band = data_band[index_21]
biomass = biomass[index_21]

###计算光谱参数
##第一类
#计算
R_square,P = section5_function.corr_singlewave(data_band,biomass)
high_index,high_singlewave_interact = section5_function.high_singlewave_10per(data_band,R_square)
start_end = section5_function.high_singlewave_extent(high_index)

#绘图
section5_function.corr_singlewave_plot(R_square,P,start_end)
section5_function.high_interact_plot(high_index,high_singlewave_interact)
#建立线性模型
high_data = data_band[:,high_index]
pca = PCA(n_components=1)
first_data_PCA_1 = pca.fit_transform(high_data)

first_data_PCA_1 = data_band[:,250].reshape((-1,1))

model_lines,evaluation = section5_function.evaluation_system(first_data_PCA_1,biomass)
model_best = model_lines[np.argmax(evaluation[:,0])]
evaluation_best = evaluation[np.argmax(evaluation[:,0])]
