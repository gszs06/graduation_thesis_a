from usefulfunction import function_adjust
from usefulfunction import function_plot
from usefulfunction import function_corr
from usefulfunction import read_data

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

data_band = np.load("F:/光谱数据/data_band_2.npy")
index = np.load("F:/光谱数据/index_1.npy")
fn_lai = "F:/光谱数据/LAI.txt"
lai = read_data.read_physiological_index(index,fn=fn_lai)
lai = np.array(lai,dtype=np.float32)



######第一类参数
###计算（相关性）：
R_square,P = function_corr.corr_singlewave(data_band,lai)
high_index,high_singlewave_interact = function_corr.high_singlewave_10per(data_band,R_square)
start_end = function_corr.high_singlewave_extent(high_index)

####绘图（相关性及相互相关性图）
function_plot.corr_singlewave_plot(R_square,P,start_end,path=None)
function_plot.high_interact_plot(high_index,high_singlewave_interact,path=None)

####使用PCA进行分析得出最佳维度，绘图佐证
from sklearn.decomposition import PCA
high_data = data_band[:,high_index]
pca = PCA()
pca.fit(high_data)
cumsum = np.cumsum(pca.explained_variance_ratio_)
function_plot.PCA_var_plot(cumsum)

####由上述分析得到最佳的降维数，下面进行降维并建立简单的线性模型
pca = PCA(n_components=1)
first_data_PCA_1 = pca.fit_transform(high_data)

from sklearn.linear_model import LinearRegression
model_line = LinearRegression()
random_index = [i for i in range(len(data_band))]
np.random.shuffle(random_index)
train_data_first = first_data_PCA_1[random_index[0:262]]
train_label_first = lai[random_index[0:262]]
test_data_first = first_data_PCA_1[random_index[262:]]
test_label_first = lai[random_index[262:]]
model_line.fit(train_data_first,train_label_first)

print('建立的方程为:y = {:.2f}x+{:.2f}'.format(model_line.coef_[0],model_line.intercept_))
print('使用检验数据得到的R方为:{:.2f}'.format(model_line.score(test_data_first,test_label_first)))


####第二类参数
customer_functions = ['2.5*(Rnir-R680)/(1+Rnir+6*R680-7.5*R460)',
                '(R810-R460)/(R810-R680)',
                '(R710-1.7*R680+0.7*R460)/(R710+2.3*R680-1.3*R460)',
                '3*((R710-R680)-0.2*(R700-R560)*(R710/R68))',
                '1.5*(R870-R680)/(R870+R680+0.5)',
                '0.5*(2*Rnir+1-((2*Rnir+1)**2 - 8*(Rnir-Rred))**0.5)',
                '(1+0.16)*(R810-R680+0.16)',
                '(Rgreen-Rred)/(Rgreen+Rred-Rblue)',
                '(R750-R445)/(R705-R445)',
                '(R750-R705)/(R750+R705-2*R445)',
                '(R675*R690)/(R683*R683)',
                '710+50*((1/2*(R810+R660)-R710)/(R760-R710))',
                '0.5*(120*(R750-R550)-200*(R670-R550))',
                '(R700-1.7*Rred+0.7*Rblue)/(R700+2.3*Rred-1.3*Rblue)',
                '60*(Rnir-Rgreen)-100*(Rred-Rgreen)',
                '(Rnir/Rred-1)/((Rnir/Rred+1)**0.5)',
                '(R680-R500)/(R750)',
                '(R700-R670)-0.2*(R700-R550)*(R700/R670)',
                '(R750-R710)/(R710-R680)']
customer_names = ['EVI','SIPI','VARI710','TCARI','SAVI','MSAVI','OSAVI','VARIgreen','mSR705','mND705',
        'CUR','redeage','TVI-B-L','VARI700','TVI-3','MSR','PSRI','MCARI','MTCI']
customer_names = np.array(customer_names)
customer_data,customer_R_square = function_corr.corr_customer(customer_functions,data_band,lai)
customer_interact = function_corr.all_customer_interact(customer_data,customer_R_square)

####绘图
function_plot.customer_interact_plot(customer_interact,customer_R_square,customer_names,path=None)

####使用PCA进行分析得出最佳维度，绘图佐证
from sklearn.decomposition import PCA
high_customer_data = customer_data[:,np.argsort(customer_R_square)[::-1][0:6]].copy()
pca = PCA()
pca.fit(high_customer_data)
cumsum = np.cumsum(pca.explained_variance_ratio_)

####由上述分析得到最佳的降维数，下面进行降维并建立简单的线性模型
pca = PCA(n_components=1)
second_data_PCA_1 = pca.fit_transform(high_customer_data)

from sklearn.linear_model import LinearRegression
model_line = LinearRegression()
random_index = [i for i in range(len(data_band))]
np.random.shuffle(random_index)
train_data_second = second_data_PCA_1[random_index[0:262]]
train_label_second = lai[random_index[0:262]]
test_data_second = second_data_PCA_1[random_index[262:]]
test_label_second = lai[random_index[262:]]
model_line.fit(train_data_second,train_label_second)

print('建立的方程为:y = {:.2f}x+{:.2f}'.format(model_line.coef_[0],model_line.intercept_))
print('使用检验数据得到的R方为:{:.2f}'.format(model_line.score(test_data_second,test_label_second)))






customer_number = customer_data.shape[1]
customer_interact = np.zeros(customer_number**2,dtype=np.float32)
customer_sort_index = np.argsort(customer_R_square)[::-1]
customer_data_sort = customer_data[:,customer_sort_index]


k = 0
for i in range(customer_number):
        for j in range(customer_number):
                customer_interact[k] = (pearsonr(customer_data_sort[:,i],customer_data_sort[:,j])[0])**2
                k = k + 1
customer_interact = customer_interact.reshape(customer_number,customer_number)

fig = plt.figure(figsize=(5,4),dpi=200)
ax0 = fig.add_subplot(111)
im = ax0.imshow(customer_interact,origin='lower',cmap=plt.cm.jet,interpolation='nearest')
fig.colorbar(im,ax=ax0,shrink=1)
plt.show()

####https://matplotlib.org/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(6.5,4),dpi=200,constrained_layout=True)
spc = fig.add_gridspec(ncols=5,nrows=1)
ax1 = fig.add_subplot(spc[:,0:1])
colorlist = ['r','r','r','r','r','r','b','b','b','b','b','b','b','b','b','b','b','b','b']
ax1.barh(range(19),np.sort(customer_R_square)[::-1],color=colorlist)
ax1.set_yticks(range(19))
ax1.set_yticklabels(customer_names[np.argsort(customer_R_square)[::-1]],fontsize=10,color='k')
plt.ylim(-0.5,18.5)

ax1.set_ylabel('已有的植被指数类型',fontsize=10,color='k')
ax1.set_xlabel('决定系数$R^2$',fontsize=10,color='k')

ax2 = fig.add_subplot(spc[:,1:])
im = ax2.imshow(customer_interact,origin='lower',cmap=plt.cm.jet,interpolation='nearest')
fig.colorbar(im,ax=ax2,shrink=1)
ax2.set_yticks([])
ax2.set_xticks([0,4,9,14,18])
ax2.set_xticklabels([customer_names[np.argsort(customer_R_square)[::-1]][0],
                customer_names[np.argsort(customer_R_square)[::-1]][4],
                customer_names[np.argsort(customer_R_square)[::-1]][9],
                customer_names[np.argsort(customer_R_square)[::-1]][14],
                customer_names[np.argsort(customer_R_square)[::-1]][18]],
                fontsize=10,color='k')
ax2.set_xlabel('已有的植被指数类型',fontsize=10,color='k')
plt.show()

