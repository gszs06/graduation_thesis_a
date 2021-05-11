from function import section4_function
from function import read_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import subprocess
import os
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
np.set_printoptions(precision=3,suppress=True)


data_band = np.load("F:/光谱数据/data_band_2.npy")
index = np.load("F:/光谱数据/index_1.npy")
fn_lai = "F:/光谱数据/LAI.txt"
lai = read_data.read_physiological_index(index,fn=fn_lai)
lai = np.array(lai,dtype=np.float32)

###第一类
R_square,P = section4_function.corr_singlewave(data_band,lai)
high_index,high_singlewave_interact = section4_function.high_singlewave_10per(data_band,R_square)
start_end = section4_function.high_singlewave_extent(high_index)
####绘图（相关性及相互相关性图）
section4_function.corr_singlewave_plot(R_square,P,start_end,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/1.png')
section4_function.high_interact_plot(high_index,high_singlewave_interact,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/2.png')
####使用PCA进行分析得出最佳维度，绘图佐证

high_data = data_band[:,high_index]
pca = PCA()
pca.fit(high_data)
cumsum = np.cumsum(pca.explained_variance_ratio_)
section4_function.PCA_var_plot(cumsum,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/3.png')
####由上述分析得到最佳的降维数，下面进行降维并建立简单的线性模型
pca = PCA(n_components=1)
first_data_PCA_1 = pca.fit_transform(high_data)
model_lines,evaluation = section4_function.evaluation_system(first_data_PCA_1,lai)
model_best = model_lines[np.argmax(evaluation[:,0])]
evaluation_best = evaluation[np.argmax(evaluation[:,0])]
section4_function.line_scatter_plot(model_lines,evaluation,first_data_PCA_1,lai,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/4.png')

###第二类
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
customer_names = ['EVI**','SIPI','VARI710','TCARI','SAVI**','MSAVI**','OSAVI**','VARIgreen**','mSR705','mND705',
        'CUR','redeage','TVI-B-L','VARI700','TVI-3**','MSR*','PSRI','MCARI','MTCI']
customer_names = np.array(customer_names)
customer_data,customer_R_square = section4_function.corr_customer(customer_functions,data_band,lai)
customer_interact = section4_function.all_customer_interact(customer_data,customer_R_square)
####绘图
section4_function.customer_interact_plot(customer_interact,customer_R_square,customer_names,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/5.png')

high_customer_data = customer_data[:,np.argsort(customer_R_square)[-6:][::-1]]
high_customer_names = customer_names[np.argsort(customer_R_square)[-6:][::-1]]
pca = PCA(n_components=1)
second_data_PCA_1 = pca.fit_transform(high_customer_data)
second_data_1 = np.hstack((high_customer_data,second_data_PCA_1))
second_data_1_name = np.hstack((high_customer_names,['index']))

result = []
model_best = []
evaluation_best = [] 
for i in range(second_data_1.shape[1]):
    result.append(section4_function.evaluation_system(second_data_1[:,i].reshape(-1,1),lai))
for model_lines,evaluation in result:
    model_best.append(model_lines[np.argmax(evaluation[:,0])])
    evaluation_best.append(evaluation[np.argmax(evaluation[:,0])])

section4_function.line_scatter_plot_2(model_best,evaluation_best,second_data_1,lai,second_data_1_name,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/6.png')

###全波段差值、归一化植被指数
runing = 'python {}'.format(os.path.abspath("script/double_band_index.py"))
data_path = "{}\n{}\n".format(os.path.abspath("data/data_band_2.npy"),os.path.abspath("data/lai.npy"))
data_path = bytes(data_path,encoding="utf8")
a = subprocess.run(args=runing,input=data_path,capture_output=True)

result_lai = np.load(os.path.abspath("data/result_lai.npy"))
section4_function.plot_doubelband(result_lai,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/7.png')
DVI = section4_function.doubelband_10(result_lai[0],data_band,index='DVI')
NDVI = section4_function.doubelband_10(result_lai[2],data_band,index='NDVI')

pca_DVI = PCA(n_components=1)
pca_NDVI = PCA(n_components=1)
DVI_pca = pca_DVI.fit_transform(DVI)
NDVI_pca = pca_NDVI.fit_transform(NDVI)
model_lines_DVI,evaluation_DVI = section4_function.evaluation_system(DVI_pca,lai)
model_best_DVI = model_lines_DVI[np.argmax(evaluation_DVI[:,0])]
evaluation_best_DVI = evaluation_DVI[np.argmax(evaluation_DVI[:,0])]

model_lines_NDVI,evaluation_NDVI = section4_function.evaluation_system(NDVI_pca,lai)
model_best_NDVI = model_lines_NDVI[np.argmax(evaluation_NDVI[:,0])]
evaluation_best_NDVI = evaluation_NDVI[np.argmax(evaluation_NDVI[:,0])]

section4_function.line_scatter_plot_3(model_best_DVI,evaluation_best_DVI,DVI_pca,model_best_NDVI,evaluation_best_NDVI,NDVI_pca,lai,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/8.png')


######第三类
runing_2 = 'python {}'.format(os.path.abspath("script/eemd_envelope.py"))
a = subprocess.run(args=runing_2,capture_output=True)

data_sum_Enorm = np.load("F:/光谱数据/data_sum_Enorm.npy")
corr_area = []

mask = [0]*len(data_band)
mask[5] = 1
mask[40] = 1
mask[315] = 1
lai_mask  = np.ma.array(lai,mask=mask)

for i in range(9):
    corr_area.append(pearsonr(data_sum_Enorm[:,i],lai_mask.compressed()))
model_lines_1,evaluation_1 = section4_function.evaluation_system(data_sum_Enorm[:,0].reshape(-1,1),lai_mask.compressed())
model_best_1 = model_lines_1[np.argmax(evaluation_1[:,0])]
evaluation_best_1 = evaluation_1[np.argmax(evaluation_1[:,0])]

model_lines_2,evaluation_2 = section4_function.evaluation_system(data_sum_Enorm[:,1].reshape(-1,1),lai_mask.compressed())
model_best_2 = model_lines_2[np.argmax(evaluation_2[:,0])]
evaluation_best_2 = evaluation_2[np.argmax(evaluation_2[:,0])]

model_lines_3,evaluation_3 = section4_function.evaluation_system(1-data_sum_Enorm[:,7].reshape(-1,1),lai_mask.compressed())
model_best_3 = model_lines_3[np.argmax(evaluation_3[:,0])]
evaluation_best_3 = evaluation_3[np.argmax(evaluation_3[:,0])]

model_lines_4,evaluation_4 = section4_function.evaluation_system(1-data_sum_Enorm[:,8].reshape(-1,1),lai_mask.compressed())
model_best_4 = model_lines_4[np.argmax(evaluation_4[:,0])]
evaluation_best_4 = evaluation_4[np.argmax(evaluation_4[:,0])]

###建立预测模型
##数据准备
X_1 = np.delete(first_data_PCA_1,[5,40,315],axis=0)
X_2 = np.delete(second_data_PCA_1,[5,40,315],axis=0)
X_3 = np.delete(DVI_pca,[5,40,315],axis=0)
X_4 = data_sum_Enorm[:,-2].reshape(-1,1)
X = np.concatenate((X_1,X_2,X_3,X_4),axis=1)
Y = np.delete(lai,[5,40,315]).reshape(-1,1)
##保存数据
np.save(os.path.abspath("data/X.npy"),X)
np.save(os.path.abspath("data/Y.npy"),Y)
##读取数据
#X = np.load(os.path.abspath("data/X.npy"))
#Y = np.load(os.path.abspath("data/Y.npy"))
##岭回归
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from joblib import dump,load

alphas = [0.1,0.5,1,10,20,100]
models_Ridge = [Ridge(alpha=i) for i in alphas]
models_Ridge = section4_function.plot_Ridge_learning_curves(models_Ridge,alphas,X,Y,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/9.png')
section4_function.save_models(models_Ridge,name='Ridge')
##决策树回归
models_Tree = [DecisionTreeRegressor(max_depth=2),DecisionTreeRegressor(max_depth=3),DecisionTreeRegressor(max_depth=4)]
names = [r'$R^1_{pca(771-872)}$',r'inde$x^1_{pca}$',r'DV$I^1_{pca}$',r'$D_{969}$']
models_Tree = section4_function.plot_Tree_learning_curves(models_Tree,X,Y,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/10.png')
section4_function.plot_Tree(models_Tree[-2],names,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/11.png')
section4_function.save_models(models_Tree,name='Tree')
##支持向量机回归
models_SVR = [  SVR(kernel='linear',epsilon=0.1,C=1),
                SVR(kernel='rbf',epsilon=0.1,C=50,gamma=0.005),
                SVR(kernel='poly',epsilon=0.01,C=50,degree=2,gamma=0.005),
             ]
models_SVR = section4_function.plot_SVR_learning_curves(models_SVR,X,Y,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/12.png')
section4_function.save_models(models_SVR,name='SVR')

######讨论部分











##数据准备
index = np.delete(index,[5,40,315],axis=0)

index_0 = index[read_data.select_data_2(index,day='14',year='2016')]
X_0 = X[read_data.select_data_2(index,day='14',year='2016')]
_ = read_data.pre_statis_test(X_0,index_0)

index_1 = index[read_data.select_data_2(index,day='21',year='2016')]
X_1 = X[read_data.select_data_2(index,day='21',year='2016')]
_ = read_data.pre_statis_test(X_1,index_1)

index_2 = index[read_data.select_data_2(index,day='28',year='2016')]
X_2 = X[read_data.select_data_2(index,day='28',year='2016')]
_ = read_data.pre_statis_test(X_2,index_2)

index_3 = index[read_data.select_data_2(index,day='14',year='2017')]
X_3 = X[read_data.select_data_2(index,day='14',year='2017')]
_ = read_data.pre_statis_test(X_3,index_3)

index_4 = index[read_data.select_data_2(index,day='21',year='2017')]
X_4 = X[read_data.select_data_2(index,day='21',year='2017')]
_ = read_data.pre_statis_test(X_4,index_4)

index_5 = index[read_data.select_data_2(index,day='28',year='2017')]
X_5 = X[read_data.select_data_2(index,day='28',year='2017')]
_ = read_data.pre_statis_test(X_5,index_5)



T1_14 = np.array([[-0.37,1.47,-8.36,0.27]])
T2_14 = np.array([[0.21,-1.69,6.9,0.34]])
T3_14 = np.array([[0.74,-3.66,18.37,0.39]])



#######

lai_par_ck16 = np.mean(X[read_data.select_data_2(index,['ck1','ck2','ck3'],year='2016')],axis=0)
lai_par_p16 = np.mean(X[read_data.select_data_2(index,['p1','p2','p3'],year='2016')],axis=0)
lai_par_m16 = np.mean(X[read_data.select_data_2(index,['m1','m2','m3'],year='2016')],axis=0)
lai_par_ck17 = np.mean(X[read_data.select_data_2(index,['ck1','ck2','ck3'],year='2017')],axis=0)
lai_par_p17 = np.mean(X[read_data.select_data_2(index,['p1','p2','p3'],year='2017')],axis=0)
lai_par_m17 = np.mean(X[read_data.select_data_2(index,['m1','m2','m3'],year='2017')],axis=0)
pre_T1 = [-0.4,2.69,-13.81,0.29]
pre_T2 = [-0.21,1.13,-6.11,0.31]
pre_T3 = [-0.2,1.24,-6.55,0.31]
##模型读取
models = [load(os.path.abspath('models/lai_SVR_0.joblib')),
            load(os.path.abspath('models/lai_Tree_1.joblib')),
            load(os.path.abspath('models/lai_SVR_1.joblib'))]
##绘图
section4_function.plot_predict_10_60(pre_T1,pre_T2,pre_T3,models,path='F:/吟咏之间，吐纳珠玉之声/论文改2/4章/图片/13.png')

models[0].ppre

