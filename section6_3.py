from function import section6_function
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
index = np.delete(index,[5,40,315],axis=0)
fn_output = "F:/光谱数据/yield.txt"
output = read_data.read_physiological_index(index,fn=fn_output)
output = np.array(output,dtype=np.float32)

index_28 = index[read_data.select_data_2(index,day='28')]
data_band_28 = data_band[read_data.select_data_2(index,day='28')]
output_28 = output[read_data.select_data_2(index,day='28')]

###计算光谱参数
##第一类
#计算
R_square,P = section6_function.corr_singlewave(data_band_28,output_28)
high_index,high_singlewave_interact = section6_function.high_singlewave_10per(data_band_28,R_square)
start_end = section6_function.high_singlewave_extent(high_index[4:])
start_end = [(521,614)]
#绘图
section6_function.corr_singlewave_plot(R_square,P,start_end,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/1.png")
#section6_function.high_interact_plot(high_index,high_singlewave_interact,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/2.png")
#建立线性模型
high_data = data_band_28[:,high_index]
pca = PCA(n_components=1)
first_data_PCA_1 = pca.fit_transform(high_data)
model_lines,evaluation = section6_function.evaluation_system(first_data_PCA_1,output_28)
model_best = model_lines[np.argmax(evaluation[:,0])]
evaluation_best = evaluation[np.argmax(evaluation[:,0])]
section6_function.line_scatter_plot(model_lines,evaluation,first_data_PCA_1,output_28,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/2.png")
##第二类
#计算
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
customer_names = ['EVI','SIPI','VARI710','TCARI','SAVI','MSAVI','OSAVI','VARIgreen*','mSR705*','mND705*',
        'CUR*','redeage*','TVI-B-L','VARI700','TVI-3','MSR','PSRI','MCARI**','MTCI*']
customer_names = np.array(customer_names)
customer_data,customer_R_square = section6_function.corr_customer(customer_functions,data_band_28,output_28)
customer_interact = section6_function.all_customer_interact(customer_data,customer_R_square)
#绘图
section6_function.customer_interact_plot(customer_interact,customer_R_square,customer_names,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/3.png")
#建立线性模型
high_customer_data = customer_data[:,np.argsort(customer_R_square)[-1:][::-1]]
high_customer_names = customer_names[np.argsort(customer_R_square)[-1:][::-1]]
second_data_1 = high_customer_data
second_data_1_name = high_customer_names
result = []
model_best = []
evaluation_best = [] 
for i in range(second_data_1.shape[1]):
    result.append(section6_function.evaluation_system(second_data_1[:,i].reshape(-1,1),output_28))
for model_lines,evaluation in result:
    model_best.append(model_lines[np.argmax(evaluation[:,0])])
    evaluation_best.append(evaluation[np.argmax(evaluation[:,0])])
section6_function.line_scatter_plot_2(model_best,evaluation_best,second_data_1,output_28,second_data_1_name,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/4.png")
##全波段差值、归一化植被指数
#计算
np.save(os.path.abspath("data/output_28.npy"),output_28)
np.save(os.path.abspath("data/data_band_28.npy"),data_band_28)

runing = 'python {}'.format(os.path.abspath("script/double_band_index.py"))
data_path = "{}\n{}\n".format(os.path.abspath("data/data_band_28.npy"),os.path.abspath("data/output_28.npy"))
data_path = bytes(data_path,encoding="utf8")
a = subprocess.run(args=runing,input=data_path,capture_output=True)
result_output_28 = np.load(os.path.abspath("data/result_output_28.npy"))
#绘图
section6_function.plot_doubelband(result_output_28,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/5.png")
#建立线性模型
DVI = section6_function.doubelband_10(result_output_28[0],data_band_28,index='DVI')
NDVI = section6_function.doubelband_10(result_output_28[2],data_band_28,index='NDVI')
pca_DVI = PCA(n_components=1)
pca_NDVI = PCA(n_components=1)
DVI_pca = pca_DVI.fit_transform(DVI)
NDVI_pca = pca_NDVI.fit_transform(NDVI)
model_lines_DVI,evaluation_DVI = section6_function.evaluation_system(DVI_pca,output_28)
model_best_DVI = model_lines_DVI[np.argmax(evaluation_DVI[:,0])]
evaluation_best_DVI = evaluation_DVI[np.argmax(evaluation_DVI[:,0])]
model_lines_NDVI,evaluation_NDVI = section6_function.evaluation_system(NDVI_pca,output_28)
model_best_NDVI = model_lines_NDVI[np.argmax(evaluation_NDVI[:,0])]
evaluation_best_NDVI = evaluation_NDVI[np.argmax(evaluation_NDVI[:,0])]
section6_function.line_scatter_plot_3(model_best_DVI,evaluation_best_DVI,DVI_pca,model_best_NDVI,evaluation_best_NDVI,NDVI_pca,output_28,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/6.png")
##第三类
#runing_2 = 'python {}'.format(os.path.abspath("script/eemd_envelope.py"))
#a = subprocess.run(args=runing_2,capture_output=True)
data_sum_Enorm = np.load("F:/光谱数据/data_sum_Enorm.npy")
index = np.delete(index,[5,40,315],axis=0)

data_sum_Enorm_28 = data_sum_Enorm[read_data.select_data_2(index,day='28')]
corr_area = []
output_28 = output[read_data.select_data_2(index,day='28')]
for i in range(9):
    corr_area.append(pearsonr(data_sum_Enorm_28[:,i],output_28))
#建立线性模型
model_lines_1,evaluation_1 = section6_function.evaluation_system(data_sum_Enorm_28[:,3].reshape(-1,1),output_28)
model_best_1 = model_lines_1[np.argmax(evaluation_1[:,0])]
evaluation_best_1 = evaluation_1[np.argmax(evaluation_1[:,0])]
model_lines_2,evaluation_2 = section6_function.evaluation_system(data_sum_Enorm_28[:,1].reshape(-1,1),output_28)
model_best_2 = model_lines_2[np.argmax(evaluation_2[:,0])]
evaluation_best_2 = evaluation_2[np.argmax(evaluation_2[:,0])]
###建立预测模型
##数据准备
X_1 = first_data_PCA_1
X_2 = second_data_1
X_3 = DVI_pca
X_4 = NDVI_pca
X_5 = data_sum_Enorm_28[:,3]
X_5 = X_5.reshape((-1,1))
X_output_28 = np.concatenate((X_1,X_2,X_3,X_4,X_5),axis=1)
Y_output_28 = output_28
##保存数据
np.save(os.path.abspath("data/X_output_28.npy"),X_output_28)
np.save(os.path.abspath("data/Y_output_28.npy"),Y_output_28)
##读取数据
X = np.load(os.path.abspath("data/X_output_28.npy"))
Y = np.load(os.path.abspath("data/Y_output_28.npy"))
##岭回归
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from joblib import dump,load
from sklearn.preprocessing import StandardScaler
alphas = [0.01,0.1,0.5,1,10,100]
models_Ridge = [Ridge(alpha=i) for i in alphas]
models_Ridge = section6_function.plot_Ridge_learning_curves(models_Ridge,alphas,X,Y,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/7.png")
section6_function.save_models(models_Ridge,name='Ridge')
##决策树回归
models_Tree = [DecisionTreeRegressor(max_depth=2),DecisionTreeRegressor(max_depth=3),DecisionTreeRegressor(max_depth=4)]
names = [r'$R^1_{pca(521-614)}$',r'MCARI',r'DV$I^1_{pca}$',r'NDV$I^1_{pca}$',r'$A_{969-1072}$']
models_Tree = section6_function.plot_Tree_learning_curves(models_Tree,X,Y,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/8.png")
section6_function.plot_Tree(models_Tree[0],names,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/9.png")
section6_function.save_models(models_Tree,name='Tree')
##支持向量机回归
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X  = scaler.fit_transform(X)
models_SVR = [  SVR(kernel='linear',epsilon=0.01,C=1),
                SVR(kernel='rbf',epsilon=0.01,C=0.1,gamma=0.2),
                SVR(kernel='poly',epsilon=0.001,C=0.1,degree=2),
             ]
models_SVR = section6_function.plot_SVR_learning_curves(models_SVR,X,Y,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/10.png")
section6_function.save_models(models_SVR,name='SVR')
#绘制三个模型的散点图
X = np.load(os.path.abspath("data/X_output_28.npy"))
models = [load(os.path.abspath('models/output_28_Ridge_2.joblib')),
            load(os.path.abspath('models/output_28_Tree_0.joblib')),
            load(os.path.abspath('models/output_28_SVR_1.joblib'))]

section6_function.ridge_tree_svr_scatter(models,X,Y,path="F:/吟咏之间，吐纳珠玉之声/论文改2/6章/图片3/11.png")



index_0 = index_28[read_data.select_data_2(index_28,year='2016')]
X_0 = X[read_data.select_data_2(index_28,year='2016')]
_ = read_data.pre_statis_test(X_0,index_0)

