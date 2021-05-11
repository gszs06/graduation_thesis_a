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

###计算光谱参数
##第一类
#计算
R_square,P = section5_function.corr_singlewave(data_band,biomass)
high_index,high_singlewave_interact = section5_function.high_singlewave_10per(data_band,R_square)
start_end = section5_function.high_singlewave_extent(high_index)
#绘图
section5_function.corr_singlewave_plot(R_square,P,start_end,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/1.png")
section5_function.high_interact_plot(high_index,high_singlewave_interact,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/2.png")
#建立线性模型
high_data = data_band[:,high_index]
pca = PCA(n_components=1)
first_data_PCA_1 = pca.fit_transform(high_data)
model_lines,evaluation = section5_function.evaluation_system(first_data_PCA_1,biomass)
model_best = model_lines[np.argmax(evaluation[:,0])]
evaluation_best = evaluation[np.argmax(evaluation[:,0])]
section5_function.line_scatter_plot(model_lines,evaluation,first_data_PCA_1,biomass,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/3.png")
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
customer_names = ['EVI*','SIPI','VARI710','TCARI','SAVI','MSAVI**','OSAVI','VARIgreen**','mSR705','mND705',
        'CUR','redeage*','TVI-B-L','VARI700','TVI-3*','MSR','PSRI','MCARI','MTCI']
customer_names = np.array(customer_names)
customer_data,customer_R_square = section5_function.corr_customer(customer_functions,data_band,biomass)
customer_interact = section5_function.all_customer_interact(customer_data,customer_R_square)
#绘图
section5_function.customer_interact_plot(customer_interact,customer_R_square,customer_names,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/4.png")
#建立线性模型
high_customer_data = customer_data[:,np.argsort(customer_R_square)[-2:][::-1]]
high_customer_names = customer_names[np.argsort(customer_R_square)[-2:][::-1]]
second_data_1 = high_customer_data
second_data_1_name = high_customer_names
result = []
model_best = []
evaluation_best = [] 
for i in range(second_data_1.shape[1]):
    result.append(section5_function.evaluation_system(second_data_1[:,i].reshape(-1,1),biomass))
for model_lines,evaluation in result:
    model_best.append(model_lines[np.argmax(evaluation[:,0])])
    evaluation_best.append(evaluation[np.argmax(evaluation[:,0])])
section5_function.line_scatter_plot_2(model_best,evaluation_best,second_data_1,biomass,second_data_1_name,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/5.png")
##全波段差值、归一化植被指数
#计算
np.save(os.path.abspath("data/above_ground_biomass.npy"),biomass)
runing = 'python {}'.format(os.path.abspath("script/double_band_index.py"))
data_path = "{}\n{}\n".format(os.path.abspath("data/data_band_2.npy"),os.path.abspath("data/above_ground_biomass.npy"))
data_path = bytes(data_path,encoding="utf8")
a = subprocess.run(args=runing,input=data_path,capture_output=True)
result_biomass = np.load(os.path.abspath("data/result_biomass.npy"))
#绘图
section5_function.plot_doubelband(result_biomass,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/6.png")
#建立线性模型
DVI = section5_function.doubelband_10(result_biomass[0],data_band,index='DVI')
NDVI = section5_function.doubelband_10(result_biomass[2],data_band,index='NDVI')
pca_DVI = PCA(n_components=1)
pca_NDVI = PCA(n_components=1)
DVI_pca = pca_DVI.fit_transform(DVI)
NDVI_pca = pca_NDVI.fit_transform(NDVI)
model_lines_DVI,evaluation_DVI = section5_function.evaluation_system(DVI_pca,biomass)
model_best_DVI = model_lines_DVI[np.argmax(evaluation_DVI[:,0])]
evaluation_best_DVI = evaluation_DVI[np.argmax(evaluation_DVI[:,0])]
model_lines_NDVI,evaluation_NDVI = section5_function.evaluation_system(NDVI_pca,biomass)
model_best_NDVI = model_lines_NDVI[np.argmax(evaluation_NDVI[:,0])]
evaluation_best_NDVI = evaluation_NDVI[np.argmax(evaluation_NDVI[:,0])]
section5_function.line_scatter_plot_3(model_best_DVI,evaluation_best_DVI,DVI_pca,model_best_NDVI,evaluation_best_NDVI,NDVI_pca,biomass,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/7.png")
##第三类
#runing_2 = 'python {}'.format(os.path.abspath("script/eemd_envelope.py"))
#a = subprocess.run(args=runing_2,capture_output=True)
data_sum_Enorm = np.load("F:/光谱数据/data_sum_Enorm.npy")
corr_area = []
mask = [0]*len(data_band)
mask[5] = 1
mask[40] = 1
mask[315] = 1
biomass_mask  = np.ma.array(biomass,mask=mask)
for i in range(9):
    corr_area.append(pearsonr(data_sum_Enorm[:,i],biomass_mask.compressed()))
#建立线性模型
model_lines_1,evaluation_1 = section5_function.evaluation_system(data_sum_Enorm[:,0].reshape(-1,1),biomass_mask.compressed())
model_best_1 = model_lines_1[np.argmax(evaluation_1[:,0])]
evaluation_best_1 = evaluation_1[np.argmax(evaluation_1[:,0])]
model_lines_2,evaluation_2 = section5_function.evaluation_system(data_sum_Enorm[:,1].reshape(-1,1),biomass_mask.compressed())
model_best_2 = model_lines_2[np.argmax(evaluation_2[:,0])]
evaluation_best_2 = evaluation_2[np.argmax(evaluation_2[:,0])]

###建立预测模型
##数据准备
X_1 = np.delete(first_data_PCA_1,[5,40,315],axis=0)
X_2 = np.delete(second_data_1[:,0],[5,40,315],axis=0)
X_2 = X_2.reshape(-1,1)
X_3 = np.delete(second_data_1[:,1],[5,40,315],axis=0)
X_3 = X_3.reshape(-1,1)
X_4 = np.delete(DVI_pca,[5,40,315],axis=0)
X_5 = np.delete(NDVI_pca,[5,40,315],axis=0)
X_6 = data_sum_Enorm[:,1].reshape(-1,1)
X_biomass = np.concatenate((X_1,X_2,X_3,X_4,X_5,X_6),axis=1)
Y_biomass = np.delete(biomass,[5,40,315]).reshape(-1,1)
##保存数据
np.save(os.path.abspath("data/X_biomass.npy"),X_biomass)
np.save(os.path.abspath("data/Y_biomass.npy"),Y_biomass)
##读取数据
X = np.load(os.path.abspath("data/X_biomass.npy"))
Y = np.load(os.path.abspath("data/Y_biomass.npy"))
##岭回归
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from joblib import dump,load
from sklearn.preprocessing import StandardScaler
alphas = [0.1,0.3,0.5,0.8,1,1.2]
models_Ridge = [Ridge(alpha=i) for i in alphas]
models_Ridge = section5_function.plot_Ridge_learning_curves(models_Ridge,alphas,X,Y,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/8.png")
section5_function.save_models(models_Ridge,name='Ridge')
##决策树回归
models_Tree = [DecisionTreeRegressor(max_depth=2),DecisionTreeRegressor(max_depth=3),DecisionTreeRegressor(max_depth=4)]
names = [r'$R^1_{pca(756-864)}$',r'VARIgreen',r'MSAVI',r'NDV$I^1_{pca}$',r'DV$I^1_{pca}$',r'$A_{553-673}$']
models_Tree = section5_function.plot_Tree_learning_curves(models_Tree,X,Y,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/9.png")
section5_function.plot_Tree(models_Tree[-2],names,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/10.png")
section5_function.save_models(models_Tree,name='Tree')
##支持向量机回归
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X  = scaler.fit_transform(X)
models_SVR = [  SVR(kernel='linear',epsilon=0.1,C=1),
                SVR(kernel='rbf',epsilon=0.05,C=10,gamma=0.03),
                SVR(kernel='poly',epsilon=0.01,C=50,degree=2),
             ]
models_SVR = section5_function.plot_SVR_learning_curves(models_SVR,X,Y,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/11.png")
section5_function.save_models(models_SVR,name='SVR')
#绘制三个模型的散点图
X = np.load(os.path.abspath("data/X_biomass.npy"))
models = [load(os.path.abspath('models/biomass_Ridge_3.joblib')),
            load(os.path.abspath('models/biomass_Tree_1.joblib')),
            load(os.path.abspath('models/biomass_SVR_1.joblib'))]

section5_function.ridge_tree_svr_scatter(models,X,Y,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/12.png")
###讨论部分
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


T1_14 = np.array([[-0.13,-0.36,0.35,-4.4,17.2,40.6]])
T2_14 = np.array([[0.15,-0.53,0.4,1.42,-4.9,41.8]])
T3_14 = np.array([[0.51,-0.59,0.43,4.8,-12.4,46.47]])



models = [load(os.path.abspath('models/biomass_Ridge_3.joblib')),
            load(os.path.abspath('models/biomass_Tree_1.joblib')),
            load(os.path.abspath('models/biomass_SVR_1.joblib'))]

for i in range(3):
    if i==2:
        TT = scaler.transform(T3_14)
        print(models[i].predict(TT))
    else:
        print(models[i].predict(T3_14))



###丢弃

lai_par_ck16 = np.mean(X[read_data.select_data_2(index,['ck1','ck2','ck3'],day='21',year='2016')],axis=0)
lai_par_p16 = np.mean(X[read_data.select_data_2(index,['p1','p2','p3'],day='21',year='2016')],axis=0)
lai_par_m16 = np.mean(X[read_data.select_data_2(index,['m1','m2','m3'],day='21',year='2016')],axis=0)
lai_par_ck17 = np.mean(X[read_data.select_data_2(index,['ck1','ck2','ck3'],day='21',year='2017')],axis=0)
lai_par_p17 = np.mean(X[read_data.select_data_2(index,['p1','p2','p3'],day='21',year='2017')],axis=0)
lai_par_m17 = np.mean(X[read_data.select_data_2(index,['m1','m2','m3'],day='21',year='2017')],axis=0)
pre_T1 = [-0.43,-0.32,0.32,-7.83,11.55,38.8]
pre_T2 = [-0.25,-0.41,0.37,-2.07,6.15,43.9]
pre_T3 = [-0.25,-0.41,0.37,-2.07,6.15,43.9]





pre_T1 = [-0.48,-0.31,0.32,-7.84,24.75,35.9]
pre_T2 = [-0.55,-0.44,0.34,-5.4,10.25,32.4]
pre_T3 = [-0.55,-0.44,0.34,-5.4,10.25,32.4]

##模型读取
models = [load(os.path.abspath('models/biomass_Ridge_3.joblib')),
            load(os.path.abspath('models/biomass_Tree_1.joblib')),
            load(os.path.abspath('models/biomass_SVR_1.joblib'))]
##绘图
section5_function.plot_predict_10_60(pre_T1,pre_T2,pre_T3,models,path="F:/吟咏之间，吐纳珠玉之声/论文改2/5章/图片/13.png")


def create_data10_60(pre_data,rate=[0.035,0.005,0.003,0.341,0.755,0.47]):
    #函数说明
    """
    函数说明：生成不同初始条件下（由pre_data给定）散射比例10%-60%的四个参数，之后用于lai的预测
    input:
            pre_data: 根据不同条件计算所得的当散射辐射为10%的四个基础参数
            rate: 四个参数随散射辐射比例变化的变化率
    output:
            pre_data_2dim: 已经生成的散射辐射比例10%-60%的四个参数，矩阵（60*4）
    """
    pre_data_2dim = np.array([pre_data]*50)
    par_1 = np.array([rate[0] * i for i in range(0,50)])
    par_2 = np.array([rate[1] * i for i in range(0,50)])
    par_3 = np.array([rate[2] * i for i in range(0,50)])
    par_4 = np.array([rate[3] * i for i in range(0,50)])
    par_5 = np.array([rate[4] * i for i in range(0,50)])
    par_6 = np.array([rate[5] * i for i in range(0,50)])

    pre_data_2dim[:,0] = pre_data_2dim[:,0] + par_1
    pre_data_2dim[:,1] = pre_data_2dim[:,1] - par_2
    pre_data_2dim[:,2] = pre_data_2dim[:,2] + par_3
    pre_data_2dim[:,3] = pre_data_2dim[:,3] + par_4
    pre_data_2dim[:,4] = pre_data_2dim[:,4] - par_5
    pre_data_2dim[:,5] = pre_data_2dim[:,5] + par_6
    return pre_data_2dim

def plot_predict_10_60(pre_T1,pre_T2,pre_T3,models,X,path=None):
    #函数说明
    """
    函数说明：绘制不同初始条件下不同散射辐射比例下LAI的变化趋势
    input:
            pre_T1: T1处理下的初始条件（散射辐射比例100%）
            pre_T2: T2处理下的初始条件（散射辐射比例85%）
            pre_T3: T3处理下的初始条件（散射辐射比例85%）
            models: 训练完成的三个模型（岭回归、决策树、支持向量机）
            path: 保存绘图的路径
    output：
            无
    """
    pre_T = [pre_T1,pre_T2,pre_T3]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(15,4),dpi=200)
    for i,pre in zip(range(1,4),pre_T):
        pre_data_2dim = create_data10_60(pre)
        pre_ridge = models[0].predict(pre_data_2dim)
        pre_tree = models[1].predict(pre_data_2dim)
        scaler = StandardScaler()
        scaler = scaler.fit(X)
        pre_data_2dim_scale = scaler.transform(pre_data_2dim)
        pre_rbf = models[2].predict(pre_data_2dim_scale)
        ax = fig.add_subplot(1,3,i)
        ax.plot(pre_ridge,color='black',linestyle='-',label='岭回归')
        ax.plot(pre_tree,color='black',linestyle=':',label='决策树')
        ax.plot(pre_rbf,color='black',linestyle='-.',label='支持向量机')
        ax.legend(edgecolor='w',fontsize=13,loc=1)
        ax.set_xticks([0,10,20,30,40,50])
        ax.set_xticklabels([10,20,30,40,50,60],fontsize=14,color='black')
        ax.set_xlabel('散射辐射比例（%）',fontsize=14,color='black')
        if i == 1:
            ax.set_ylabel('LAI',fontsize=14,color='black')
            plt.title(r'基于T1处理（透光率100%）',fontsize=16,color='black')
        if i == 2:
            plt.title(r'基于T2处理（透光率85%）',fontsize=16,color='black')
        if i == 3:
            plt.title(r'基于T3处理（透光率85%）',fontsize=16,color='black')
    plt.tight_layout()
    if not path is None:
        plt.savefig(path)
    plt.show() 