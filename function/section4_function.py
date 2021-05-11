import numpy as np
from scipy.stats import pearsonr
from sympy import sympify,Symbol

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import tree


from joblib import dump,load


"""
计算选择的四类值植指数的值，并计算植被指数与相关生理指数的决定系数R方
"""
"""第一类所需要的函数"""
def corr_singlewave(data_band,crop_index):
    """
    函数作用：计算光谱数据中单一波段反射率与植物生理指数的决定系数大小（R^2）及检验结果
    input: 
        data_band 原始冠层光谱反射率，二维数组每一行为一条光谱数据
        crop_index  植物生理指数
    out：
         R_square  决定系数
         p  假设检验结果
    """
    wave_len = data_band.shape[1]
    R_square = np.zeros(wave_len,dtype=np.float32)
    p = R_square.copy()
    for i in range(wave_len):
        pear = pearsonr(data_band[:,i],crop_index)
        R_square[i] = pear[0]**2
        p[i] = pear[1]
    return R_square,p

def high_singlewave_10per(data_band,R_square):
    """
    函数作用：根据函数corr_singlewave计算所得的结果来选出决定系数前10%的波段，并计算这些波段相互的相关性
    input:
        data_band  原始冠层光谱反射率
        R_square  每个波段对应的决定系数
    out:
        high_index  前10%（100个）决定系数的波段索引，如果要得到波段需要加350
        high_singlewave_interact  决定系数前10%波段间相互相关性（100*100）
    """
    wave_len = int(len(R_square)*0.1)
    high_index = np.argsort(R_square)[-wave_len:]
    high_singlewave = data_band[:,np.sort(high_index)].copy()
    high_singlewave_interact = np.zeros((wave_len)**2,dtype=np.float32)
    k = 0
    for i in range(wave_len):
        for j in range(wave_len):
            high_singlewave_interact[k] = (pearsonr(high_singlewave[:,i],high_singlewave[:,j])[0])**2
            k = k + 1
    high_singlewave_interact = high_singlewave_interact.reshape(wave_len,wave_len)
    return high_index,high_singlewave_interact

def high_singlewave_extent(high_index):
    """
    函数作用：根据函数high_singlewave_10_per计算所得到单波段R方最高10%的波段索引，得出对应真实波段下
            R方最高的波段范围，即将多个难以表述的敏感单波段融合为多个波段范围。
    input:
        high_index  由high_singlewave_10per计算所得的前10%决定系数的波段索引。
    out:
        start_end  列表，列表中包含多个元组，每个元组代表每个敏感范围，其中元组的第一个元素为起始波段，
                   第二个元素为结束波段。
    """
    high_index_sort = np.sort(high_index)
    high_index_misplace = np.hstack((high_index_sort[1:],high_index_sort[-1]))
    high_index_space = high_index_misplace - high_index_sort
    start_end = []
    start = 0
    for end in np.where(high_index_space != 1)[0]:
        start_end.append((start+high_index_sort[0]+350,end+high_index_sort[0]+350))
        start = end+1
    return start_end
"""
第二类所需要的函数
"""
def find_index(symlists,refl):
    """
    函数的作用：寻找一个数学表达式中的参数（此参数形式是固定的具体见corr_customer函数说明），后根据对应的
            参数找出在光谱数据中对应的值，简单的例子：
            如数学表达式'R810-Rblue/R460'，表示在810nm的反射率减去蓝光平均反射率后除以460nm的反射率，通
            过此函数就会通过R810,Rblue,R460的参数名称去光谱数据中refl去索引这些参数对应的反射率，并通过
            一个元组列表返回以便corr_customer函数使用，值得注意的是此函数是一个过渡函数，不需要手动调用，
            这里只是提一嘴。
    input:
        symlists  数学表达式中的参数列表，如上表达式中这个参数的值为['R810','Rblue','R460']
        refl   一个高光谱数据
    out:
        indexs  各个参数对应的数值如['R810','Rblue','R460']返回三个值，810反射率、蓝光平均反射率、460反射率
    """
    indexs = []
    for sym in symlists:
        strsym = str(sym)
        if strsym[1:] == 'blue':
            indexs.append((strsym,np.mean(refl[435-350:450-350])))
        elif strsym[1:] == 'green':
            indexs.append((strsym,np.mean(refl[492-350:577-350])))
        elif strsym[1:] == 'red':
            indexs.append((strsym,np.mean(refl[622-350:760-350])))
        elif strsym[1:] == 'nir':
            indexs.append((strsym,np.mean(refl[780-350:1100-350])))
        else:
            indexs.append((strsym,refl[int(strsym[1:])-350]))
    return indexs

def corr_customer(customer_functions,data_band,crop_index):
    """
    原谅我贫瘠的知识只能写循环套用，性能应该不太行。
    函数作用：输入多个计算特定计算植被指数的数学公式（其参数使用特定的格式，格式见后文），多组高光谱数据（一行为一条），
            生理指数和植被指数的名称（非必须），即可返回这些植被指数与对应生理指数相关性大小和显著性检验结果。
            参数格式：
            使用“R...”格式，...内容有五种情况:
            1、最常见的，直接是某个波长对应的反射率其参数为R+波长，如R810、R755等；
            2、天依蓝波段，此时参数名称是固定的即Rblue，其返回蓝光范围（435-450nm）反射率平均值；
            3、宝强绿波段，名称也为固定Rgreen(492-577nm)；
            4、慈父红波段，名称固定Rred(622-760nm)；
            5、近红外短波，名称固定Rnir(780-1100nm)
            使用以上格式的参数定义一系列计算植被指数的数学公式从而组成一个列表
    input:
        customer_functions  数学公式列表，注意一定是列表，就是求一个植被指数也需要使用方括号括起来，
                            如['(R810-R510)/Rblue','(Rnir-Rgreen)/(1+R750)']
        data_band  高光谱数据组成的数组，一行为一条高光谱数据
        crop_index  要求相关性的生理指数
    out:
        customer_data  计算所得已有的植被指数数据
        customer_R_square  各种植被指数与某植物生理指数的决定系数R方
    """
    customer_data = []
    for i in range(data_band.shape[0]):
        single_data = []
        for strfunct in customer_functions:
            symfunct = sympify(strfunct)
            indexs = find_index(symfunct.atoms(Symbol),data_band[i])
            single_data.append(symfunct.subs(indexs).evalf())
        customer_data.append(single_data)
    customer_data = np.array(customer_data,dtype=np.float32)#前面算出的数据类型不是公认的浮点型
    customer_data = customer_data.reshape(data_band.shape[0],-1)
    customer_R_square = []
    
    for i in range(customer_data.shape[1]):
        pear = pearsonr(customer_data[:,i],crop_index)
        customer_R_square.append((pear[0])**2)

    return customer_data,customer_R_square

def all_customer_interact(customer_data,customer_R_square):
    """
    函数作用：根据function_corr文件中的corr_customer函数计算所得的19个已有植被指数数组customer_data，
            依据customer_R_square排序计算出这19个已有植被指数的相关关系，结果为customer_interact
    input:
        customer_data  由customer_data函数计算所得每条曲线的19个植被指数（size 327*19）
        customer_R_square  由customer_data函数所得的每个植被指数与某生理指数的决定系数R方
    out:
        customer_interact  这19个植被指数相互计算所得到的决定系数R方矩阵（size 19*19）
    """
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
    return customer_interact

def doubelband_10(result_single,data_band,index='NDVI'):
    #函数说明
    """
    函数作用：输入之前全波段组合的植株指数计算结果和名称（DVI，NDVI），返回其决定系数前10%的植被指数值
    input:
            result_single: 全波段组合运行出结果的一部分，如result[0]
            index: 为要提取出的植被指数的名称，只支持差值与归一化（DVI、NDVI）默认为NDVI
    out:
            NDVI_10: 为决定系数前10%的某一类型的植被指数
    """
    a = result_single.reshape(1000,1000)
    a = a**2
    c = result_single**2
    c = np.where(np.isnan(c),0,c)
    c = np.sort(c)
    max_10 = c[-100000]
    index_10 = np.argwhere(a > max_10)
    mask = index_10[:,0] > index_10[:,1]
    index_10 = index_10[mask]
    NDVI_a = data_band[:,index_10[:,0]].copy()
    NDVI_b = data_band[:,index_10[:,1]].copy()
    if index == 'NDVI':
        NDVI_10 = (NDVI_a - NDVI_b) / (NDVI_a + NDVI_b) 
    elif index == 'DVI':
        NDVI_10 = NDVI_a - NDVI_b
    else:
        print("需要计算的参数不存在！")
    return NDVI_10


"""
模型检测
"""
from sklearn.linear_model import LinearRegression
def evaluation_system(X,label):
    #函数说明
    """
    函数作用：系统地重复多次地去评价一个模型的好坏，将所有的数据输入后会随机按比例7：3取出训练数据和测试数据，训练完成后使用测试数据计算模型的R方、RMSE、MAE，重复训练和测试50次将结果汇总后返回
    input: 
            X：所有的特征数据
            label：所有的标签数据
    out：
            models: 训练完成的50个模型
            data_evaluation: 所有训练过程中的评估参数
    """
    data_evaluation = np.zeros((50,3))
    models = []
    for j in range(50):
        model = LinearRegression()
        random_index = [i for i in range(len(label))]
        np.random.shuffle(random_index)
        train_data = X[random_index[0:262]]
        train_label = label[random_index[0:262]]
        test_data = X[random_index[262:]]
        test_label = label[random_index[262:]]
        model.fit(train_data,train_label)
        R = model.score(test_data,test_label)
        RMSE = np.sqrt(np.sum((model.predict(X)-label)**2) / len(test_label))
        MAE = np.sum(np.abs(model.predict(X)-label)) / len(test_label)
        data_evaluation[j,0] = R
        data_evaluation[j,1] = RMSE
        data_evaluation[j,2] = MAE
        models.append(model)

    return models,data_evaluation







"""
绘制图像
"""
"""第一类绘制图像"""
def corr_singlewave_plot(R_square,P,start_end=None,path=None):
    """
    函数作用：绘制随波长增长，植物某生理指数与某波段反射率决定系数R方的变化图，并绘制出不显著的
            波段范围，同时如果传入start_end列表会在图中标注出敏感波段的范围
    input:
        R_square  由文件function_corr中的corr_singlewave函数计算所得到的决定系数
        P  计算决定系数R方时检验结果
        start_end  由文件function_corr中的high_singlewave_extent函数计算所得到的敏感波段范围
                   默认值为无，如果传入则会在图中标记敏感波段
        path  如果你想保存这张漂亮的图片的话将保存路径写道这边
    """
    wave = np.arange(350,1350)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(5,4),dpi=200)
    ax0 = fig.add_subplot(111)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_color('black')
    ax0.spines['bottom'].set_color('black')
    ax0.plot(wave,R_square,':',color='black',label='决定系数$R^2$')
    P_001 = np.where(P>0.01)[0]
    P_001y = R_square[P_001]
    P_001x = P_001 + 350
    ax0.scatter(P_001x,P_001y,s=10,marker='_',c='black',label='P值>0.01')
    labels = ax0.get_yticklabels()
    [label.set_color('black') for label in labels]
    [label.set_size(10) for label in labels]
    ax0.set_xticks([350,600,850,1100,1350])
    ax0.set_xticklabels([350,600,850,1100,1350],fontsize=10,color='black')
    ax0.set_ylabel('决定系数$R^2$',fontsize=10,color='black')
    ax0.set_xlabel('波长（nm）',fontsize=10,color='black')
    ax0.legend(edgecolor='w',fontsize=9,loc=0)
    plt.tight_layout()

    if start_end:
        square = []
        extent = ax0.axis()
        for i in start_end:
            if i[1] - i[0] > 10:
                square.append(patches.Rectangle((i[0],extent[2]),i[1]-i[0],extent[3]-extent[2]))
        ax0.add_collection(PatchCollection(square, facecolor='cornflowerblue', alpha=0.5,edgecolor=None))

    if path:
        plt.savefig(path,dpi=200)
    plt.show()
         
def high_interact_plot(high_index,high_singlewave_interact,path=None):
    """
    函数作用：绘制敏感波段内两两波段相互的决定系数R方的彩色图像，以判断这些敏感波段是否具有共线性
    input:
        high_index   由文件function_corr中函数high_singlewave_10per计算所得前10%（100个）决
                     定系数的波段索引
        high_singlewave_interact  由文件function_corr中函数high_singlewave_10per计算所得的
                                  R方前10%的波段反射率两两计算得到的R方矩阵
        path  如果你想保存这张漂亮的图片的话将路径传入到此参数中
    """
    high_index_sort = np.sort(high_index)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(5,4),dpi=200)
    ax0 = fig.add_subplot(111)
    im = ax0.imshow(high_singlewave_interact,origin='lower',cmap=plt.cm.jet,interpolation='nearest')
    ax0.set_yticks([0,99])
    ax0.set_yticklabels([str(high_index_sort[0]+350)+'nm',str(high_index_sort[-1]+350)+'nm'],fontsize=10,color='black')
    ax0.set_xticks([0,99])
    ax0.set_xticklabels([str(high_index_sort[0]+350)+'nm',str(high_index_sort[-1]+350)+'nm'],fontsize=10,color='black')
    fig.colorbar(im,ax=ax0,shrink=1)

    if path:
        plt.savefig(path,dip=200)
    plt.show()

def PCA_var_plot(cumsum,path=None):
    """
    ###^&&&注意！此绘图函数是以（叶面积指数）为例书写的，应该不适合之后的使用
    函数作用：对10%的敏感参数进行主成分分析，绘制其解释方差随特征数量的变化图
    input:
        cumsum  对100个敏感参数进行主成分分析后各个特征的累积解释方差
        path  如果你想将这张漂亮的图片保存的话请将保存路径传个此参数
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(5,4),dpi=200)
    ax0 = fig.add_subplot(111)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_color('black')
    ax0.spines['bottom'].set_color('black')
    ax0.plot(cumsum,color='black',lw=1)
    ax0.set_yticks([0.9990,0.9995,1])
    ax0.set_yticklabels([0.9990,0.9995,1],fontsize=10,color='black')
    ax0.set_ylabel('解释方差',fontsize=10,color='black')
    ax0.set_xlabel('波段数量',fontsize=10,color='black')

    #由于数据跳动较小，无法在原图解释，因此放大了增长部分进行解释说明
    ax0_1 = fig.add_axes([0.3,0.2,0.5,0.4])
    ax0_1.plot(cumsum[0:5],color='black',lw=1)
    ax0_1.axhline(y=cumsum[1],xmax=0.3,linestyle=':',color='black',lw=1)
    ax0_1.axvline(x=1,ymax=0.8,linestyle=':',color='black',lw=1)
    ax0_1.set_yticks([0.9990,0.9995,cumsum[1],1])
    ax0_1.set_yticklabels([0.9990,0.9995,0.9997,1],fontsize=5,color='black')
    ax0_1.set_xticks([0,1,2,3,4])
    ax0_1.set_xticklabels([0,1,2,3,4],fontsize=5,color='black')
    ax0_1.set_title('前4个波段数量下的解释方差',fontsize=7,color='black')
    plt.tight_layout()
    if path:
        plt.savefig(path,dpi=200)
    plt.show()

def line_scatter_plot(model_lines,evaluation,X,lai,path=None):
    #函数说明
    """
    函数作用：绘制拟合最好线性模型的真实值与预测值叶面积指数的散点图及拟合方程
    input:
        model_lines: 为sklearn中的模型对象，是50次建立后训练完成后的模型对象
        evaluation: 50个模型的评估参数分别为R方、RMSE、MAE
        X: 使用pca降维后的特征值
        lai: 叶面积指数的真实值
        path: 如果要保存这张漂亮的图片将保存路径写在这边
    out:
        无
    """
    model_best = model_lines[np.argmax(evaluation[:,0])]
    lai_predict = model_best.predict(X)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(5,4),dpi=200)
    ax0 = fig.add_subplot(111)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_color('black')
    ax0.spines['bottom'].set_color('black')

    ax0.scatter(lai,lai_predict,c='black',s=10)
    line = np.linspace(np.min([lai,lai_predict]),np.max([lai,lai_predict]),num=50,endpoint=True)
    ax0.plot(line,line,color='black',label='1:1线')
    ax0.set_xticks([4.6,5.0,5.4,5.8])
    ax0.set_xticklabels([4.6,5.0,5.4,5.8],fontsize=10,color='black')
    ax0.set_yticks([4.6,5.0,5.4,5.8])
    ax0.set_yticklabels([4.6,5.0,5.4,5.8],fontsize=10,color='black')
    ax0.set_ylabel('LAI模拟值',fontsize=10,color='black')
    ax0.set_xlabel('LAI实测值',fontsize=10,color='black')
    ax0.legend(edgecolor='w',fontsize=9,loc=0)
    ax0.text(4.6,5.8,'LAI = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}  '.format(model_best.coef_[0],model_best.intercept_,np.max(evaluation[:,0])))

    plt.tight_layout()
    if path:
        plt.savefig(path,dpi=200)
    plt.show()

"""第二类参数绘制图像"""
def customer_interact_plot(customer_interact,customer_R_square,customer_names,path=None):
    """
    函数作用：绘制19个植被指数之间的决定系数R方的矩阵图，及植被指数与某植物生理参数决定系数柱形图
    input:
        customer_interact  由文件function_corr中的all_customer_interact函数计算所得出的植被
                           指数之间的决定系数矩阵
        customer_R_square  由文件function_corr中的corr_customer函数计算所得的各个植被指数的
                           决定系数大小
        customer_names  各个植被指数的具体名称，注意需要是array数组
        path  如果要保存这张漂亮的图片将保持的路径传入到此参数
    """
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
    ax2.set_xticklabels([customer_names[np.argsort(customer_R_square)[::-1]][0].split("*")[0],
                    customer_names[np.argsort(customer_R_square)[::-1]][4].split("*")[0],
                    customer_names[np.argsort(customer_R_square)[::-1]][9].split("*")[0],
                    customer_names[np.argsort(customer_R_square)[::-1]][14].split("*")[0],
                    customer_names[np.argsort(customer_R_square)[::-1]][18].split("*")[0]],
                    fontsize=10,color='k')
    ax2.set_xlabel('已有的植被指数类型',fontsize=10,color='k')
    
    if path:
        plt.savefig(path,dpi=200)
    plt.show()

def line_scatter_plot_2(model_best,evaluation_best,second_data_1,lai,second_data_1_name,path=None):
    #函数说明
    """
    函数作用：绘制挑选出的7个（6+1）已有的植被指数的线性模型的真实值与预测值的图像散点图及拟合方程
    input:
        model_best: 7个植被指数对应的最佳的线性回归模型
        evaluation_best: 7个最佳模型对应的评估参数R方、RMSE、MAE
        second_data_1：计算所得的植被指数加pca后的数据
        lai: 叶面积指数的真实值
        second_data_1_name：植被指数的名称
        path: 如果要保存这张漂亮的图片将保存路径写在这边
    out:
        无
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(10,16),dpi=200)
    for k,i,j,name in zip(range(7),model_best,evaluation_best,second_data_1_name):
        ax = fig.add_subplot(4,2,k+1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        lai_predict = i.predict(second_data_1[:,k].reshape(-1,1))
        ax.scatter(lai,lai_predict,c='black',s=10)
        line = np.linspace(np.min([lai,lai_predict]),np.max([lai,lai_predict]),num=50,endpoint=True)
        ax.plot(line,line,color='black',label='1:1线')
        ax.set_xticks([4.6,5.0,5.4,5.8])
        ax.set_xticklabels([4.6,5.0,5.4,5.8],fontsize=10,color='black')
        ax.set_yticks([4.6,5.0,5.4,5.8])
        ax.set_yticklabels([4.6,5.0,5.4,5.8],fontsize=10,color='black')
        ax.set_ylabel('LAI模拟值',fontsize=10,color='black')
        ax.set_xlabel('LAI实测值',fontsize=10,color='black')
        ax.legend(edgecolor='w',fontsize=9,loc=0)
        ax.text(4.6,5.7,'LAI = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}**'.format(i.coef_[0],i.intercept_,j[0]),fontsize=13)
        if k == 6:
            ax.text(5.0,6.0,'$index^1$$_{pca}$',fontsize=13)
        else:
            ax.text(5.0,6.0,'{:s}'.format(name.split("*")[0]),fontsize=13)
        plt.tight_layout()
    if path:
        plt.savefig(path,dpi=200)
    plt.show()

def plot_doubelband(result,path=None):
    #函数说明
    """
    函数作用：将之前运行出的结果result数组进行绘图，总共有四幅图分别为全波段组合下差值、归一化植被指数与水稻生理指数的决定系数，及前10%的区域
    input：
            result  使用文件test6.py运行出的结果
            path  如果想保存这张漂亮的图片将绝对路径写在这里
    out:
            无
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    a = result[0].reshape(1000,1000)
    a = a**2
    c = result[0]**2
    c = np.where(np.isnan(c),0,c)
    c = np.sort(c)
    max_DVI10 = c[-100000]
    c = np.where(a > max_DVI10,a,np.nan)

    b = result[2].reshape(1000,1000)
    b = b**2
    d = result[2]**2
    d = np.where(np.isnan(d),0,d)
    d = np.sort(d)
    max_NDVI10 = d[-100000]
    d = np.where(b >max_NDVI10,b,np.nan)

    fig = plt.figure(figsize=(20,16))
    
    ax1 = fig.add_subplot(221)
    im = ax1.imshow(a,origin='lower',cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(im,ax=ax1,shrink=1)
    ax1.set_yticks([0,200,400,600,800,1000])
    ax1.set_yticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax1.set_xticks([0,200,400,600,800,1000])
    ax1.set_xticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax1.set_xlabel('波长 Wavelength(nm)',fontsize=14,color='black')
    ax1.set_ylabel('波长 Wavelength(nm)',fontsize=14,color='black')
    plt.title('a,差值植被指数与叶面积指数决定系数$R^2$',fontsize=20,color='black')

    ax2 = fig.add_subplot(222)
    im = ax2.imshow(b,origin='lower',cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(im,ax=ax2,shrink=1)
    ax2.set_yticks([0,200,400,600,800,1000])
    ax2.set_yticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax2.set_xticks([0,200,400,600,800,1000])
    ax2.set_xticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax2.set_xlabel('波长 Wavelength(nm)',fontsize=14,color='black')
    ax2.set_ylabel('波长 Wavelength(nm)',fontsize=14,color='black')
    plt.title('b,归一化植被指数与叶面积指数决定系数$R^2$',fontsize=20,color='black')
    
    ax3 = fig.add_subplot(223)
    im = ax3.imshow(c,origin='lower',cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(im,ax=ax3,shrink=1)
    ax3.set_yticks([0,200,400,600,800,1000])
    ax3.set_yticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax3.set_xticks([0,200,400,600,800,1000])
    ax3.set_xticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax3.set_xlabel('波长 Wavelength(nm)',fontsize=14,color='black')
    ax3.set_ylabel('波长 Wavelength(nm)',fontsize=14,color='black')
    plt.title('c,决定系数前10%差值植被指数',fontsize=20,color='black')
    
    ax4 = fig.add_subplot(224)
    im = ax4.imshow(d,origin='lower',cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(im,ax=ax4,shrink=1)
    ax4.set_yticks([0,200,400,600,800,1000])
    ax4.set_yticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax4.set_xticks([0,200,400,600,800,1000])
    ax4.set_xticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax4.set_xlabel('波长 Wavelength(nm)',fontsize=14,color='black')
    ax4.set_ylabel('波长 Wavelength(nm)',fontsize=14,color='black')
    plt.title('d,决定系数前10%归一化植被指数',fontsize=20,color='black')
    plt.tight_layout()

    if not path is None:
        plt.savefig(path)
    plt.show()

def line_scatter_plot_3(model_best_DVI,evaluation_best_DVI,DVI_pca,model_best_NDVI,evaluation_best_NDVI,NDVI_pca,lai,path=None):
    #函数说明
    """
    函数作用：绘制拟合好的降维后的DVI和NDVI线性模型的真实值与预测值LAI的散点图及拟合方程
    input:
        model_best_DVI: 为sklearn中模型对象，是关于DVI指数50次建模效果最好的模型
        evaluation_best_DVI: DVI最佳模型对应的评估参数
        DVI_pca: 使用pca将前10%DVI降维后的的数据
        model_best_NDVI: 关于NDVI指数50次建模效果最好的模型
        evaluation_best_NDVI: NDVI最佳模型对应的评估参数
        NDVI_pca: 使用pca将前10%NDVI降维的数据
        lai: 叶面积指数的真实值
        path: 如果要保存这张漂亮的图片将保存路径写在这边
    out:
        无
    """
    lai_predict_DVI = model_best_DVI.predict(DVI_pca)
    lai_predict_NDVI = model_best_NDVI.predict(NDVI_pca)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(10,4),dpi=200)
    ax0 = fig.add_subplot(121)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_color('black')
    ax0.spines['bottom'].set_color('black')
    ax0.scatter(lai,lai_predict_DVI,c='black',s=10)
    line = np.linspace(np.min([lai,lai_predict_DVI]),np.max([lai,lai_predict_DVI]),num=50,endpoint=True)
    ax0.plot(line,line,color='black',label='1:1线')
    ax0.set_xticks([4.6,5.0,5.4,5.8])
    ax0.set_xticklabels([4.6,5.0,5.4,5.8],fontsize=10,color='black')
    ax0.set_yticks([4.6,5.0,5.4,5.8])
    ax0.set_yticklabels([4.6,5.0,5.4,5.8],fontsize=10,color='black')
    ax0.set_ylabel('LAI模拟值',fontsize=10,color='black')
    ax0.set_xlabel('LAI实测值',fontsize=10,color='black')
    ax0.legend(edgecolor='w',fontsize=9,loc=0)
    ax0.text(4.6,5.8,'LAI = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}  '.format(model_best_DVI.coef_[0],model_best_DVI.intercept_,evaluation_best_DVI[0]))
    ax0.text(5.0,6.0,'$a:DVI^1$$_{pca}$',fontsize=11)

    ax1 = fig.add_subplot(122)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    ax1.scatter(lai,lai_predict_NDVI,c='black',s=10)
    line = np.linspace(np.min([lai,lai_predict_NDVI]),np.max([lai,lai_predict_NDVI]),num=50,endpoint=True)
    ax1.plot(line,line,color='black',label='1:1线')
    ax1.set_xticks([4.6,5.0,5.4,5.8])
    ax1.set_xticklabels([4.6,5.0,5.4,5.8],fontsize=10,color='black')
    ax1.set_yticks([4.6,5.0,5.4,5.8])
    ax1.set_yticklabels([4.6,5.0,5.4,5.8],fontsize=10,color='black')
    ax1.set_ylabel('LAI模拟值',fontsize=10,color='black')
    ax1.set_xlabel('LAI实测值',fontsize=10,color='black')
    ax1.legend(edgecolor='w',fontsize=9,loc=0)
    ax1.text(4.6,5.8,'LAI = {:.2f}x + {:.2f}\n$R^2$ = {:.2f}  '.format(model_best_NDVI.coef_[0],model_best_NDVI.intercept_,evaluation_best_NDVI[0]))
    ax1.text(5.0,6.0,'$b:NDVI^1$$_{pca}$',fontsize=11)
    plt.tight_layout()
    if path:
        plt.savefig(path,dpi=200)
    plt.show()


"""模型训练过程"""

def save_models(models,name):
    #函数说明
    """
    函数作用：将一些训练完成的模型保存在子目录models下
    input: 
            models: 训练完成的一些模型
            name: 字符串，指定模型的类型如Ridge便于保存
    out:
            无
    """
    for i,model in enumerate(models):
        filename  = 'models/lai_' + name + '_' + str(i) + '.joblib'
        dump(model,os.path.abspath(filename))


"""岭回归训练过程（绘图及训练）"""
def plot_Ridge_learning_curves(models,alphas,X,y,path=None):
    #函数说明
    """
    函数作用：绘制6个不同正则参数的岭回归训练曲线，并将训练结果模型返回
    input:
            models: 已经建立完成的6个岭回归模型（区别是正则化参数不同）
            alphas: 具体的正则化参数数值
            X: 训练数据
            y: 标签数据
            path: 图片保存路径
    out:
            models: 返回已经训练完成的模型
    注：训练过程具有不稳定性，需要多次训练取最优的模型！
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(15,8),dpi=200)
    for model,i,alpha in zip(models,list(range(1,7)),alphas):
        X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
        train_errors,val_errors = [],[]
        for m in range(1,len(X_train)):
            model.fit(X_train[:m],y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict,y_val))

        ax = fig.add_subplot(2,3,i)
        ax.plot(np.sqrt(train_errors),color='black',linestyle='-',label='训练集')
        ax.plot(np.sqrt(val_errors),color='black',linestyle=':',label='验证集')
        ax.legend(edgecolor='w',fontsize=13)
        ax.set_yticks([0,0.1,0.2,0.3,0.4,])
        ax.set_yticklabels([0,0.1,0.2,0.3,0.4],fontsize=14,color='black')
        plt.ylim(0,0.4)
        if i==1 or i==4:
            ax.set_ylabel('RMSE误差',fontsize=14,color='black')
        if i==4 or i==5 or i==6:
            ax.set_xlabel('训练数量',fontsize=14,color='black')
        plt.title(r'$\alpha$={}'.format(alpha),fontsize=16,color='black')

        timely_error = (np.sqrt(train_errors[-50].mean()) + np.sqrt(val_errors[-50].mean()))/2.0
        distinguish = np.sqrt(train_errors[-50].mean()) - np.sqrt(val_errors[-50].mean())
        R2 = model.score(X,y)
        print('alpha值为{}时，模型最后的误差为{:.3f}，区别为{:.3f}，拟合效果为{:.3f}'.format(alpha,timely_error,distinguish,R2))
    plt.tight_layout()
    if not path is None:
        plt.savefig(path)
    plt.show()
    return models

"""决策树回归训练过程（绘图及训练）"""
def plot_Tree_learning_curves(models,X,y,path=None):
    #函数说明
    """
    函数作用：绘制3个不同深度的决策树回归训练曲线，并将训练结果模型返回
    input:
            models: 已经建立完成的3个决策树回归模型（区别是最大深度不同）
            X: 训练数据
            y: 标签数据
            path: 图片保存路径
    out:
            models: 返回已经训练完成的模型
    注：训练过程具有不稳定性，需要多次训练取最优的模型！
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(15,4),dpi=200)
    for model,i in zip(models,list(range(1,4))):
        X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
        train_errors,val_errors = [],[]
        for m in range(1,len(X_train)):
            model.fit(X_train[:m],y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict,y_val))

        ax = fig.add_subplot(1,3,i)
        ax.plot(np.sqrt(train_errors),color='black',linestyle='-',label='训练集')
        ax.plot(np.sqrt(val_errors),color='black',linestyle=':',label='验证集')
        ax.legend(edgecolor='w',fontsize=13)
        ax.set_yticks([0,0.1,0.2,0.3,0.4,])
        ax.set_yticklabels([0,0.1,0.2,0.3,0.4],fontsize=14,color='black')
        plt.ylim(0,0.4)
        if i==1:
            ax.set_ylabel('RMSE误差',fontsize=14,color='black')
            plt.title(r'最大深度为2',fontsize=16,color='black')
        if i==2:
            plt.title('最大深度为3',fontsize=16,color='black')
        if i==3:
            plt.title('最大深度为4',fontsize=16,color='black')
        ax.set_xlabel('训练数量',fontsize=14,color='black')
        
        timely_error = (np.sqrt(train_errors[-50].mean()) + np.sqrt(val_errors[-50].mean()))/2.0
        distinguish = np.sqrt(train_errors[-50].mean()) - np.sqrt(val_errors[-50].mean())
        R2 = model.score(X,y)
        print('第{}个模型的最后的误差为{:.3f}，区别为{:.3f}，拟合效果为{:.3f}'.format(i,timely_error,distinguish,R2))
    plt.tight_layout()
    if not path is None:
        plt.savefig(path)
    plt.show()
    return models

def plot_Tree(model,names,path=None):
    #函数说明
    """
    函数作用：绘制一个决策树的形状
    input:
            model: 已经训练完成的决策树模型
            names: 各个标签的名称
            path: 图片保存的路径
    out:
            无
    """
    plt.figure(figsize=(30,15),dpi=200)
    dt = tree.plot_tree(model,feature_names=names,fontsize=20)
    if path:
        plt.savefig(path)
    plt.show()

"""不同核函数的支持向量机回归模型的训练过程（绘图及训练）"""
def plot_SVR_learning_curves(models,X,y,path=None):
    #函数说明
    """
    函数作用：绘制3个不同核函数（线性、高斯、多项式）支持向量机回归训练曲线，并将训练结果模型返回
    input:
            models: 已经建立完成的3个支持向量机回归模型（区别是核函数不同）
            X: 训练数据
            y: 标签数据
            path: 图片保存路径
    out:
            models: 返回已经训练完成的模型
    注：训练过程具有不稳定性，需要多次训练取最优的模型！
    """
    import warnings
    warnings.filterwarnings('ignore')
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(15,4),dpi=200)
    for model,i in zip(models,list(range(1,4))):
        X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
        train_errors,val_errors = [],[]
        for m in range(1,len(X_train)):
            model.fit(X_train[:m],y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict,y_val))

        ax = fig.add_subplot(1,3,i)
        ax.plot(np.sqrt(train_errors),color='black',linestyle='-',label='训练集')
        ax.plot(np.sqrt(val_errors),color='black',linestyle=':',label='验证集')
        ax.legend(edgecolor='w',fontsize=13)
        ax.set_yticks([0,0.1,0.2,0.3,0.4,])
        ax.set_yticklabels([0,0.1,0.2,0.3,0.4],fontsize=14,color='black')
        plt.ylim(0,0.4)
        if i==1:
            ax.set_ylabel('RMSE误差',fontsize=14,color='black')
            plt.title(r'kernel:linear',fontsize=16,color='black')
        if i==2:
            plt.title('kernel:rbf',fontsize=16,color='black')
        if i==3:
            plt.title('kernel:poly',fontsize=16,color='black')
        ax.set_xlabel('训练数量',fontsize=14,color='black')
        
        timely_error = (np.sqrt(train_errors[-50].mean()) + np.sqrt(val_errors[-50].mean()))/2.0
        distinguish = np.sqrt(train_errors[-50].mean()) - np.sqrt(val_errors[-50].mean())
        R2 = model.score(X,y)
        print('第{}个模型的最后的误差为{:.3f}，区别为{:.3f}，拟合效果为{:.3f}'.format(i,timely_error,distinguish,R2))
    plt.tight_layout()
    if not path is None:
        plt.savefig(path)
    plt.show()
    return models


"""模型应用"""
def create_data10_60(pre_data,rate=[0.024,0.113,0.611,0.002]):
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
    pre_data_2dim[:,0] = pre_data_2dim[:,0] + par_1
    pre_data_2dim[:,1] = pre_data_2dim[:,1] - par_2
    pre_data_2dim[:,2] = pre_data_2dim[:,2] + par_3
    pre_data_2dim[:,3] = pre_data_2dim[:,3] + par_4
    return pre_data_2dim

def plot_predict_10_60(pre_T1,pre_T2,pre_T3,models,path=None):
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
        pre_rbf = models[2].predict(pre_data_2dim)
        ax = fig.add_subplot(1,3,i)
        ax.plot(pre_ridge,color='black',linestyle='-',label='岭回归')
        ax.plot(pre_tree,color='black',linestyle=':',label='决策树')
        ax.plot(pre_rbf,color='black',linestyle='-.',label='支持向量机')
        ax.legend(edgecolor='w',fontsize=13,loc=2)
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