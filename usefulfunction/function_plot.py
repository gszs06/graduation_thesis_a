import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

"""
此文件的作用，绘制图像
"""

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
    ax0.set_yticklabels([str(high_index_sort[0])+'nm',str(high_index_sort[-1])+'nm'],fontsize=10,color='black')
    ax0.set_xticks([0,99])
    ax0.set_xticklabels([str(high_index_sort[0])+'nm',str(high_index_sort[-1])+'nm'],fontsize=10,color='black')
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
    ax0.set_xlabel('特征数量',fontsize=10,color='black')

    #由于数据跳动较小，无法在原图解释，因此放大了增长部分进行解释说明
    ax0_1 = fig.add_axes([0.3,0.2,0.5,0.4],facecolor='lightgray')
    ax0_1.plot(cumsum[0:5],color='black',lw=1)
    ax0_1.axhline(y=cumsum[1],xmax=0.3,linestyle=':',color='black',lw=1)
    ax0_1.axvline(x=1,ymax=0.8,linestyle=':',color='black',lw=1)
    ax0_1.set_yticks([0.9990,0.9995,cumsum[1],1])
    ax0_1.set_yticklabels([0.9990,0.9995,0.9997,1],fontsize=5,color='black')
    ax0_1.set_xticks([0,1,2,3,4])
    ax0_1.set_xticklabels([0,1,2,3,4],fontsize=5,color='black')
    ax0_1.set_title('前4个特征数量下的解释方差',fontsize=7,color='black')

    if path:
        plt.savefig(path,dpi=200)
    plt.show()

###第二类光谱参数绘图
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
    ax2.set_xticklabels([customer_names[np.argsort(customer_R_square)[::-1]][0],
                    customer_names[np.argsort(customer_R_square)[::-1]][4],
                    customer_names[np.argsort(customer_R_square)[::-1]][9],
                    customer_names[np.argsort(customer_R_square)[::-1]][14],
                    customer_names[np.argsort(customer_R_square)[::-1]][18]],
                    fontsize=10,color='k')
    ax2.set_xlabel('已有的植被指数类型',fontsize=10,color='k')
    
    if path:
        plt.savefig(path,dpi=200)
    plt.show()