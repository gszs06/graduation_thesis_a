import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps



def select_data_2(index,strlist=None,day=None,year=None):
    #函数说明
    """
    作用为，返回要选择数据集的索引
    index: 特定的索引矩阵
    strlist: 处理索引关键字
    day: 天数索引关键字
    year: 年份索引关键字
    """
    order1 = []
    order2 = []
    order3 = []
    order = np.arange(index.shape[0])
    if not strlist is None:
        for i in range(index.shape[0]):
            if index[i,2] in strlist:
                order1.append(i)
    else:
        order1 = order
    if not day is None:
        order2 = np.where(index[:,3]==day)[0]
    else:
        order2 = order
    if not year is None:
        order3 = np.where(index[:,4]==year)[0]
    else:
        order3 = order
    order = set(order1)&set(order2)&set(order3)
    order = np.array(list(order),dtype=np.int)
    return order

def derivative(Refl,lamda1=680,lamda2=760):
    #函数说明
    """
    函数作用：求一条或多条光谱曲线的指定范围内的导数光谱，默认为红光波段，参考：天依蓝（490-530）、
    搞黄色（550-580）、苏联红（680-750）、近红1（920-980）、近红2（1000-1060）、近红3（1100-1180）
    input:
        Refl: 要求导数光谱的原光谱数据（可以是多条曲线）
        lamda1: 默认为680，要求导数波段范围的开始波段
        lamda2: 默认为760，要求导数波段范围的结束波段
    output:
        derivate_data: 指定范围内的导数光谱反射率
    """
    if Refl.ndim == 1:
        Refl = Refl.reshape((1,-1))
    derivative_data1 = Refl[:,0:(Refl.shape[1]-2)]
    derivative_data2 = Refl[:,2:Refl.shape[1]]
    derivative_data = (derivative_data2 - derivative_data1) / 2.0
    lamda1 = int(lamda1)
    lamda2 = int(lamda2)
    return derivative_data[:,lamda1-350:lamda2-350]

def side(Refl,lamda1=680,lamda2=760):
    #函数说明
    """
    函数作用：求解一个或多个光谱数据的一定范围内的三边参数
    input:
        Refl: 求解三边参数的原光谱反射率数据（支持多条）
        lamba1: 默认为680，要求三边参数波段范围的开始波段
        lamba2: 默认为760，要求三边参数波段范围的结束波段
    output:
        side_data: 三边参数数据，逐列分别为边位置(maxWave)、边最大斜率(maxRefl_d)、面积(area)
    """
    Wave = np.arange(350,1350)
    lamda1 = int(lamda1)
    lamda2 - int(lamda2)
    x = Wave[lamda1-350:lamda2-350]
    y = derivative(Refl,lamda1,lamda2)
    y = np.where(y<0,0,y)
    maxRefl_d = np.max(y,axis=1)
    maxWave = np.argmax(y,axis=1) + lamda1
    area = simps(y,x,axis=1)
    side_data = np.concatenate((maxWave.reshape((-1,1)),maxRefl_d.reshape((-1,1)),area.reshape((-1,1))),axis=1)
    return side_data

def IG_red(Refl):
    #函数说明
    """
    函数作用：使用IG高斯模型求解红边参数，返回红边位置和红谷宽度
    input:
        Refl: 需要求解IG三边参数的原始光谱数据
    output:
        IG_data: 计算返回结果，逐列分别为红边位置(lamda0)，红谷宽度(sigma)
    """   
    if Refl.ndim == 1:
        Refl = Refl.reshape((1,-1))
    Rs = np.mean(Refl[:,780-350:795-350],axis=1)
    Rs = Rs.reshape((-1,1))
    Ro = np.mean(Refl[:,670-350:675-350],axis=1)
    Ro = Ro.reshape((-1,1))
    Wave = np.arange(350,1350)
    y = Refl[:,685-350:780-350]
    x = Wave[685-350:780-350]
    y = np.sqrt(-np.log((Rs - y) / (Rs - Ro)))
    w = np.polyfit(x,y.T,1)
    lamda0 = -w[1,:] / w[0,:]
    sigma = 1.0 / np.sqrt(2*w[0,:])
    IG_data = np.concatenate((lamda0.reshape((-1,1)),sigma.reshape((-1,1))),axis=1)
    return IG_data

def pre_statis_test(data,index):
    #函数说明
    """
    函数说明：分别计算导数三边参数和IG三边参数，合并后按照处理进行分类，T1为第一类，T2为
        第二类，T3为第三类，并将类别号插入到第一列中，方便进行多重检验
    input:
        data: 需要计算的波段数据
        index: 每一个波段数据对应的索引信息
    out:
        test_data: 计算完成的数据，这个矩阵同样也会保持到data文件夹下，方便移植到R语言中运算 
    """
    side_d_data = side(data)
    side_IG_data = IG_red(data)
    side_data = np.concatenate((side_d_data,side_IG_data),axis=1)
    factor = np.zeros((side_data.shape[0],1))
    factor[select_data_2(index,strlist=['ck1','ck2','ck3'])] = 1
    factor[select_data_2(index,strlist=['p1','p2','p3'])] = 2
    factor[select_data_2(index,strlist=['m1','m2','m3'])] = 3
    test_data = np.concatenate((factor,side_data),axis=1)
    save_path = os.path.join(os.path.abspath("data"),"statis_test_data.txt")
    np.savetxt(save_path,test_data,fmt=['%d','%d','%.4f','%.3f','%d','%.2f'])
    return test_data


