#这个文件定义了计算双波段植被指数（差值，比值和归一化）与叶面积指数（可进行修改见下）的皮尔逊相关性
#运行说明：只需要将此模块全部导入脚本，使用函数time_is_life函数来计算，此函数需要传入一个参数（DVI，RVI，NDVI）来定义计算何种植被指数，
##########光谱数据和生理指数的传入需要在此函数中的data_band = np.load('D:/光谱数据/data_band_2.npy')和lai = np.load('D:/光谱数据/lai.npy')
##########两行来修改，什么？你问我为什么不直接把路径通过参数传进去？你在教我做事啊？害，还不是因为计算时间的问题，当你一个一个单核计算需要半个多
##########小时，通过这样修改使用多进程只需要十分钟左右，为什么不用省下来的二十分钟来一局简单而又刺激的深渊大乱斗呢？关于如何使用多进程详情见“使用
##########多进程来计算双波段植被指数.ipy”文件，这里不表。
#需要注意的是这里计算植被指数和相关系数的方法是一种船新的方法，如果需要了解此方法需要阅读文档“植被指数计算方法.pdf”


import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from glob import glob
from toolz import curried as c
import toolz as tz
import time

def DVI(Refl):
    #函数作用
    """
    计算一个高光谱任意两个波段组合的差值，即差值植被指数，返回一个1*1000000的数组
    Refl: 一个高光谱数据
    """
    DVI_t = []
    leng = len(Refl)
    DVI = np.zeros((leng,leng))
    for i in range(leng):
        Defl_slip = np.roll(Refl,i+1)
        DVI_t.append(Refl-Defl_slip)
    DVI_t = np.array(DVI_t)
    for i in range(leng):
        DVI[i,:] = np.roll(DVI_t[:,i][::-1],i)
    DVI = DVI.flatten()
    return DVI
def RVI(Refl):
    #函数作用
    """
    计算一个高光谱数据任意两个波段比值，即比值植被指数，返回一个1*1000000的数组
    Refl: 一个高光谱数据
    """
    leng = len(Refl)
    RNDVI = Refl
    RNDVI_flat = np.broadcast_to(RNDVI,(leng,leng)).flatten(order='F')
    RNDVI_X = np.linspace(0,leng**2,num=leng**2,endpoint=False,dtype=int)
    RNDVI_Y = np.linspace(0,leng,num=leng,endpoint=False,dtype=int)
    RNDVI_Y = np.broadcast_to(RNDVI_Y,(leng,leng)).flatten(order='C')
    RNDVI_csr = sparse.coo_matrix((RNDVI_flat,(RNDVI_X,RNDVI_Y))).tocsr()
    RNDVI_re = 1/np.array(RNDVI)
    RNDVI_flatdata = RNDVI_csr@RNDVI_re
    return RNDVI_flatdata
def NDVI(Refl):
    #函数作用
    """
    计算一个高光谱数据任意两个波段的归一化植被指数，返回一个1*1000000的数组
    Refl: 一个高光谱数据
    """
    leng = len(Refl)
    RNDVI = Refl
    RNDVI_flat = np.broadcast_to(RNDVI,(leng,leng)).flatten(order='F')
    RNDVI_X = np.linspace(0,leng**2,num=leng**2,endpoint=False,dtype=int)
    RNDVI_Y = np.linspace(0,leng,num=leng,endpoint=False,dtype=int)
    RNDVI_Y = np.broadcast_to(RNDVI_Y,(leng,leng)).flatten(order='C')
    RNDVI_csr = sparse.coo_matrix((RNDVI_flat,(RNDVI_X,RNDVI_Y))).tocsr()
    RNDVI_re = 1/np.array(RNDVI)
    RNDVI_flatdata = RNDVI_csr@RNDVI_re
    NDVI = np.array(list(map(lambda x: (1-x)/(1+x),RNDVI_flatdata)))
    return NDVI

def sum1(x,y):
    #函数作用
    """
    计算两个输入值的和，聪明的你一定会问为什么不用add，因为我这个函数支持向量加减
    x,y: 要求和的两个值
    """
    x = np.array(x)
    y = np.array(y)
    return x+y
@tz.curry
def mul1(x,y):
    #函数作用
    """
    计算两个输入值的乘积，并且进行了柯里化，支持向量乘积
    x,y: 要求积的两个值
    """
    x = np.array(x)
    y = np.array(y)
    return x*y
@tz.curry
def sub(x,y):
    #函数作用
    """
    计算两个输入的差，并且进行了柯里化，支持向量的差计算
    x,r: 要求差的两个值
    """
    x = np.array(x)
    y = np.array(y)
    return x-y
def pow1(x):
    #函数作用
    """
    计算一个值的平方，支持向量运算
    
    """
    x = np.array(x)
    return x**2
def dotproduct(vec1,vec2):
    #函数作用
    """
    此函数调用了mul1函数来计算两个值（也可以是向量）的乘积，返回一个迭代器
    vec1,vec2: 要进行乘积的两个迭代器
    """
    return map(mul1,vec1,vec2)

def time_is_life(name):
    #函数作用
    """
    此函数是通过传入的函数名称来计算植被指数与生理指数的皮尔逊相关性大小，返回一个1*1000000数组
    name: 要进行计算的植被指数
    需要注意的是，光谱数据和生理指数是在此函数内部进行读取的，因此需要时常去修改此函数来计算不同的生理指数相关性
    """
    data_band = np.load('D:/光谱数据/data_band_2.npy')
    lai = np.load('D:/光谱数据/lai.npy')
    lai_mean = np.mean(lai)
    lai = lai - lai_mean
    VarY_sum = np.sqrt(np.sum(lai**2))
    numerbs = len(lai)

    data_band_iter = iter(data_band)
    data_flow = tz.pipe(data_band_iter,c.map(name))
    Refl_aver = tz.last(tz.accumulate(sum1,data_flow))/numerbs

    data_band_iter = iter(data_band)
    lai_iter = iter(lai)
    cov = tz.pipe(data_band_iter,c.map(name),c.map(sub(Refl_aver)))
    covv = dotproduct(cov,lai_iter)
    cov_sum = tz.last(tz.accumulate(sum1,covv))

    data_band_iter = iter(data_band)
    VarX = tz.pipe(data_band_iter,c.map(name),c.map(sub(Refl_aver)),c.map(pow1))
    VarX_sum = np.sqrt(tz.last(tz.accumulate(sum1,VarX)))

    summary_R = cov_sum/(VarX_sum*VarY_sum)
    return summary_R