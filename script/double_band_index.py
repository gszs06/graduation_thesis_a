import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from glob import glob
from toolz import curried as c
import toolz as tz
import time
import os

import multiprocessing
np.seterr(divide='ignore',invalid='ignore')

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
    
    NDVI = (1 - RNDVI_flatdata) / (1 + RNDVI_flatdata)
    #NDVI = np.array(list(map(lambda x: (1-x)/(1+x),RNDVI_flatdata)))
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

def time_is_life(name,path_data,path_lai):
    #函数作用
    """
    此函数是通过传入的函数名称来计算植被指数与生理指数的皮尔逊相关性大小，返回一个1*1000000数组
    name: 要进行计算的植被指数
    需要注意的是，光谱数据和生理指数是在此函数内部进行读取的，因此需要时常去修改此函数来计算不同的生理指数相关性
    
    算叶面积时前面两行代码为：
    path_data = 'D:/光谱数据/data_band_2.npy'
    path_lai = 'D:/光谱数据/lai.npy'
    计算14天产量前面两行代码为：
    path_data = 'D:/光谱数据/data_band_14.npy'
    path_lai = 'D:/光谱数据/yields_14.npy'
    计算21天产量前面两行代码为：
    path_data = 'D:/光谱数据/data_band_21.npy'
    path_lai = 'D:/光谱数据/yields_21.npy'
    计算28天产量前面两行代码为：
    path_data = 'D:/光谱数据/data_band_28.npy'
    path_lai = 'D:/光谱数据/yields_28.npy'
    计算总的产量前面两行代码为：
    path_data = 'D:/光谱数据/data_band_2.npy'
    path_lai = D:/光谱数据/yields.npy'
    计算地上干物质前面两行代码为：
    path_data = 'D:/光谱数据/data_band_2.npy'
    path_lai = 'D:/光谱数据/above_ground_biomass.npy'
    """
    #path_data = 'D:/光谱数据/data_band_2.npy'
    #path_lai = 'D:/光谱数据/yields.npy'
    #print(path_data)
    data_band = np.load(path_data)
    lai = np.load(path_lai)
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

if __name__ == '__main__':
    
    start = time.time()

    path_data = input()
    path_lai = input()

    functions = iter([(DVI,path_data,path_lai),(RVI,path_data,path_lai),(NDVI,path_data,path_lai)])

    #functions = iter([DVI,RVI,NDVI])
    with multiprocessing.Pool() as p:
        result = p.starmap_async(time_is_life,functions).get()
    end = time.time()
    if 'lai' in path_lai:
        save_path = os.path.abspath("data/result_lai.npy")
    if 'biomass' in path_lai:
        save_path = os.path.abspath("data/result_biomass.npy")
    if 'output_14' in path_lai:
        save_path = os.path.abspath("data/result_output_14.npy")
    if 'output_21' in path_lai:
        save_path = os.path.abspath("data/result_output_21.npy")
    if 'output_28' in path_lai:
        save_path = os.path.abspath("data/result_output_28.npy")
    np.save(save_path,result)
    print('运行时间为{:.2f}s'.format(end-start))