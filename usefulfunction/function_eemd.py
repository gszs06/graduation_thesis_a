#1、该文件定义了使用EEMD方法来对高光谱曲线进行降噪的函数、使用降噪后的曲线计算反射峰和吸收谷的特征参数的函数、绘制降噪后结果
#2、使用方法是，有两种实现方法，一种是使用emd，该函数只会返回计算的参数（字典）和降噪后的结果，适合套用循环大量数据运算，值
#   注意的是，使用eemd的包默认是使用多进程的，因此在这里可以直接使用循环而不再去考虑多进程；另一种方法区别不是很大，但会返回
#   分解结果，为了节约内存请务必不要对大数据使用此方法
#3、实现原理，见"使用eemd降噪过程.pdf"文件

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import simps
import PyEMD

def valley(Rs,Rc,Re,lamda_s,lamda_c,lamda_e,Refl_emd,Wave):
    #函数说明
    """
    此函数用于计算一个吸收谷的特征参量，计算过程见"使用eemd降噪过程.pdf"文件
    Rs,Rc,Re: 吸收谷起始、中点和终点的反射率
    lamda_s,lamda_c,lamda_e: 吸收谷起始、中点和终点的位置
    Refl_emd: 降噪后的曲线
    Wave: 反射率对应的波长
    """
    Vd = 1 - Rc/(Rs+(Re-Rs)*(lamda_c-lamda_s)/(lamda_e-lamda_s))
    x = Wave[int(lamda_s-350):int(lamda_e-350)]
    y_R = Refl_emd[int(lamda_s-350):int(lamda_e-350)]
    y_l = list(map(lambda z:(Re-Rs)*(z-lamda_s)/(lamda_e-lamda_s) + Rs ,x))
    area1 = simps(y_R,x)
    area2 = simps(y_l,x)
    V_area = area2-area1
    NVD = Vd/V_area
    Vd = np.around(Vd,decimals=3)
    V_area = np.around(V_area,decimals=3)
    NVD = np.around(NVD,decimals=3)
    Rc = np.around(Rc,decimals=3)
    return Vd,V_area,NVD,lamda_c,Rc

def peak(Rs,Rc,Re,lamda_s,lamda_c,lamda_e,Refl_emd,Wave):
    #函数说明
    """
    此函数用于计算一个反射峰的特征参量，计算过程见"使用eemd降噪过程.pdf"文件
    Rs,Rc,Re: 反射峰起始、中点和终点的反射率
    lamda_s,lamda_c,lamda_e: 反射峰起始、中点和终点的位置
    Refl_emd: 降噪后的曲线
    Wave: 反射率对应的波长
    """
    Ph = 1 - (Rs+(Re-Rs)*(lamda_c-lamda_s)/(lamda_e-lamda_s))/Rc
    x = Wave[int(lamda_s-350):int(lamda_e-350)]
    y_R = Refl_emd[int(lamda_s-350):int(lamda_e-350)]
    y_l = list(map(lambda z:(Re-Rs)*(z-lamda_s)/(lamda_e-lamda_s) + Rs ,x))
    area1 = simps(y_R,x)
    area2 = simps(y_l,x)
    P_area = area1-area2
    NPH = Ph/P_area
    Ph = np.around(Ph,decimals=3)
    P_area = np.around(P_area,decimals=3)
    NPH = np.around(NPH,decimals=3)
    Rc = np.around(Rc,decimals=3)
    return Ph,P_area,NPH,lamda_c,Rc

def find_median(refle,a,b,c):
    #函数说明
    """
    此函数用于寻找曲线中一段范围内的最优极大值点，如果没有则返回这个范围的中点，寻找原则：找峰宽度大于20，如果有多个则返回最中间的峰
    refle: 降噪后的曲线
    a,b: 需要寻找极值点的范围
    c: 为范围的中点
    """
    poles = signal.find_peaks(refle[int(a):int(b)],width=20)[0]
    if len(poles) == 0:
        pole = c
    else:
        pole = np.median(poles)+a
    return pole

def eemd(Refl):
    #函数作用
    """
    这个函数是使用eemd方法来对光谱数据进行降噪并计算反射吸收峰谷及降噪后的曲线，这个函数支持多线程应该，为了节约内存不会返回所有的分解结果，
    如果要返回所有分解需要使用eemd1函数
    """
    Wave = np.arange(350,1350)
    E_IMFs = PyEMD.EEMD(trials=50)
    E_IMFs.noise_seed(0)
    E_IMFs = E_IMFs.eemd(Refl,Wave)
    imfNo = E_IMFs.shape[0]
    index = np.zeros(imfNo)
    for i in range(imfNo):
        index[i] = np.var(E_IMFs[i])
    index = np.where(index>10e-5)
    Refl_emd = np.sum(E_IMFs[index],axis=0)
    
        
    R_min = np.min(Refl_emd)
    if R_min <= 0:
        Refl_emd = Refl_emd - R_min

    start_index = [100,200,400,500,600,700,800]
    end_index = [300,400,600,700,800,900,1000]
    miss_values = [200,300,500,600,700,800,900]
    extremes = []
    sigma = 1
    for start,end,miss_value in zip(start_index,end_index,miss_values):
        extremes.append(find_median(sigma*Refl_emd,start,end,miss_value))
        sigma = sigma*-1
    extremes = np.array(extremes,dtype=np.int)
    
    f_mark = False
    sumdata = {}
    for i in range(len(extremes)):   
        Rc = Refl_emd[extremes[i]]
        lamda_c = extremes[i]+350
        if i == 0 :
            Rs = Refl_emd[0]
            Re = Refl_emd[extremes[i+1]]
            lamda_s = Wave[0]
            lamda_e = extremes[i+1]+350
        elif i == len(extremes)-1:
            Rs = Refl_emd[extremes[i-1]]
            Re = Refl_emd[-1]
            lamda_s = extremes[i-1]+350
            lamda_e = Wave[-1]
        else:
            Rs = Refl_emd[extremes[i-1]]
            Re = Refl_emd[extremes[i+1]]
            lamda_s = extremes[i-1]+350
            lamda_e = extremes[i+1]+350
        if f_mark:
            vall = valley(Rs,Rc,Re,lamda_s,lamda_c,lamda_e,Refl_emd,Wave)
            strv = str(i)
            sumdata[strv] = vall
        else:
            pea = peak(Rs,Rc,Re,lamda_s,lamda_c,lamda_e,Refl_emd,Wave)
            strp = str(i)
            sumdata[strp] = pea
        f_mark = not f_mark
    return sumdata,Refl_emd

def eemd1(Refl):
    #函数说明
    """
    这个函数是为了单次调用计算使用的，会返回所有的分解结果、反射吸收峰谷及降噪后的曲线
    """
    Wave = np.arange(350,1350)
    E_IMFs = PyEMD.EEMD(trials=50)
    E_IMFs.noise_seed(0)
    E_IMFs = E_IMFs.eemd(Refl,Wave)
    imfNo = E_IMFs.shape[0]
    index = np.zeros(imfNo)
    for i in range(imfNo):
        index[i] = np.var(E_IMFs[i])
    index = np.where(index>10e-5)
    Refl_emd = np.sum(E_IMFs[index],axis=0)
    
        
    R_min = np.min(Refl_emd)
    if R_min <= 0:
        Refl_emd = Refl_emd - R_min

    start_index = [100,200,400,500,600,700,800]
    end_index = [300,400,600,700,800,900,1000]
    miss_values = [200,300,500,600,700,800,900]
    extremes = []
    sigma = 1
    for start,end,miss_value in zip(start_index,end_index,miss_values):
        extremes.append(find_median(sigma*Refl_emd,start,end,miss_value))
        sigma = sigma*-1
    extremes = np.array(extremes,dtype=np.int)
    
    f_mark = False
    sumdata = {}
    for i in range(len(extremes)):   
        Rc = Refl_emd[extremes[i]]
        lamda_c = extremes[i]+350
        if i == 0 :
            Rs = Refl_emd[0]
            Re = Refl_emd[extremes[i+1]]
            lamda_s = Wave[0]
            lamda_e = extremes[i+1]+350
        elif i == len(extremes)-1:
            Rs = Refl_emd[extremes[i-1]]
            Re = Refl_emd[-1]
            lamda_s = extremes[i-1]+350
            lamda_e = Wave[-1]
        else:
            Rs = Refl_emd[extremes[i-1]]
            Re = Refl_emd[extremes[i+1]]
            lamda_s = extremes[i-1]+350
            lamda_e = extremes[i+1]+350
        if f_mark:
            vall = valley(Rs,Rc,Re,lamda_s,lamda_c,lamda_e,Refl_emd,Wave)
            strv = str(i)
            sumdata[strv] = vall
        else:
            pea = peak(Rs,Rc,Re,lamda_s,lamda_c,lamda_e,Refl_emd,Wave)
            strp = str(i)
            sumdata[strp] = pea
        f_mark = not f_mark
    return E_IMFs,sumdata,Refl_emd

def plot_eemd(Wave,Refl,IMFS,path=None):
    #函数说明
    """
    此函数用来绘制使用eemd降噪后的结果
    Wave: 反射率对应的波长
    Refl: 原始高光谱反射数据
    IMFS: 使用eemd分解后的结果
    path: 如果需要保存绘图结果需要将保存路径传入到此参数
    """
    imfNo = IMFS.shape[0]
    m = np.floor(np.sqrt(imfNo+1))
    n = np.ceil((imfNo+1)/m)
    fig = plt.figure(figsize=(20,10))
    fig.add_subplot(n,m,1)
    plt.plot(Wave,Refl,'black')
    plt.title("Original signal")
    for num in range(imfNo):
        plt.subplot(n, m, num + 2)
        plt.plot(Wave, IMFS[num], 'black')
        plt.title("Imf " + str(num + 1))
    plt.tight_layout()
    if not path is None:
        plt.savefig(path)
    plt.show()
    