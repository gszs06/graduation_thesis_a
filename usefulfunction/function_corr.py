import numpy as np
from scipy.stats import pearsonr
from sympy import sympify,Symbol


"""
此文件的作用为：计算选择的四类值植指数的值，并计算植被指数与相关生理指数的决定系数R方
用到的计算包有：
----------numpy
----------scipy
----------sympy

"""

##计算第一类参数
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

##计算第二类参数
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