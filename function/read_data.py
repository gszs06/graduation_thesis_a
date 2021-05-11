#此部分主要定义了读写文件有关函数
import numpy as np
import os

def read_physiological_index(index,fn):
    #读取生理指数函数
    """
    函数作用，传入光谱索引文件index和保存到本地的生理指数txt文件路径，返回一个与光谱索引一样长度的列表为每个光谱文件所对应的生理指数，需要指出的是生理
    指数的txt文件是特定格式的，分为两列，第一列为处理名称，名称组成为年份+距离开花的时间+处理名称（如‘201614ck1’），第二列为对应的生理指数，必须是这种
    格式否则会读写出错
    index: 为光谱索引文件
    fn: 为本地生理指数txt文件路径
    """
    data_txt = {}
    data = []
    with open(fn) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        a = line.split('\t')
        data_txt[a[0]] = float(a[1])
    for i in range(index.shape[0]):
        name = index[i,4]+index[i,3]+index[i,2]
        data.append(data_txt[name])
    return data

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

def pre_statis_test(data,index):
    #函数说明
    """
    函数作用：对数据data依据index进行指定行的分类，分列的序号（1，2，3）插入到
            数据第一列中将，以方便之后的统计检验，T1对应了1，T2对应了2，T3对应
            了3，并将结果保存到data文件夹中。
    input:
        data: 需要分类的数据
        index: 每一个数据对应的索引信息
    out:
        test_data: 计算完成的数据，这个矩阵同样也会保持到data文件夹下，方便移植到R语言中运算 
    """

    factor = np.zeros((data.shape[0],1))
    factor[select_data_2(index,strlist=['ck1','ck2','ck3'])] = 1
    factor[select_data_2(index,strlist=['p1','p2','p3'])] = 2
    factor[select_data_2(index,strlist=['m1','m2','m3'])] = 3
    test_data = np.concatenate((factor,data),axis=1)
    save_path = os.path.join(os.path.abspath("data"),"statis_test_data.txt")
    fmt = ['%.4f']
    fmt = fmt * test_data.shape[1]

    np.savetxt(save_path,test_data,fmt=fmt)
    return test_data
