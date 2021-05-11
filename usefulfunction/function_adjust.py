import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from sklearn.ensemble import IsolationForest
from scipy.interpolate import UnivariateSpline

def select_data(index,strlist,day=None):
    bool_data = []
    order = []
    if day==None:
        for i in range(index.shape[0]):
            if index[i,2] in strlist:
                order.append(int(index[i,1]))
    else:
        for i in range(index.shape[0]):
            if (index[i,2] in strlist) and (index[i,3] == day):
                order.append(int(index[i,1]))
    return np.array(order)

def select_data_1(index,strlist=None,day=None):
    order = []
    if (strlist == None) and (day == None):
        order = index[:,4]
        order = np.array(order,dtype=np.int)
    elif (strlist == None)and(day != None):
        order = np.where(index[:,3] == day)[0]
    elif (strlist != None)and(day == None):
        for i in range(index.shape[0]):
            if index[i,2] in strlist:
                order.append(int(index[i,4]))
        order = np.array(order,dtype=np.int)
    else:
        for i in range(index.shape[0]):
            if (index[i,2] in strlist) and (index[i,3] == day):
                order.append(int(index[i,4]))
        order = np.array(order,dtype=np.int)
    return order
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
    if not strlist is None:
        for i in range(index.shape[0]):
            if index[i,2] in strlist:
                order1.append(int(index[i,1]))
    else:
        order1 = np.array(index[:,1],dtype=np.int)
    if not day is None:
        order2 = np.where(index[:,3]==day)[0]
    else:
        order2 = np.array(index[:,1],dtype=np.int)
    if not year is None:
        order3 = np.where(index[:,4]==year)[0]
    else:
        order3 = np.array(index[:,1],dtype=np.int)
    order = set(order1)&set(order2)&set(order3)
    order = np.array(list(order),dtype=np.int)
    return order
def mean_part(data,order):
    model = IsolationForest(random_state=0).fit(data)
    outliers = model.predict(data)
    mean_part = np.mean(data[outliers == 1],axis=0)
    return mean_part,order[outliers == 1],order[outliers == -1]

#计算任意两个波段位置，最大斜率及面积
#参考 蓝（490-530）黄（550-580）红（680-750）近红1（920-980）近红2（1000-1060）近红3（1100-1180）
def derivative(Wave,Refl,lamda1,lamda2):
    derivates = [0]
    for i in range(len(Refl)-2):
        derivate = (Refl[i+2] - Refl[i])/2.0
        derivates.append(derivate)
    derivates.append(0)
    derivates = np.array(derivates)
    lamda1,lamda2 = int(lamda1),int(lamda2)
    x = Wave[lamda1-350:lamda2-350]
    y = derivates[lamda1-350:lamda2-350]
    y = np.where(y<=0,0,y)
    maxRefl_d = np.max(y)
    maxWave = np.argmax(y)+lamda1
    area = simps(y,x)
    return maxWave,maxRefl_d,area
#使用IG模型来计算红谷光谱位置和吸收谷宽度
def IG_red(Wave,Refl):
    Rs = np.mean(Refl[780-350:795-350])
    R0 = np.mean(Refl[670-350:675-350])
    y = Refl[685-350:780-350]
    x = Wave[685-350:780-350]
    y = list(map(lambda x: np.sqrt(-np.log((Rs-x)/(Rs-R0))),y))
    y = np.array(y)
    a1,a0 = np.polyfit(x,y,1)
    lamda0 = -a0/a1
    sigma = 1.0/np.sqrt(2*a1)
    return lamda0,sigma
class data:
    def __init__(self,mean_ck,mean_p,mean_m):
        self.wave = np.arange(350,1350)
        self.mean_ck = mean_ck
        self.mean_p = mean_p
        self.mean_m = mean_m
    def update(self,mean_ck,mean_p,mean_m):
        self.mean_ck = mean_ck
        self.mean_p = mean_p
        self.mean_m = mean_m
    def red_side_plot(self):
        mean_ck_d,wave_ck,are_ck = derivative(self.wave,self.mean_ck,680,750)
        mean_ck_igd,width_ck = IG_red(self.wave,self.mean_ck)
        mean_p_d,wave_p,are_p = derivative(self.wave,self.mean_p,680,750)
        mean_p_igd,width_p = IG_red(self.wave,self.mean_p)
        mean_m_d,wave_m,are_m = derivative(self.wave,self.mean_m,680,750)
        mean_m_igd,width_m = IG_red(self.wave,self.mean_m)
        print('CK处理的红边位置为：%d，最大斜率为：%f，红边面积为：%f，IG模型中红边位置为：%d，红谷宽度为：%f'%(mean_ck_d,wave_ck,are_ck,mean_ck_igd,width_ck))
        print(' P处理的红边位置为：%d，最大斜率为：%f，红边面积为：%f，IG模型中红边位置为：%d，红谷宽度为：%f'%(mean_p_d,wave_p,are_p,mean_p_igd,width_p))
        print(' M处理的红边位置为：%d，最大斜率为：%f，红边面积为：%f，IG模型中红边位置为：%d，红谷宽度为：%f'%(mean_m_d,wave_m,are_m,mean_m_igd,width_m))
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.plot(self.mean_ck,label='ck',color='blue')
        ax.plot(self.mean_p,label='p',color='green')
        ax.plot(self.mean_m,label='m',color='red')
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_color('w') for label in labels]
        [label.set_size(14) for label in labels]
        ax.legend()
    def plot1(self):
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus']=False
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        ax.plot(self.mean_ck,label='ck',color='blue')
        ax.plot(self.mean_p,label='p',color='green')
        ax.plot(self.mean_m,label='m',color='red')
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_color('w') for label in labels]
        [label.set_size(14) for label in labels]
        ax.legend()
        
def weight_guss(mid,sigma,low,a=0,b=430):
    #函数说明
    """
    作用为使用高斯函数产生一组从a到b的权重
    mid为差值最大的索引数；
    sigma为控制高斯函数范围；
    low为最大缩小（扩大）比例；
    a与b为要应用的范围
    例子：在总m处理时各参数为200,90000,0.6,0,430
    """
    weight = np.arange(a,b)
    weight = 1-np.exp((weight-mid)**2/(-sigma))+low
    weight = np.where(weight>1.0,1,weight)
    return weight
def refl_adjust(refl,weight,a=0,b=430):
    #函数说明
    """
    作用为，将权重矩阵或调整浮点值乘到相应的反射率数组上
    refl: 要进行调整的反射率数组，类型为array
    weight： 要应用的权重，类型如果是一个浮点值就会将这个值应用到整个要调整的数组上
             如果是一个array数组则按照a、b索引对反射率数组部分进行调整
    a,b: 要调整的范围，如果weight为浮点则忽略这两个值，类型为整型
    """
    dim = np.ndim(refl)
    if dim == 1:
        refl = np.expand_dims(refl,0)
    if type(weight) == float:
        refl = refl*weight
    else:
        refl[:,a:b] = refl[:,a:b] * weight
    if dim ==1:
        refl = np.squeeze(refl,0)
    return refl
def adjust_line(refl,a,b):
    #函数作用
    """
    作用为，输入一个波段和一个波段范围，在这个范围内使用线性拟合的值来代替原值
    refl: 为要进行修改的波段值
    a,b: 为在波段上进行线性拟合的范围
    """
    refl_1 = refl.copy()
    dim = np.ndim(refl_1)
    if dim == 1:
        refl_1 = np.expand_dims(refl_1,0)
    if a==b:
        raise Exception('兄弟，请输入一个范围，而不是两个相同的点')
    m = refl_1.shape[0]
    x = np.arange(a,b)
    x = np.broadcast_to(x,(m,b-a))
    k = (refl_1[:,a] - refl_1[:,b]) / (a - b)
    k = k.reshape(m,-1)
    y = k*(x-a) + refl_1[:,a].reshape(m,-1)
    refl_1[:,a:b] = y
    if dim == 1:
        refl_1 = np.squeeze(refl_1,0)
    return refl_1
def adjust_spline(refl):
    #函数说明
    """
    函数作用，这个函数局限性很高，专门为调p28定义，作用是对远红外段（实际索引为800-1000）进行样条插值（5次），选取的样条插值点为800、827、895、926、962、
            999（开始点、极大值点、结束点），对其反射率的缩小权重分别为1.、0.9、0.6、0.66、0.9、1.
    refl: 要进行调整的数据
    """
    refl_1 = refl.copy()
    x = np.arange(800,1000)
    x1 = np.array([800,827,895,926,962,999])
    dim = np.ndim(refl_1)
    if dim == 1:
        
        y1 = np.array([refl_1[800],refl_1[827]*0.9,refl_1[895]*0.6,refl_1[926]*0.66,refl_1[962]*0.9,refl_1[999]])
        y2 = UnivariateSpline(x1,y1,k=4)
        refl_1[800:1000] = y2(x)
    else:
        m = refl_1.shape[0]
        for i in range(m):
            y1 = np.array([refl_1[i][800],refl_1[i][827]*0.9,refl_1[i][895]*0.6,refl_1[i][926]*0.66,refl_1[i][962]*0.9,refl_1[i][999]])
            y2 = UnivariateSpline(x1,y1,k=4)
            refl_1[i][800:1000] = y2(x)
    return refl_1