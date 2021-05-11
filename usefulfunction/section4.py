import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from function_eemd import *
from operator import itemgetter
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def correlate(data_band,lai,path=None):
    #函数作用
    """
    此函数是用来计算原始高光谱数据与生理指数的相关性大小及显著性检验，并绘图
    data_band: 原始高光谱数据，为二维数组，每一行为一条高光谱数据
    lai: 生理指数数据
    path: 生成的图像如果要保存的话将保存路径传入给此参数
    """
    data = []
    p = []
    for i in range(data_band.shape[1]):
        pear = pearsonr(data_band[:,i],lai)
        data.append(pear[0])
        p.append(pear[1])
    data = np.array(data)
    p = np.array(p)
    Wave = np.arange(350,1350)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(5,4))
    ax0 = fig.add_subplot(111)
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax0.spines['left'].set_color('black')
    ax0.spines['bottom'].set_color('black')
    ax0.plot(Wave,data,':',color='black',label='相关系数')
    p_005 = np.where(p>0.05)[0]
    p_005y = data[p_005]
    p_005x = p_005 + 350
    ax0.scatter(p_005x,p_005y,s=10,marker='_',c='black',label='P值>0.05')
    labels = ax0.get_yticklabels()
    [label.set_color('black') for label in labels]
    [label.set_size(14) for label in labels]
    ax0.set_xticks([350,600,850,1100,1350])
    ax0.set_xticklabels([350,600,850,1100,1350],fontsize=14,color='black')
    ax0.set_ylabel('相关系数 Correlation coefficient',fontsize=14,color='black')
    ax0.set_xlabel('波长 Wavelength(nm)',fontsize=14,color='black')
    ax0.legend(edgecolor='w',fontsize=13,loc=2)
    plt.tight_layout()
    if not path is None:
        plt.savefig(path)
    
    plt.show()
    return data

def find_index(symlists,refl):
    #函数说明
    """
    此函数的作用是寻找一个数学表达式中的参数（此参数形式是固定的具体见corr_customer函数说明），后根据对应的参数找出在光谱数据中对应的值，简单的例子：
    如数学表达式'R810-Rblue/R460'，表示在810nm的反射率减去蓝光平均反射率后除以460nm的反射率，通过此函数就会通过R810,Rblue,R460的参数名称去光谱数据中refl去索引这些参数对应的反射率，并通过一个元组列表返回以便corr_customer函数使用，值得注意的是此函数是一个过渡函数，不需要手动调用，这里只是提一嘴。
    symlists: 数学表达式中的参数列表，如上表达式中这个参数的值为['R810','Rblue','R460']
    refl: 一个高光谱数据
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

def corr_customer(strfunctions,data_band,lai,names=None):
    #函数作用
    """
    原谅我贫瘠的知识只能写循环套用，性能应该不太行。
    函数作用，输入多个计算特定计算植被指数的数学公式（其参数使用特定的格式，格式见后文），多组高光谱数据（一行为一条），生理指数和植被指数的名称（非必须），即可返回这些植被指数与对应生理指数相关性大小和显著性检验结果。
    参数格式：
    使用“R...”格式，...内容有五种情况:
    1、最常见的，直接是某个波长对应的反射率其参数为R+波长，如R810、R755等；
    2、天依蓝波段，此时参数名称是固定的即Rblue，其返回蓝光范围（435-450nm）反射率平均值；
    3、宝强绿波段，名称也为固定Rgreen(492-577nm)；
    4、慈父红波段，名称固定Rred(622-760nm)；
    5、近红外短波，名称固定Rnir(780-1100nm)
    使用以上格式的参数定义一系列计算植被指数的数学公式从而组成一个列表
    strfunctions: 数学公式列表，注意一定是列表，就是求一个植被指数也需要使用方括号括起来，如['(R810-R510)/Rblue','(Rnir-Rgreen)/(1+R750)']
    data_band: 高光谱数据组成的数组，一行为一条高光谱数据
    lai: 要求相关性的生理指数
    names: 植被指数对应的名称，非必须，注意如果传入也必须是列表，如果传入名称结果最后会返回一个字典
    """
    data = []
    for i in range(data_band.shape[0]):
        single_data = []
        for strfunct in strfunctions:
            symfunct = sympify(strfunct)
            indexs = find_index(symfunct.atoms(Symbol),data_band[i])
            single_data.append(symfunct.subs(indexs).evalf())
        data.append(single_data)
    data = np.array(data,dtype=float)#前面算出的数据类型不是公认的浮点型
    data = data.reshape(data_band.shape[0],-1)
    corr_data = []
    
    for i in range(data.shape[1]):
        pear = pearsonr(data[:,i],lai)
        corr_data.append(pear[0])
    if not names is None:
        corrr_data = {}
        for i in range(len(names)):
            corrr_data[names[i]] = corr_data[i]
    else:
        corrr_data = corr_data  
    return data,corrr_data
        
def corr_emd(data_band,lai):
    #函数作用
    """
    使用function_eemd.py中定义的函数对输入的高光谱数据集进行eemd降噪，并计算其反射峰（4个），吸收谷（3个）的三个参数（21个），并计算输入的这21个参数与生理指数的相关性，返回三个参数分别为，降噪后的数据（array格式，样本*1000）、计算的参数（array格式，样本*21）、及相关性（list，21）
    注意这个函数的运算巨慢，慎重调用
    data_band: 高光谱数据集
    lai: 生理指数
    """
    data_band_emd = np.zeros(data_band.shape)
    pv_data_array = np.zeros((data_band.shape[0],21))
    for i in range(data_band.shape[0]):
        pv_single,data_band_emd[i,:] = eemd(data_band[i])
        pv_list = list(map(itemgetter(1),pv_single.items()))
        pv_list_segment = []
        for j in pv_list:
            pv_list_segment.extend(j[0:3])
        pv_data_array[i,:] = pv_list_segment
    pv_corr = []
    for i in range(pv_data_array.shape[1]):
        pear = pearsonr(pv_data_array[:,i],lai)
        pv_corr.append(pear[0])
    return data_band_emd,pv_data_array,pv_corr


def plot_doubel_band(result,path=None):
    #函数说明
    """
    此函数用来绘制全波的差值，归一化植被指数的图像，主要是联系之前function_doubel_band.py文件中的time_is_life函数（使用多进程调用此函数计算后的结果，具体见section4.ipynb文件）进行
    result: 使用多进程运算出的结果，为列表，三个元素（差值、比值和归一化），每一个元素为一个array数组大小为1*1000000
    path: 非必须，如果你要保存这张漂亮的图片就需要将保存路径传给此参数
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    a = result[0].reshape(1000,1000)
    c = result[2].reshape(1000,1000)
    m = a.copy()
    n = c.copy()
    m = np.where(m>0.8,1,0)
    n = np.where(n>0.8,1,0)
    
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
    plt.title('a,差值植被指数与叶面积相关性',fontsize=20,color='black')

    ax2 = fig.add_subplot(222)
    im = ax2.imshow(c,origin='lower',cmap=plt.cm.jet,interpolation='nearest')
    fig.colorbar(im,ax=ax2,shrink=1)
    ax2.set_yticks([0,200,400,600,800,1000])
    ax2.set_yticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax2.set_xticks([0,200,400,600,800,1000])
    ax2.set_xticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax2.set_xlabel('波长 Wavelength(nm)',fontsize=14,color='black')
    ax2.set_ylabel('波长 Wavelength(nm)',fontsize=14,color='black')
    plt.title('b,归一化植被指数与叶面积相关性',fontsize=20,color='black')
    
    ax3 = fig.add_subplot(223)
    im = ax3.imshow(m,origin='lower',cmap='gray',interpolation='nearest')
    fig.colorbar(im,ax=ax3,shrink=1)
    ax3.set_yticks([0,200,400,600,800,1000])
    ax3.set_yticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax3.set_xticks([0,200,400,600,800,1000])
    ax3.set_xticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax3.set_xlabel('波长 Wavelength(nm)',fontsize=14,color='black')
    ax3.set_ylabel('波长 Wavelength(nm)',fontsize=14,color='black')
    plt.title('c,差值植被指数相关性>0.8',fontsize=20,color='black')
    
    ax4 = fig.add_subplot(224)
    im = ax4.imshow(n,origin='lower',cmap='gray',interpolation='nearest')
    fig.colorbar(im,ax=ax4,shrink=1)
    ax4.set_yticks([0,200,400,600,800,1000])
    ax4.set_yticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax4.set_xticks([0,200,400,600,800,1000])
    ax4.set_xticklabels([350,550,750,950,1150,1350],fontsize=14,color='black')
    ax4.set_xlabel('波长 Wavelength(nm)',fontsize=14,color='black')
    ax4.set_ylabel('波长 Wavelength(nm)',fontsize=14,color='black')
    plt.title('d,归一化植被指数相关性>0.8',fontsize=20,color='black')
    plt.tight_layout()
    if not path is None:
        plt.savefig(path)
    plt.show()
    
def divmod1(data):
    #函数作用
    """
    作用为计算一组数据除以1000的整除数和余数，由于计算出的差值，归一化的植被指数的形状是1*1000000的，为了将其对应到二维数据上可以让一维坐标除以1000，整除数为二维数组的行坐标，余数为二维数组的列坐标
    data: 要进行计算的数据
    """
    xs = []
    ys = []
    for i in data:
        x,y = divmod(i,1000)
        xs.append(x)
        ys.append(y)
    return np.array(xs,dtype=int),np.array(ys,dtype=int)

def susceptible_doubel_band(result,data_band,index_type=0):
    #函数作用
    """
    作用为，提取出全波段组合结果相关性最强的前50个植被指数，并返回根据原始数据计算出的这50个植被指数及对应的组合方式（即组合波段x，y，需要注意的是差值植被指数中返回的x，y是相反的，即组合方式是y-x，可能是前期编写差值植被指数计算时出现的失误，就这样吧，不想改了）
    result: 使用多进程运算出的结果，为列表，三个元素（差值、比值和归一化），每一个元素为一个array数组大小为1*1000000
    data_band: 高光谱数据集
    index_type: 提取什么类型的植被指数，可选择两类差值、归一化，当要提取差值时将0传入给此参数，反之将2传递给此参数。（之前计算过比值，但是效果与归一化的差别不是很大，因此也就没有去提取，就是这样）
    """
    data = result[int(index_type)].copy()
    data[np.isnan(data)] = 0
    data50 = np.argsort(data)[-50:]
    x,y = divmod1(data50)
    if index_type == 0:
        susceptible_index_doubel_band =  data_band[:,y] - data_band[:,x]
    if index_type == 2:
        susceptible_index_doubel_band =  (data_band[:,x] - data_band[:,y])/(data_band[:,y] + data_band[:,x])
    return susceptible_index_doubel_band,x,y


def plot_learning_curves(models,X,y,path=None):
    #函数说明
    """
    此函数局限性很高（专门为我的论文服务的），专门来绘制我所选择的三个模型的关于RMSE和MAE误差的学习曲线，选择的三个模型分别为岭回归、决策树和支持向量机，最后会出现6幅图来描述这三个模型的性能
    models: 建立完成的模型对象，注意这个为列表，必须是三个，否则会出错
    X: 特征数组
    y: 标签数组
    path: 如果你想保存这张漂亮的图片的话需要将保存路径传入到此参数
    """
    import warnings
    warnings.filterwarnings('ignore')
    names = ['a,岭回归学习曲线(RMSE)','b,决策树回归学习曲线(RMSE)','c,支持向量机回归学习曲线(RMSE)']
    names1 = ['d,岭回归学习曲线(MAE)','e,决策树回归学习曲线(MAE)','f,支持向量机回归学习曲线(MAE)']
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(15,8))
    for model,i,name,name1 in zip(models,[1,2,3],names,names1):
        train_errors,val_errors = [],[]
        train_errors_MAE,val_errors_MAE = [],[]
        for m in range(1,len(X_train)):
            model.fit(X_train[:m],y_train[:m])
            y_train_predict = model.predict(X_train[:m])
            y_val_predict = model.predict(X_val)
            train_errors.append(mean_squared_error(y_train_predict,y_train[:m]))
            val_errors.append(mean_squared_error(y_val_predict,y_val))
            train_errors_MAE.append(mean_absolute_error(y_train_predict,y_train[:m]))
            val_errors_MAE.append(mean_absolute_error(y_val_predict,y_val))
            
            
        ax = fig.add_subplot(2,3,i)
        ax.plot(np.sqrt(train_errors),color='black',linestyle='-',label='训练集')
        ax.plot(np.sqrt(val_errors),color='black',linestyle=':',label='验证集')
        print(name +'最终的RMSE误差为%.3f'%((np.sqrt(train_errors[-1])+np.sqrt(val_errors[-1]))/2.0))
        ax.legend(edgecolor='w',fontsize=13)
        ax.set_yticks([0,0.1,0.2,0.3,0.4,])
        ax.set_yticklabels([0,0.1,0.2,0.3,0.4],fontsize=14,color='black')
        plt.ylim(0,0.4)
        if i==1:
            ax.set_ylabel('RMSE',fontsize=14,color='black')
        #ax.set_xlabel('训练数量',fontsize=14,color='black')
        plt.title(name,fontsize=16,color='black')
        
        ax = fig.add_subplot(2,3,i+3)
        ax.plot(train_errors_MAE,color='black',linestyle='-',label='训练集')
        ax.plot(val_errors_MAE,color='black',linestyle=':',label='验证集')
        print(name +'最终的MAE误差为%.3f'%((train_errors_MAE[-1]+val_errors_MAE[-1])/2.0))
        ax.legend(edgecolor='w',fontsize=13)
        ax.set_yticks([0,0.1,0.2,0.3,0.4,])
        ax.set_yticklabels([0,0.1,0.2,0.3,0.4],fontsize=14,color='black')
        plt.ylim(0,0.4)
        if i==1:
            ax.set_ylabel('MAE',fontsize=14,color='black')
        ax.set_xlabel('训练数量',fontsize=14,color='black')
        plt.title(name1,fontsize=16,color='black')     
    plt.tight_layout()
    if not path is None:
        plt.savefig(path)
    plt.show()
    
def plot_true_simulate(models,X,y,path=None):
    #函数说明
    """
    此函数的局限性也很高（专门为我论文的出图设立），专门来绘制我选择的三个模型真实值、预测值的散点图和R方,选择的三个模型分别为岭回归、决策树和支持向量机
    models：三个建立完成的模型，必须为三个，否则会出错
    X: 特征数组
    y: 标签数组
    path: 如果你想保存这张漂亮的图片的话需要将保存路径传入到此参数
    """
    x_1 = np.arange(4.5,6.5,0.1)
    y_1 = np.arange(4.5,6.5,0.1)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(15,4))
    names = ['a,岭回归','b,决策树回归','c,支持向量机回归']
    for i,model,name in zip([1,2,3],models,names):
        model.fit(X,y)
        predict_y = model.predict(X)
        R = model.score(X,y)
        ax = fig.add_subplot(1,3,i)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.scatter(y,predict_y,s=15,c='black',)
        ax.plot(x_1,y_1,color='black',linestyle='-',label='1:1线')
        ax.text(4.75,6.0,r'$R^2$ = %.3f'%R,fontsize=13,color='black')

        ax.set_xticks([4.5,5.0,5.5,6.0,6.5])
        ax.set_xticklabels([4.5,5.0,5.5,6.0,6.5],fontsize=14,color='black')
        ax.set_yticks([4.5,5.0,5.5,6.0,6.5])
        ax.set_yticklabels([4.5,5.0,5.5,6.0,6.5],fontsize=14,color='black')
        ax.set_xlabel('实测值',fontsize=14,color='black')
        ax.set_ylabel('模拟值',fontsize=14,color='black')
        ax.legend(edgecolor='w',fontsize=13,loc=2,framealpha=0)
        plt.title(name,fontsize=16,color='black')
    plt.tight_layout()
    if not path is None:
        plt.savefig(path)
    plt.show()  
    
def transform_name(susceptible_index_name):
    #函数说明
    """
    此函数是用来转换之前计算出敏感参数名称的，由于植被指数是分四部分计算的，因此最后提取出的名称也分了四部分，且要么是序号要么是不规范的名称，这个函数就是将之前不规范的名称进行规范化，返回一个简单的列表，依次为各个敏感植被指数的名称，同时最后两个名称为遮光率（0-1）和散射辐射比例（0-1）最后的格式是：
    10*原始波段、6*波峰波谷参数、10*自定义植被指数、50*差值植被指数、50*归一化植被指数、遮光率、散射辐射比例
    这里有必要提一嘴波峰、波谷的命名过程，我们在之前选择了拢共4个波峰3个波谷，每个波峰或者波谷有三个参数：高度/深度、面积、归一化指数，一个有21个参数，这21参数命名方式为（第一组为例），Pd1,Pa1,NP1这便分别代表了第一个波峰的高度、面积及归一化参数，而Vd1,Va1,NV1代表了第一个波谷的深度、面积及归一化参数，依次类推。注意由于我们默认是从绿峰开始，所以我们的第一个为波峰，及顺序为：波峰、波谷、波峰、波谷、波峰、波谷、波峰，望自知！
    susceptible_index_name: 由前面计算所得出的各敏感参数名称
    """
    pv_names = ['Pd1','Pa1','NP1','Vd1','Va1','NV1','Pd2','Pa2','NP2','Vd2','Va2','NV2',
           'Pd3','Pa3','NP3','Vd3','Va3','NV3','Pd4','Pa4','NP4']
    pv_names = np.array(pv_names)
    names_sum = []
    for i in (susceptible_index_name[0]):
        name = 'R'+str(i+350)
        names_sum.append(name)
    for i in pv_names[susceptible_index_name[1]]:
        names_sum.append(i)
    for i in susceptible_index_name[2]:
        names_sum.append(i)
    for i in susceptible_index_name[3]:
        name = 'R'+str(i[0]+350) +'-'+'R'+str(i[1]+350)
        names_sum.append(name)
    for i in susceptible_index_name[4]:
        name = '(R'+str(i[0]+350) +'-'+'R'+str(i[1]+350) + ')/(' + 'R'+str(i[0]+350) +'+'+'R'+str(i[1]+350)+')'
        names_sum.append(name)
    names_sum.append('RGR')
    names_sum.append('Fdiff')
    return names_sum