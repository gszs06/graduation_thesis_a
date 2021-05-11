import numpy as np
import matplotlib.pyplot as plt
import PyEMD
from scipy.signal import find_peaks
from scipy.integrate import simps

import traceback
import sys
import time
import os 

np.seterr(divide='ignore',invalid='ignore')
def plot_a(a,peak_points,i):
    fig = plt.figure() 
    ax0 = fig.add_subplot(111)
    ax0.plot(a)
    x = peak_points
    y = a[x]
    ax0.scatter(x,y,c='r')
    path = "D:/图片/{:s}.png".format(str(i))
    plt.savefig(path)
    plt.close(fig)

def line_inter(relf,a,b):
    x = np.arange(a,b)
    x1 = a
    y1 = relf[a]
    x2 = b
    y2 = relf[b]
    y = (x - x2) * (y1 - y2) / (x1 - x2) + y2
    return y

def nor_area_absor(a,peak_points,valley_points):
    peak_points = peak_points.reshape(1,-1)
    valley_points = valley_points.reshape(1,-1)
    a = a - min(a)
    envelope = a.copy()
    for k in range(peak_points.shape[1] - 1):
        envelope[peak_points[0][k]:peak_points[0][k+1]] = line_inter(a,peak_points[0][k],peak_points[0][k+1])        
    normalize = a / envelope
    polars = np.hstack((peak_points,valley_points))
    polars = np.sort(polars)
    areas = []
    for k in range(polars.shape[1] - 1):
        x = np.arange(polars[0][k]+350,polars[0][k+1]+350)
        y = normalize[polars[0][k]:polars[0][k+1]]
        areas.append(polars[0][k+1] - polars[0][k] - simps(y,x))
    deep = []
    for k in valley_points[0]:
        deep.append(a[k])
    return areas,deep

if __name__ == '__main__':
    data_band = np.load(os.path.abspath("data/data_band_2.npy"))
    Wave = np.arange(350,1350)


    file = os.path.abspath("data/data.txt")
    err_index = []
    err_message = []
    data = []
    data_sum = []
    E_IMFs = PyEMD.EEMD(trials=50,max_imf=8)
    #E_IMFs.noise_seed(0)
    start = time.time()
    len_data = len(data_band)
    for i in range(len_data):
        try:
            imfs = E_IMFs.eemd(data_band[i,:],Wave)
            s = imfs[4] + imfs[5] + imfs[6] + imfs[7]
            peak_points_1 = find_peaks(s[:400])[0]
            peak_point_200 = peak_points_1[np.argmax(s[peak_points_1])]
            peak_points_2 = find_peaks(s[400:600])[0] + 400
            
            if len(peak_points_2) > 2:
                peak_point_400 = peak_points_2[0]
                if i == 102:
                    peak_point_500 = peak_points_2[-3]
                elif i == 186:
                    peak_point_500 = peak_points_2[-1]
                elif i == 226:
                    peak_point_400 = 442
                    peak_point_500 = peak_points_2[-3]
                else:
                    peak_point_500 = peak_points_2[-2]
            elif len(peak_points_2) == 2:
                if np.abs(peak_points_2[0] - peak_points_2[1]) < 40:
                    peak_point_400 = 442
                    peak_point_500 = peak_points_2[0]
                else:
                    peak_point_400 = peak_points_2[0]
                    peak_point_500 = peak_points_2[1]
            else:
                peak_point_400 = 442
                peak_point_500 = peak_points_2[0]

            peak_points_3 = find_peaks(s[600:])[0] + 600

            if i == 150 or i == 326:
                peak_point_700 = peak_points_3[1]
            else:
                peak_point_700 = peak_points_3[0]
            peak_point_900 = peak_points_3[-1]
            
            peak_point_200 = np.array(peak_point_200,dtype=np.int)
            peak_point_400 = np.array(peak_point_400,dtype=np.int)
            peak_point_500 = np.array(peak_point_500,dtype=np.int)
            peak_point_700 = np.array(peak_point_700,dtype=np.int)
            peak_point_900 = np.array(peak_point_900,dtype=np.int)
            
            peak_points = np.hstack((peak_point_200,peak_point_400,peak_point_500,peak_point_700,peak_point_900))


            valley_points = []

            valley_points.append(np.argmin(s[peak_points[0]:peak_points[1]]) + peak_points[0])
            valley_points.append(np.argmin(s[peak_points[2]:peak_points[3]]) + peak_points[2])
            valley_points.append(np.argmin(s[peak_points[3]:peak_points[4]]) + peak_points[3])
            
            valley_points = np.array(valley_points,dtype=np.int)



            #plot_a(s,np.hstack((peak_points,valley_points)),i)

            area1,deep1 = nor_area_absor(s,peak_points[0:2],valley_points[0])
            area2,deep2 = nor_area_absor(s,peak_points[2:],valley_points[1:])
            line = '样本为{:^3d}各面积为{:.2f}、{:.2f}、{:.2f}、{:.2f}、{:.2f}、{:.2f}。深度为{:.2f}、{:.2f}、{:.2f}\n'.format(i,area1[0],area1[1],area2[0],area2[1],area2[2],area2[3],deep1[0],deep2[0],deep2[1])
            data_sum.append(np.hstack((area1,area2,deep1,deep2)))
            data.append(line)
            if i % 10 == 0:
                number_star = "*" * (i//10)
                number_point = "." * (len_data//10-i//10)
                progress = (i/len_data) * 100
                dur = time.time() - start
                print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(progress,number_star,number_point,dur),end = "")

        except Exception as e:
            err_index.append(i)
            err_message.append(e)
            #print(traceback.format_exc())
        continue
    with open(file,'w') as f:
        for i in data:
            f.write(i)
    data_sum = np.array(data_sum)
    np.save(os.path.abspath("data/data_sum_Enorm.npy"),data_sum)    
    sys.stdout.write('\n')
    sys.stdout.flush()
    print("出现分解错误的样本为{}".format(err_index))
    print("错误信息为{}".format(err_message))