import numpy as np
import PyEMD
import os 


if __name__ == '__main__':
    data_band = np.load(os.path.abspath("data/data_band_2.npy"))
    Wave = np.arange(350,1350)
    number = input()
    number = int(number)
    E_IMFs = PyEMD.EEMD(trials=50,max_imf=8,parallel=False)
    imfs = E_IMFs.eemd(data_band[number,:],Wave)
    np.save(os.path.abspath("data/{:s}_eemd.npy".format(str(number))),imfs)
    print("分解完成")