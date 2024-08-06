import numpy as np
import pandas as pd
import math

R = 8.31446261815324  # J/(mol*K) from the 26th General Conference on Weights and Measures (CGPM)
R_ = 82.057338  # atm*cm3/(mol*K). CHEMKIN uses cgs unit.
cal = 4.184  # CHEMKIN uses the thermal calorie, 1 cal = 4.184 Joules

df = pd.read_csv('dataset.csv')
selected_features = ['0.001','0.01','0.1','1','10','100','HPL']
data_X = df['Tem']
data_Y = df[selected_features]
ln_data_Y = data_Y.apply(np.log)
#df[selected_features] = data_Y.apply(np.log)
#df.to_csv('lnk.csv')

def cal_lnk(lnA, n, E, T):
    return lnA + n * math.log(T) - (E * cal) / (R * T)

def fitting_Arrhenius(temperature, lnk):
    """
    Mathematical principles referenced from CHEMRev (https://doi.org/10.1002/kin.20049)
    :param temperature:
    :param lnk:
    :return:
    """
    X = np.array([[1, math.log(t), 1/t] for t in temperature])
    Y = np.array(lnk)
    beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    lnA, n, E = list(beta)
    E = -E*R/cal  # cal/mol

    avg_err = 0
    for i, temp in enumerate(temperature):
        lnk_fitting = cal_lnk(lnA, n, E, temp)
        lnk_truth = lnk[i]
        avg_err += abs((lnk_truth - lnk_fitting) / lnk_truth)
    avg_err /= len(temperature)
    return lnA, n, E, avg_err
with open('AnE_fit.txt','w') as f:
    print('Pressure     ','A     ','n     ','E      ','fitting error',file=f)
    for feature in selected_features:
        lnA, n, E, avg_err = fitting_Arrhenius(data_X, ln_data_Y[feature])
        print(feature,math.exp(lnA), f"{n:.3f}", f"{E:.3f}", f"{avg_err:.3f}",file=f)

# lnA, n, E, avg_err = fitting_Arrhenius(data_X,ln_data_Y['0.001'])
# print(math.exp(lnA), n, E, avg_err)
# for T in data_X:
#     lnk = cal_lnk(lnA,n,E,T)
#     k = math.exp(lnk)
#     print(k)

