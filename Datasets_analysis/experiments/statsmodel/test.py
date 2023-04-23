import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
from statsmodels.tsa.stattools import acf
import seaborn as sns
from scipy.signal import periodogram





if __name__ == "__main__" :

    nb = 20
    coef_mult = 10
    period = 3

    X = np.linspace(0,nb-1, coef_mult*nb)

    def f(x) : 
        return np.sin((2*np.pi/period)*x) + np.cos((np.pi/period)*x) #+ np.random.normal(scale=0.5, size=len(x))
    np.vectorize(f)

    Y = f(X)
    autocorrelation = acf(Y, nlags = coef_mult*nb)

    data = pd.DataFrame(columns = ["num", "value"])
    for i in range(len(autocorrelation)) :
        if i%coef_mult == 0 :
            data.loc[i//coef_mult] = [i//coef_mult, autocorrelation[i]]


    plt.figure(figsize=(10,6))

    sns.barplot(x="num", y="value", data=data)
    plt.plot(X,Y)

    plt.savefig("acf_test.png", dpi = 300)
    plt.close()

    ##
    plt.figure(figsize=(10,6))
    x,y = periodogram(Y)
    plt.plot(x,y)
    plt.savefig("periodogram_test.png", dpi = 300)





