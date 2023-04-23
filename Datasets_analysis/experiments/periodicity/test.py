import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm
import sys
sys.path.append("/Users/quentinlao/Documents/GitHub/Data-augmentation-time-series/Datasets_analysis/res")
import entropy as ent


def calc(data) -> dict :
    return ent.spectral_entropy(data, sf = 200, method='fft', normalize=True)



if __name__ == "__main__" :

    X = np.linspace(0,20, 100)
    score = []
    nb_scales = 100

    for scale in tqdm( range(nb_scales) ):
        def f(x) : 
            return 1 + np.random.normal(scale=0.0001 + scale*0.01, size=len(x))

        np.vectorize(f)

        Y = f(X)
        Y = (Y - np.min(Y))/(np.max(Y) - np.min(Y))


        if scale%5 == 0 :
            plt.plot(X,Y)
            plt.savefig("{}.png".format(round( scale*0.01, 3)))
            plt.close()
        


        score.append( calc(Y)  )

    plt.plot(np.linspace(0,nb_scales-1,nb_scales),score)
    plt.savefig("periodicity_evolution.png")


