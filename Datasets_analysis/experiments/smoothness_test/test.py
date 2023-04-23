import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm


def calc_smoothness_each_label(data) -> dict :

    return np.std( np.diff(data) ) 



if __name__ == "__main__" :

    X = np.linspace(0,20, 100)
    score = []
    nb_scales = 100

    for scale in tqdm( range(nb_scales) ):
        def f(x) : 
            return np.sin(x) + np.random.normal(scale=scale*0.01, size=len(x))

        np.vectorize(f)

        Y = f(X)
        Y = (Y - np.min(Y))/(np.max(Y) - np.min(Y))


        if scale%5 == 0 :
            plt.plot(X,Y)
            plt.savefig("{}.png".format(round( scale*0.01, 3)))
            plt.close()
        


        score.append( calc_smoothness_each_label(Y)  )

    plt.plot(np.linspace(0,nb_scales-1,nb_scales),score)
    plt.savefig("smoothness_evolution.png")


