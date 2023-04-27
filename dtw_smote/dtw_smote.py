from fastdtw import fastdtw
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import pandas as pd
from tqdm import tqdm

def polate(X_basic, X_target, coef = None, mode = None):
    """
        mode : "interpolate", "extrapolate", "best"

        'X_basic' & 'X_target' must be numpy array
    """
    if coef == None :
        coef = np.random.uniform(0,1)
    if mode == None :
        mode = "interpolate"

    X_inter = X_basic + coef*(X_target - X_basic)
    X_extra = X_basic + coef*(X_basic - X_target)

    if mode == "interpolate" :
        return X_inter
    elif mode == "extrapolate" :
        return X_extra
    elif mode == "best" :
        mean_inter, std_inter = np.mean(X_inter), np.std(X_inter)
        mean_extra, std_extra = np.mean(X_extra), np.std(X_extra)
        mean_basic, std_basic = np.mean(X_basic), np.std(X_basic)
        mean_target, std_target = np.mean(X_target), np.std(X_target)

        D_inter_basic = (mean_inter - mean_basic)**2 + (std_inter - std_basic)**2
        D_inter_target = (mean_inter - mean_target)**2 + (std_inter - std_target)**2
        D_extra_basic = (mean_extra - mean_basic)**2 + (std_extra - std_basic)**2
        D_extra_target = (mean_extra - mean_target)**2 + (std_extra - std_target)**2

        if D_extra_basic <= 4*D_inter_basic or D_extra_target <= 4*D_inter_target :
            return X_extra
        else :
            return X_inter



def get_k_matrix(data, k_neighbors = 3) :
    """
        Returns a matrix (data.shape[0], k_neighbors + 1) with the k_neighbors nearest neighbors for each series
    """

    def DTW_FAST(x,y) :
        dist, _ = fastdtw(x, y)
        return dist

    dist_matrix = np.zeros( (data.shape[0], data.shape[0]) )
    k_matrix = []
    k = min(data.shape[0] - 1, k_neighbors + 1)

    for i in tqdm(range(data.shape[0]), desc = "TDW-Smote ") :
        for j in range(i+1,data.shape[0]) :
            distance_i_j = DTW_FAST(data[i],data[j])
            dist_matrix[i,j] = distance_i_j
            dist_matrix[j,i] = distance_i_j

        k_matrix.append(np.argpartition(dist_matrix[i], k)[:k])

    return k_matrix


def new_samples(data, n_new, k_neighbors = 3, k_matrix = None) :
    """
        Generate 'n_new' new samples of same type as 'data'

        'data' must be an numpy matrix
    """

    if k_matrix is None :
        k_matrix = get_k_matrix(data = data, k_neighbors = k_neighbors)

    new_data = [] #Liste avec les nouvelles data

    while n_new > 0 :
        index = rd.randint(0,data.shape[0] - 1)
        for neighbors in k_matrix[index] :
            if neighbors != index and n_new > 0: #k_matrix contient forcément lui-même car dist_matrix a que des zeros sur sa diagonale ==> comme on a pris les k_neighbors + 1 plus petites distances, lui-même y est forcément
                new_data.append( polate(data[neighbors], data[index], mode = "extrapolate") )
                n_new -= 1


    return np.array(new_data), k_matrix



if __name__ == "__main__" :
        
    data = pd.read_csv("../datasets/MiddlePhalanxOutlineCorrect/MiddlePhalanxOutlineCorrect_TRAIN.tsv", sep='\t', header =None)

    data = np.array(data[data[0] == 0])[:,1:]

    a = new_samples(data, 3) 

    for Y in data : 
        plt.plot(Y, color = "grey")

    for Y in a :
        plt.plot(Y, color = "r")
    

    plt.show()
