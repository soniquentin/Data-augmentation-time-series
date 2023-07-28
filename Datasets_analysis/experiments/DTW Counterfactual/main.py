from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import random
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def create_distance_matrix(matrix1, matrix2 = None):
    """
    Create distance matrix between matrix1 and matrix2

    If matrix2 is None, then matrix2 = matrix1
    """
    if matrix2 is None : 
        n = matrix1.shape[0]
        distance_matrix = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(i):
                distance = dtw.distance(matrix1[i], matrix1[j])
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
    else : 
        n1, n2 = matrix1.shape[0], matrix2.shape[0]
        distance_matrix = np.zeros((n1, n2))
        for i in tqdm(range(n1)):
            for j in range(n2):
                distance_matrix[i][j] = dtw.distance(matrix1[i], matrix2[j])

    return distance_matrix


def initilize_points(matrix, sub_n) :
    """
    Return a sub_matrix of matrix with sub_n rows and the rest of matrix
    """
    n = matrix.shape[0]
    random_rows = np.random.choice(n, sub_n, replace=False)

    return np.array([1 if i in random_rows else 0 for i in range(n)])



def DTW_C(matrix_min, matrix_max) :
    """
    DTW_C algorithm
    """


    ### ========================================== ###
    ### ============== CREATE PAIRS ============== ###
    ### ========================================== ###
    
    # Initilize sub_matrix_max and rest_matrix_max
    n_min = matrix_min.shape[0]
    n_max = matrix_max.shape[0]
    selected_points = initilize_points(matrix_max, n_min)

    #Create distance matrix between matrix_min and sub_matrix_max
    distance_matrix = create_distance_matrix(matrix_max[selected_points == 1], matrix_min)
    print(distance_matrix.shape)
    exit()

    #For each row in sub_matrix_max, find the row in matrix_min with the smallest distance
    pairs = []
    for i in range(n_min) :
        pairs.append(np.argmin(distance_matrix[:, i]))

    ### ========================================== ###
    ### ========================================== ###
    ### ========================================== ###




    ### ========================================== ###
    ### =====
    ### ========================================== ###
    
    #Create distance matrix between sub_matrix_max and rest_matrix_max
    distance_matrix = create_distance_matrix(sub_matrix_max, rest_matrix_max)










    



#Import datatset.tsv
df = pd.read_csv('dataset.tsv', sep='\t', header = None)

#Label is the first column
#Sub dataframe with row with label 0
df_0 = df.loc[df[0] == 0]
#Sub dataframe with row with label 1
df_1 = df.loc[df[0] == 1]

#Drop the first column and convert to numpy array with type double
matrix_0 = df_0.drop([0], axis=1).to_numpy()
matrix_1 = df_1.drop([0], axis=1).to_numpy()

"""
#Create distance matrix for df_0
distance_matrix_0 = create_distance_matrix(df_0)
"""

if matrix_0.shape[0] < matrix_1.shape[0] : 
    matrix_min = matrix_0
    matrix_max = matrix_1
else :
    matrix_min = matrix_1
    matrix_max = matrix_0

DTW_C(matrix_min, matrix_max)





