from fastdtw import fastdtw
import numpy as np
from tqdm import tqdm
from threading import Thread
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import random as rd
from pyts.classification import TimeSeriesForest
from sklearn.neighbors import KNeighborsClassifier
from pyts.classification import LearningShapelets
from rocket.rocket_functions import generate_kernels
from sklearn.linear_model import RidgeClassifierCV


def get_model(model_name, data):
    """
        model_name : "RF" (Random Forest), "NN" (Simple fully connected layer)
    """

    if model_name == "RF" :
        random_state = rd.randint(1,100)
        return  (RandomForestClassifier(n_estimators = 130,
                                       max_depth = 50, #Set to 50 instead of None to prevent from overfitting
                                       random_state = random_state), "RF") , {}
    elif model_name == "NN" :
        nb_timestamp = len(data.columns) - 1

        model = Sequential()
        
        model.add(Dense(64,  input_dim=nb_timestamp, kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        model.add(Activation("relu"))
        model.add(Dropout(rate = 0.10)) #Reduce overfitting
        model.add(Dense(len(data[0].unique()), kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
        model.add(Activation("softmax"))

        model.compile(loss='mean_absolute_error', optimizer= Adam(learning_rate = 0.001), metrics=['mean_absolute_error'])

        kwargs = {"epochs" : 100, "batch_size" : 32, "verbose" : 0}

        return (model, "NN"), kwargs

    elif model_name == "DTW_NEIGBOURS" :

        def DTW_FAST(x,y) :
            dist, _ = fastdtw(x, y)
            return dist

        return (KNeighborsClassifier(n_neighbors = 3, metric = DTW_FAST), "DTW_NEIGBOURS"), {}
    
    elif model_name == "TS-RF" :
        random_state = rd.randint(1,100)
        return (TimeSeriesForest(n_estimators = 130,
                                       max_depth = 50, #Set to 50 instead of None to prevent from overfitting
                                       n_windows = 10,
                                       random_state = random_state), "TS-RF") , {}

    elif model_name == "SHAPELET" :
        random_state = rd.randint(1,100)
        return ( LearningShapelets(random_state=random_state, tol=0.01) , "SHAPELET"), {}

    elif model_name == "KERNEL" :
        nb_timestamp = len(data.columns) - 1
        kernels = generate_kernels(nb_timestamp, 10000)
        return ( (RidgeClassifierCV(alphas = np.logspace(-3, 3, 10)),kernels), "KERNEL") , {}
    
if __name__ == "__main__" :


    import pandas as pd
    from sklearn.metrics import accuracy_score
    import time
    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt

    path = "../datasets/Wafer/Wafer_TRAIN.tsv"
    path_test = "../datasets/Wafer/Wafer_TEST.tsv"
    data = pd.read_csv(path ,sep='\t', header =None)
    """
    X_train, y_train = np.array( data.drop([0], axis = 1) ),  np.array(data[0])
    data_test = pd.read_csv(path_test ,sep='\t', header =None)
    X_test, y_test = np.array( data_test.drop([0], axis = 1) ), np.array(data_test[0])
    """

    def DTW_FAST(x,y) :
        dist, _ = fastdtw(x, y)
        return dist  

    tsne = TSNE(n_components = 2, perplexity = 100,  metric = DTW_FAST)

    data_transformed = tsne.fit_transform(data.drop([0], axis = 1))
    colours = sns.color_palette("hls", len(np.unique(data[0])))
    sns.scatterplot(x=data_transformed[:,0], y=data_transformed[:,1], hue = data[0], legend='full', palette=colours)
    plt.show()




