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


class Dtw_neigbours :

    def __init__(self) :
        pass

    def fit(self, X, y, k_neighbors = 5):
        self.train_data = X
        self.train_label = y
        self.nb_train = len(X)
        self.k_neighbors = k_neighbors
    
    def predict(self, X_test) :

        final_prediction = []

        for i in tqdm( range(len(X_test)) ) :
            dist_0, _ = fastdtw(X_test[i], self.train_data[0])
            dist_list = [( self.train_label[0] ,dist_0 )]

            for j in range(1, self.nb_train):
                dist, _ = fastdtw(X_test[i], self.train_data[j])

                previous_length = len(dist_list)
                ind_in_dist_list = None
                for k in range(previous_length) :
                    a,b = dist_list[k]
                    if dist < b :
                        ind_in_dist_list = k
                        break
                
                if ind_in_dist_list != None :
                    dist_list = dist_list[:ind_in_dist_list] + [( self.train_label[j] , dist)] + dist_list[ind_in_dist_list:]
                    dist_list = dist_list[:min(self.k_neighbors, previous_length + 1)]

            count_list = {}
            for neighbors, _ in dist_list :
                if neighbors in count_list :
                    count_list[neighbors] += 1
                else :
                    count_list[neighbors] = 1

            final_prediction.append( max(count_list, key=count_list.get) )

        return np.array(final_prediction)



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
        return (Dtw_neigbours(), "DTW_NEIGBOURS"), {}
    
    elif model_name == "TS-RF" :
        random_state = rd.randint(1,100)
        return (TimeSeriesForest(n_estimators = 130,
                                       max_depth = 50, #Set to 50 instead of None to prevent from overfitting
                                       n_windows = 10,
                                       random_state = random_state), "TS-RF") , {}

    
if __name__ == "__main__" :
    x = np.array([1, 2, 3, 4, 5], dtype='float')
    y = np.array([2, 3, 4], dtype='float')

    print( fastdtw(x, y) ) 