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

    def fit(self, X, y):
        self.train_data = X
        self.train_label = y
        self.nb_train = len(X)
    
    def predict(self, X_test) :

        def thread_run(num, total_treads, nb_test) :
            for i in range(nb_test) :
                if i%total_treads == num :
                    
                    j_min = 0
                    dist_min, _ = fastdtw(X_test[i], self.train_data[0])
                    
                    for j in range(1, self.nb_train):
                        dist, _ = fastdtw(X_test[i], self.train_data[j])

                        if dist < dist_min :
                            dist_min = dist
                            j_min = j

                    final_prediction.append( (i, self.train_label[j_min]) )

                    if i%(5*total_treads) == num :
                        print("Thread {} : {}/{}".format(num, i, nb_test))


        final_prediction = []

        total_treads = 20
        threads = [Thread(target=thread_run, args=(i,total_treads,len(X_test))) for i in range(total_treads)]

        for thread in threads :
            thread.start()

        for thread in threads :
            thread.join()

        
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
        model.add(Dense(1, kernel_initializer=initializers.RandomNormal(stddev=0.1), bias_initializer = initializers.Zeros() ))
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