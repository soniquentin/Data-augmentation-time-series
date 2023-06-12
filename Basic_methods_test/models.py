from fastdtw import fastdtw
import numpy as np
from tqdm import tqdm
from threading import Thread
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
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

        Input :
            data : pandas dataframe
                - data.iloc[:,0] : label
                - data.iloc[:,1:] : time series

        Output :
            model : model
            model_name : str
            kwargs : dict
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


    elif model_name == "LSTM" :
        #infine, n_steps = min(100, len(data.columns) - 1) 
        diff = max( len(data.columns) - 1 - 100 , 0) # On voudrait 100 timestamps
        if diff > 0 :
            n_steps = 100
        else :
            n_steps = len(data.columns) - 1
        batch_size = 32
        nb_step_per_epoch = ((diff + 1)*len(data.index))//batch_size
        nb_epochs = min( max(1, 1000//nb_step_per_epoch) , 20 ) #Le nombre d'epoch dépend de la taille du dataset. Mais faut que ce soit au dessus de 20 non plus

        #Le nombre d'epoch dépend de la taille du dataset
        kwargs = {"epochs" : nb_epochs, "batch_size" : batch_size, "verbose" : 1}
        
        
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(n_steps, 1))) #Le 1 correspond signifie qu'on a affaire à des time series univariées
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(data[0].unique()), activation='softmax'))
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])

        return (model, "LSTM"), kwargs


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

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.models import Sequential
    import time
    import pandas as pd
    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt

    path = "../datasets/Wafer/Wafer_TRAIN.tsv"
    data = pd.read_csv(path ,sep='\t', header =None)

    path_test = "../datasets/Wafer/Wafer_TEST.tsv"
    data_test = pd.read_csv(path_test ,sep='\t', header =None)



    def load_data(data):
        # Split features and labels
        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values

        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        return X, y

    
    # Prepare data for LSTM
    def prepare_data(X, y, n_steps):
        X_new, y_new = [], []
        for i in range(len(X)):
            for j in range(len(X[i]) - n_steps):
                X_new.append(X[i][j:j+n_steps])
                y_new.append(y[i])
        return np.array(X_new), np.array(y_new)


    # Build the LSTM model
    def build_model(n_steps, n_features, n_classes):
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=( n_steps , n_features)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        return model

    # Train the LSTM model
    def train_model(model, X_train, y_train, epochs=10, batch_size=32):
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Evaluate the model
    def evaluate_model(model, X_test, y_test):
        _, accuracy = model.evaluate(X_test, y_test)
        print('Accuracy:', accuracy)


    # Assuming your data is stored in a pandas DataFrame called 'df'
    # where the first column is the label and the rest are time series data
    X, y = load_data(data)
    X_test, y_test = load_data(data_test)

    X,y = X[:500], y[:500]

    #n_steps = X.shape[1]
    #X, y = prepare_data(X, y, n_steps)

    X = X.reshape((X.shape[0], X.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


    n_classes = len(np.unique(y))

    # Build the LSTM model
    model = build_model(X.shape[1], 1, n_classes)

    # Train the model
    train_model(model, X, y, epochs = 2, batch_size = 32)

    # Make predictions with the trained LSTM model
    # Evaluate the model
    evaluate_model(model, X_test, y_test)





