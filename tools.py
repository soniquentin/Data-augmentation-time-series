import pandas as pd
import numpy as np
from tsaug.visualization import plot
import matplotlib.pyplot as plt
import tsaug
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise
import random as rd
from imblearn.over_sampling import SMOTE, ADASYN


def timeseries_smote(data, name_trans = "Basic",  k_neighbors = 3) :
    """
        name_trans = "Basic" (Basic Smote), "Ada"  (Adasyn)
    """

    x = data.drop([0], axis = 1)
    y = data[0]

    x = np.array(x)
    y = np.array(y)
    
    if name_trans == "Basic" :
        smote = SMOTE(sampling_strategy=1,  k_neighbors=k_neighbors)
        x, y = smote.fit_resample(x, y)
    elif name_trans == "Ada" :
        adasyn = ADASYN(sampling_strategy=1, n_neighbors=k_neighbors)
        x, y = adasyn.fit_resample(x, y)
    
    new_samples = pd.DataFrame(x, columns = [i+1 for i in range(len(x[0]))])
    new_samples[0] = pd.DataFrame(y)

    return new_samples




def timeseries_trans(data, name_trans, minor_class, major_class) :
    """
        name_trans = "TW" (timewarping) ; "Jit" (jittering) ; "ROS" (Random OverSampling)
        minor_class = (label, count)
        major_class = (label, count)
    """
    l_minor, cnt_min = minor_class
    l_major, cnt_maj = major_class

    initial_cnt_min = cnt_min

    data_minor = data[data[0] == l_minor]
    data_minor = data_minor.drop([0], axis = 1) #On retire les labels


    while cnt_maj -  cnt_min > 0 :

        new_samples = []

        def transfo(row) :
            
            X = np.array(row)

            if name_trans == "TW" :
                X_aug = TimeWarp(n_speed_change = 1, seed = rd.randint(1,200)).augment(X)
            elif name_trans == "Jit" :
                X_aug = AddNoise(scale=0.01).augment(X)

            new_samples.append(X_aug)

        nb_sample_to_create = min(initial_cnt_min, cnt_maj -  cnt_min)

        if name_trans == "ROS" :
            new_samples = data_minor.sample(nb_sample_to_create)
        else :
            data_minor.head(nb_sample_to_create).apply(transfo, axis=1)
            new_samples = np.array(new_samples)

        new_samples = pd.DataFrame(new_samples, columns = [i+1 for i in range(len(data_minor.columns))])
        new_samples[0] = l_minor


        #Concat the new sample with data
        data = pd.concat([data,new_samples], axis=0)

        #Update cnt_maj and cnt_min
        cnt_maj = data[data[0] == l_major].shape[0]
        cnt_min = data[data[0] == l_minor].shape[0]

    return data.reset_index().drop(["index"], axis = 1)


if __name__ == "__main__" :

    #dataset_folder = "./datasets"
    #data = pd.read_csv(dataset_folder + "/Earthquakes/Earthquakes_TRAIN.tsv" ,sep='\t', header =None)

    #On va tester les transformations
    data_ts = [ [0] + [rd.randint(0,10) for i in range(7)], 
                [0] + [rd.randint(0,10) for i in range(7)], 
                [0] + [rd.randint(0,10) for i in range(7)],
                [0] + [rd.randint(0,10) for i in range(7)],
                [0] + [rd.randint(0,10) for i in range(7)],
                [1, 0, 1, 2, 3, 2, 1, 0],
                [1, 1, 3, 5, 3, 1, 1, 0],
                [1, 5, 4, 2, 3, 4, 2, 1]]
    data = pd.DataFrame( np.array(data_ts, dtype = "float") , columns=[i for i in range(8)]  )

    print("OLD DATA")
    print(data[data[0] == 1]) 

    print("\n\nTEST TIMEWARPING")
    new_data = timeseries_trans(data, name_trans = "TW", minor_class = (1, 3), major_class = (0 , 5))
    print(new_data[new_data[0] == 1])

    print("\n\nTEST JITTERING")
    new_data = timeseries_trans(data, name_trans = "Jit", minor_class = (1, 3), major_class = (0 , 5))
    print(new_data[new_data[0] == 1])

    print("\n\nTEST BASIC SMOTE")
    new_data = timeseries_smote(data, name_trans = "Basic", k_neighbors = 2)
    print(new_data[new_data[0] == 1])

    print("\n\nTEST RANDOM")
    new_data = timeseries_trans(data, name_trans = "ROS", minor_class = (1, 3), major_class = (0 , 5))
    print(new_data[new_data[0] == 1])
    