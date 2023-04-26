import pandas as pd
import numpy as np
from tsaug.visualization import plot
import matplotlib.pyplot as plt
import tsaug
from tsaug import TimeWarp, Crop, Quantize, Drift, Reverse, AddNoise
import random as rd
from imblearn.over_sampling import SMOTE, ADASYN
import pickle
from sklearn.manifold import TSNE
import seaborn as sns

import sys
import os
from pathlib import Path
#Rajoute le dossier Basic_methods_test/ dans sys.path pour pouvoir importer les fonctions
parent_path = str(Path(os.getcwd()).parent.absolute())
sys.path.append(parent_path + "/GAN/timegan")
from ydata_synthetic.synthesizers import ModelParameters, gan
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN
sys.path.append(parent_path + "/twd_smote")




def timeseries_smote(data, name_trans = "Basic",  k_neighbors = 3, sampling_strategy = None) :
    """
        name_trans = "Basic" (Basic Smote), "Ada"  (Adasyn)
    """

    x = data.drop([0], axis = 1)
    y = data[0]

    x = np.array(x)
    y = np.array(y)
    
    if name_trans == "Basic" :
        smote = SMOTE(sampling_strategy= sampling_strategy ,  k_neighbors=k_neighbors)
        x, y = smote.fit_resample(x, y)
    elif name_trans == "Ada" :
        adasyn = ADASYN(sampling_strategy= sampling_strategy, n_neighbors=k_neighbors)
        x, y = adasyn.fit_resample(x, y)
    
    new_samples = pd.DataFrame(x, columns = [i+1 for i in range(len(x[0]))])
    new_samples[0] = pd.DataFrame(y)

    return new_samples.reset_index().drop(["index"], axis = 1)




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
                X_aug = AddNoise(scale=0.1*np.std(X)).augment(X)

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

    data_to_return = data.reset_index().drop(["index"], axis = 1)

    return data_to_return[data_to_return[0] == l_minor]



def gan_augmentation(data, dataset_name, sampling_strategy = None):

    datafinal = data.copy()

    for label in sampling_strategy :

        dataset_folder = f"{parent_path}/GAN/timegan/models/{dataset_name}"
        if not os.path.exists(dataset_folder) :
            os.makedirs(dataset_folder)

        current_label_nb = len(data[data[0] == label].index)
        if current_label_nb != sampling_strategy[label] : #ce label n'est pas déjà à son nombre final

            if not os.path.exists(f"{dataset_folder}/{label}.pkl") :  #Un modèle n'a pas été entrainé 
                print(f"Train GAN for label {label}...")
                data_label = np.array( data[data[0] == label] )[:,1:]
                mini, maxi = np.min(data_label), np.max(data_label)
                data_label = (data_label - mini)/(maxi - mini) #Normalize

                nb_steps = 100
                gan_args = ModelParameters(batch_size=64,
                                        lr=0.001,
                                        noise_dim=100,
                                        layers_dim=128)

                synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=data_label.shape[-1], n_seq=1, gamma=1)
                synth.train(np.reshape(data_label,(data_label.shape[0], data_label.shape[-1], 1)), train_steps=nb_steps)
                synth.save(f"{dataset_folder}/{label}.pkl")

                #Sauvegarde le min et max
                with open(f"{dataset_folder}/{label}_maxmin.pkl", 'wb') as f:
                    pickle.dump({'maxi': maxi, 'mini' : mini}, f, protocol=pickle.HIGHEST_PROTOCOL)

                #Sauvegarde des exemples de générations
                synth_data = synth.sample(5)
                fig, axs = plt.subplots(5, sharey=True)
                for j in range(5) :
                    axs[j].plot(np.array(synth_data[j,:,0]))
                plt.savefig(f"{dataset_folder}/exemples_{label}_generated.png", dpi=200)
                
                fig2, axs2 = plt.subplots(5, sharey=True)
                for j in range(5) :
                    axs2[j].plot(np.array(data_label[j,:]))
                plt.savefig(f"{dataset_folder}/exemples_{label}.png", dpi=200)
                

            with open(f"{dataset_folder}/{label}_maxmin.pkl", 'rb') as f:

                maxmin = pickle.load(f)
                maxi, mini = maxmin['maxi'], maxmin['mini']

                synth = gan.load(f"{dataset_folder}/{label}.pkl")
                synth_data = synth.sample( sampling_strategy[label] - current_label_nb )
                new_data = np.array(synth_data[:sampling_strategy[label] - current_label_nb,:,0])

                new_samples = pd.DataFrame(new_data*(maxi - mini) + mini, columns = [i+1 for i in range(len(data.columns) - 1)])
                new_samples[0] = label

                datafinal = pd.concat([datafinal,new_samples], axis=0)


    return datafinal
        


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

    print("\n\nTEST RANDOM")
    new_data = timeseries_trans(data, name_trans = "ROS", minor_class = (1, 3), major_class = (0 , 5))
    print(new_data[new_data[0] == 1])

    sampling_strategy = {0 : 5, 1 : 5}

    print("\n\nTEST BASIC SMOTE")
    new_data = timeseries_smote(data, name_trans = "Basic", k_neighbors = 2, sampling_strategy = sampling_strategy)
    print(new_data[new_data[0] == 1])

    print("\n\nGAN")
    new_data = gan_augmentation(data, "TEST", sampling_strategy = sampling_strategy)
    print(new_data)
    