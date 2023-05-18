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
import joblib

import sys
import os
from pathlib import Path
#Rajoute le dossier Basic_methods_test/ dans sys.path pour pouvoir importer les fonctions
parent_path = str(Path(os.getcwd()).parent.absolute())
sys.path.append(parent_path + "/GAN/timegan")
from ydata_synthetic.synthesizers import ModelParameters, gan
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN
sys.path.append(parent_path + "/dtw_smote")
from dtw_smote import new_samples


def plot_examples(data_label, new_data, dataset_name, label, da_method) :
    """
        Plot des examples de generations
        data_label : numpy array des vraies données
        new_data : numpy array de données synthétisées
        dataset_name : nom de dataset
        label : label qui sont générées (entier)
        da_method : nom de la méthode de DA utilisée
    """
    
    #Créer un dossier pour le model pour sauvegarder les exemples et la k_matrice
    dataset_folder = f"{parent_path}/Examples/{dataset_name}"
    if not os.path.exists(dataset_folder) :
        os.makedirs(dataset_folder)

    #Plot quelques exemples
    nb_to_plot = min(5, new_data.shape[0])
    if nb_to_plot > 1 : 
        fig, axs = plt.subplots(nb_to_plot, sharey=True)
        for j in range(nb_to_plot) :
            if da_method == "GAN" :
                axs[j].plot(np.array(new_data[j,:,0]))
            else : 
                axs[j].plot(np.array(new_data[j,:]))
    else :
        plt.plot(new_data[0])
    plt.savefig(f"{dataset_folder}/{label}_{da_method}.png", dpi=200)
    
    #Plot les vraies données
    nb_to_plot = min(5, data_label.shape[0])
    if nb_to_plot > 1 : 
        fig, axs = plt.subplots(nb_to_plot, sharey=True)
        for j in range(nb_to_plot) :
            axs[j].plot(np.array(data_label[j,:]))
    else :
        plt.plot(np.array(data_label[0]))
    plt.savefig(f"{dataset_folder}/{label}.png", dpi=200)


def timeseries_smote(data, name_trans = "Basic",  k_neighbors = 3, sampling_strategy = None, dataset_name = None) :
    """
        name_trans = "Basic" (Basic Smote), "Ada"  (Adasyn)
    """

    x = data.drop([0], axis = 1)
    y = data[0]

    x = np.array(x)
    y = np.array(y)
    nb_real_samples = x.shape[0]

    #Dictionnary with count of each label in y
    count_dict = {}
    for label in np.unique(y) :
        count_dict[label] = len(y[y == label])
    
    if name_trans == "Basic" :
        smote = SMOTE(sampling_strategy= sampling_strategy ,  k_neighbors=k_neighbors)
        x, y = smote.fit_resample(x, y)
    elif name_trans == "Ada" :
        adasyn = ADASYN(sampling_strategy= sampling_strategy, n_neighbors=k_neighbors)
        x, y = adasyn.fit_resample(x, y)

    #Plot examples
    real_x = x[:nb_real_samples]
    real_y = y[:nb_real_samples]
    synthetic_x = x[nb_real_samples:]
    synthetic_y = y[nb_real_samples:]
    for label in np.unique(synthetic_y) :
        plot_examples(real_x[real_y == label], new_data = synthetic_x[synthetic_y == label] , dataset_name = dataset_name, label = label, da_method = name_trans)
    
    new_samples = pd.DataFrame(x, columns = [i+1 for i in range(len(x[0]))])
    new_samples[0] = pd.DataFrame(y)

    return new_samples.reset_index().drop(["index"], axis = 1)




def timeseries_trans(data, name_trans, minor_class, major_class, dataset_name) :
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

    plot_examples_bool = True

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
            if plot_examples_bool :
                plot_examples_bool = False
                plot_examples(data_label = np.array(data_minor), new_data = new_samples, dataset_name = dataset_name, label = l_minor, da_method = name_trans)


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
    """
        data = pd.DataFrame
        dataset_name = "ECG5000" or "GunPoint"
        sampling_strategy = {label : nb_sample}

        Return : pd.DataFrame
    """

    datafinal = data.copy()

    for label in sampling_strategy :

        dataset_folder = f"{parent_path}/GAN/timegan/models/{dataset_name}"
        if not os.path.exists(dataset_folder) :
            os.makedirs(dataset_folder)

        current_label_nb = len(data[data[0] == label].index)
        if current_label_nb != sampling_strategy[label] : #ce label n'est pas déjà à son nombre final

            data_label = np.array( data[data[0] == label] )[:,1:]
            mini, maxi = np.min(data_label), np.max(data_label)
            data_label = (data_label - mini)/(maxi - mini) #Normalize

            if not os.path.exists(f"{dataset_folder}/{label}.pkl") :  #Un modèle n'a pas été entrainé 
                print(f"Train GAN for label {label}...")

                nb_steps = 2500
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


            with open(f"{dataset_folder}/{label}_maxmin.pkl", 'rb') as f:

                maxmin = pickle.load(f)
                maxi, mini = maxmin['maxi'], maxmin['mini']

                synth = gan.load(f"{dataset_folder}/{label}.pkl")

                #Plot des examples
                synth_data = synth.sample(5)
                plot_examples(data_label = data_label, new_data = synth_data, dataset_name = dataset_name, label = label, da_method = "GAN")


                synth_data = synth.sample( sampling_strategy[label] - current_label_nb )
                new_data = np.array(synth_data[:sampling_strategy[label] - current_label_nb,:,0])

                new_samples = pd.DataFrame(new_data*(maxi - mini) + mini, columns = [i+1 for i in range(len(data.columns) - 1)])
                new_samples[0] = label

                datafinal = pd.concat([datafinal,new_samples], axis=0)


    return datafinal



def dtw_smote(data, dataset_name, sampling_strategy = None) :
    """
        data = pd.DataFrame
        dataset_name = "ECG5000" or "GunPoint"
        sampling_strategy = {label : nb_sample}

        Return : pd.DataFrame
    """

    datafinal = data.copy()

    for label in sampling_strategy :

        #Créer un dossier pour le model pour sauvegarder les exemples et la k_matrice
        dataset_folder = f"{parent_path}/dtw_smote/models/{dataset_name}"
        if not os.path.exists(dataset_folder) :
            os.makedirs(dataset_folder)
        
        current_label_nb = len(data[data[0] == label].index)
        if current_label_nb != sampling_strategy[label] : #ce label n'est pas déjà à son nombre final

            data_label = np.array( data[data[0] == label] )[:,1:]

            if not os.path.exists(f"{dataset_folder}/k_matrix_{label}.pkl") :  #Si la k_matrix n'existe pas déjà

                new_data, k_matrix = new_samples(data = data_label, n_new = max( sampling_strategy[label] - current_label_nb , 5), k_neighbors = 3)

                #On sauvegarde la k_matrix
                joblib.dump(k_matrix, f"{dataset_folder}/k_matrix_{label}.pkl")

            else :
                k_matrix = joblib.load(f"{dataset_folder}/k_matrix_{label}.pkl")
                new_data, _ = new_samples(data = data_label, n_new = max( sampling_strategy[label] - current_label_nb , 5), k_neighbors = 3, k_matrix = k_matrix)
            
            #Plot des examples
            plot_examples(data_label = data_label, new_data = new_data, dataset_name = dataset_name, label = label, da_method = "DTW-SMOTE")


            #Formate les nouvelles données
            new_data = pd.DataFrame(new_data[:sampling_strategy[label] - current_label_nb], columns = [i+1 for i in range(len(data.columns) - 1)])
            new_data[0] = label
            
            #Concatène les nouvelles données aux données existantes
            datafinal = pd.concat([datafinal,new_data], axis=0)
    
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
                [0] + [rd.randint(0,10) for i in range(7)],
                [0] + [rd.randint(0,10) for i in range(7)],
                [0] + [rd.randint(0,10) for i in range(7)],
                [0] + [rd.randint(0,10) for i in range(7)],
                [1, 0, 1, 2, 3, 2, 1, 0],
                [1, 1, 3, 5, 3, 1, 1, 0],
                [1, 5, 4, 2, 3, 4, 2, 1]]
    data = pd.DataFrame( np.array(data_ts, dtype = "float") , columns=[i for i in range(8)]  )

    sampling_strategy = {0 : 9, 1 : 9}

    print("\n\nDTW-SMOTE")
    new_data = dtw_smote(data, "TEST", sampling_strategy = sampling_strategy)
    print(new_data)
    