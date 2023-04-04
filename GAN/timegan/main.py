import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, GRU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import RMSprop
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path
from ydata_synthetic.synthesizers import ModelParameters, gan
from ydata_synthetic.preprocessing.timeseries import processed_stock
from ydata_synthetic.synthesizers.timeseries import TimeGAN


DATASET_NAME = "PhalangesOutlinesCorrect"


"""
latent_dim = 100
batch_size = 32
"""

def load_dataset() :

    data = pd.read_csv('{}_TRAIN.tsv'.format(DATASET_NAME), sep='\t', header = None)

    unique_labels = np.sort( data[0].unique() )
    count_label = np.array( [ len(data[data[0] == label].index) for label in unique_labels ] )
    indice_min = np.argmin(count_label)
    min_label_count, label_min = np.min(count_label), unique_labels[indice_min]

    A = np.array( data[data[0] == label_min] )[:,1:]
    mini, maxi = np.min(A), np.max(A)

    A = (A - mini)/(maxi - mini)

    print(f"Taille du dataset d'entrainement : {len(A)}")
    print(f"Longueur des timesseries : {len(A[0])}\n")

    return A






if __name__ == "__main__" :
    data = load_dataset()
    nb_steps = 10000


    gan_args = ModelParameters(batch_size=64,
                           lr=0.001,
                           noise_dim=100,
                           layers_dim=128)

    synth = TimeGAN(model_parameters=gan_args, hidden_dim=24, seq_len=data.shape[-1], n_seq=1, gamma=1)
    synth.train(data, train_steps=nb_steps)
    synth.save(f'models/generator_{nb_steps}.pkl')

    synth = gan.load(f'models/generator_{nb_steps}.pkl')

    synth_data = synth.sample(5)

    fig, axs = plt.subplots(5)
    for j in range(5) :
        axs[j].plot(np.array(synth_data[j,:,0]))
    
    fig2, axs2 = plt.subplots(5)
    for j in range(5) :
        axs2[j].plot(np.array(data[j,:])) 
    
    plt.show()


    

