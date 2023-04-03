import pandas as pd
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Reshape, UpSampling2D, Conv2DTranspose, Conv2D, LeakyReLU, Flatten, Conv1DTranspose, Conv1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import RMSprop
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path
import warnings

#warnings.simplefilter(action='ignore', category=INVALID_ARGUMENT)


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


#### GENERATOR ####
#Pas besoin de le compiler car on on fera jamais de train dessus
def generator(latent_size, timeseries_size):

    G = Sequential()
    
    n_nodes = 128 * (timeseries_size//4)
    G.add(Dense(n_nodes, input_dim=latent_size))
    G.add(LeakyReLU(alpha=0.2))
    G.add(Reshape(( (timeseries_size//4), 128)))

    # Timeseries
    G.add(Conv1DTranspose(128, 4, strides=2, padding='same'))
    G.add(LeakyReLU(alpha=0.2))

    G.add(Conv1DTranspose(128, 4, strides=2, padding='same'))
    G.add(LeakyReLU(alpha=0.2))
    G.add(Conv1D(1, 7, activation='sigmoid', padding='same'))

    print(G.summary())

    return G



#### DISCRIMINATOR ####

def discriminator(in_shape = (152,1)):

    D = Sequential()

    D.add(Conv1D(64, 3, strides=2, padding='same', input_shape=in_shape))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(0.4))
    D.add(Conv1D(64, 3, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(0.4))
    D.add(Flatten())
    D.add(Dense(1, activation='sigmoid'))
    # compile D
    opt = Adam( learning_rate = 0.00005, beta_1=0.9)
    D.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    return D


#### GAN ####
def gan(G,D) :
    D.trainable = False

    GAN = Sequential()
    GAN.add(G)
    # add the discriminator
    GAN.add(D)
    # compile model
    opt = Adam(learning_rate = 0.0001, beta_1=0.9)
    GAN.compile(loss='mean_squared_error', optimizer=opt)

    return GAN



#### TRAIN #####
#Un step = entrainement d'un batch
def train(G, D, X_train, latent_size, n_epochs = 3000, batch_size = 32):

    GAN = gan(G,D)

    nb_batch_per_epoch = X_train.shape[0]//batch_size
    
    for epoch in range(n_epochs) :
        for i in range(nb_batch_per_epoch) :
        
            #Prend batch_size véritable images
            X_real = X_train[np.random.randint(0, X_train.shape[0], size=batch_size), :]
            y_real = np.ones((batch_size, 1))

            #On génère des fausses images à partir du générateur
            noise = np.random.randn(latent_size * batch_size)
            noise = noise.reshape(batch_size, latent_size)
            X_fake = G.predict(noise)[:,:,0]
            y_fake = np.zeros((batch_size, 1))

            #On concatène les fausses et les vraies images --> Former un dataset d'entrainement pour le discrimineur
            X_train_d, y_train_d = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

            #On entraine le discrimineur juste sur batch
            d_loss, _ = D.train_on_batch(X_train_d, y_train_d)

            #Une fois le discrimineur entrainé, on entraine le GAN (surtout la partie générateur)
            noise = np.random.randn(2* latent_size * batch_size)
            noise = noise.reshape(2* batch_size, latent_size)
            y_train_gan = np.ones((2*batch_size, 1)) #--> On veut in fine que le discrimneur se trompe et prend toutes les images commme réelles
            g_loss = GAN.train_on_batch(noise, y_train_gan)

            tqdm.write(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{nb_batch_per_epoch}] [D loss: {d_loss}] [G loss: {g_loss}]")


    G.save("my_D.h5")

    noise = np.random.randn(latent_size * batch_size)
    noise = noise.reshape(batch_size, latent_size)
    return G.predict(noise)[:,:,0]





if __name__ == "__main__" :
    data = load_dataset()

    latent_size = 100
    D = discriminator(in_shape = (len(data[0]), 1) )
    G = generator(latent_size, len(data[0]))
    X_temp = train(G,D, data, latent_size)


    fig, axs = plt.subplots(3)
    for j in range(3) :
        axs[j].plot(np.array(X_temp[j,:]))
    
    fig2, axs2 = plt.subplots(3)
    for j in range(3) :
        axs2[j].plot(np.array(data[j,:])) 
    
    plt.show()



    

