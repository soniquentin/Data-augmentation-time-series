from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization, Reshape, UpSampling2D, Conv2DTranspose, Conv2D, LeakyReLU, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import RMSprop

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm




##### LOAD DATASET ET AFFICHE LES 9 PREMIERES IMAGES #######
def load_dataset() :
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))

    """
    for i in range(9):
        #fig.add_subplot(235) is the same as fig.add_subplot(2, 3, 5)
        plt.subplot(330 + 1 + i)
        plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.savefig(f"example_generation_step.png", dpi = 10)
    """

    X_train = X_train.astype('float32')
    X_train = X_train / 255.0 #Scale

    return X_train




#### GENERATOR ####
#Pas besoin de le compiler car on on fera jamais de train dessus
def generator():

    G = Sequential()
    
    n_nodes = 128 * 7 * 7
    G.add(Dense(n_nodes, input_dim=100))
    G.add(LeakyReLU(alpha=0.2))
    G.add(Reshape((7, 7, 128)))

    # Upsample to 14x14
    G.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    G.add(LeakyReLU(alpha=0.2))

    # Upsample to 28x28
    G.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    G.add(LeakyReLU(alpha=0.2))
    G.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))

    return G




#### DISCRIMINATOR ####

def discriminator(in_shape = (28,28,1)):

    D = Sequential()

    D.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(0.4))
    D.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(0.4))
    D.add(Flatten())
    D.add(Dense(1, activation='sigmoid'))
    # compile D
    opt = Adam( learning_rate = 0.0002, beta_1=0.5)
    D.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return D



#### GAN ####
def gan(G,D) :
    D.trainable = False

    GAN = Sequential()
    GAN.add(G)
    # add the discriminator
    GAN.add(D)
    # compile model
    opt = Adam(learning_rate = 0.0002, beta_1=0.5)
    GAN.compile(loss='binary_crossentropy', optimizer=opt)

    return GAN




#### TRAIN #####
#Un step = entrainement d'un batch
def train(G, D, X_train, n_epochs = 50, batch_size = 128):

    GAN = gan(G,D)

    nb_batch_per_epoch = X_train.shape[0]//batch_size
    
    for i in tqdm(range(n_epochs*nb_batch_per_epoch)):
        
        #Prend batch_size véritable images
        X_real = X_train[np.random.randint(0, X_train.shape[0], size=batch_size), :, :]
        y_real = np.ones((batch_size, 1))

        #On génère des fausses images à partir du générateur
        noise = np.random.randn(100 * batch_size)
        noise = noise.reshape(batch_size, 100)
        X_fake = G.predict(noise)[:,:,:,0]
        y_fake = np.zeros((batch_size, 1))

        #On concatène les fausses et les vraies images --> Former un dataset d'entrainement pour le discrimineur
        X_train_d, y_train_d = np.vstack((X_real, X_fake)), np.vstack((y_real, y_fake))

        #On entraine le discrimineur juste sur batch
        d_loss, _ = D.train_on_batch(X_train_d, y_train_d)

        #Une fois le discrimineur entrainé, on entraine le GAN (surtout la partie générateur)
        noise = np.random.randn(100 * batch_size)
        noise = noise.reshape(batch_size, 100)
        y_train_gan = np.ones((batch_size, 1)) #--> On veut in fine que le discrimneur se trompe et prend toutes les images commme réelles
        g_loss = GAN.train_on_batch(noise, y_train_gan)

        print('d=%.3f, g=%.3f' % (d_loss, g_loss))

        if i%1000 == 0 : #Tous les 100 steps, sauvegarde une image de génération
            for j in range(9):
                #fig.add_subplot(235) is the same as fig.add_subplot(2, 3, 5)
                plt.subplot(330 + 1 + j)
                plt.imshow(X_fake[j], cmap=plt.get_cmap('gray'))

            plt.savefig(f"example_generation_step_{i}.png", dpi = 100)

    
    G.save("my_D.h5")


if __name__ == "__main__" :

    X_train = load_dataset()

    D = discriminator()
    G = generator()
    train(G,D, X_train)

