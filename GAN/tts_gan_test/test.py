import pandas as pd
import numpy as np
from functions import train
from GANModels import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import os.path


DATASET_NAME = "PhalangesOutlinesCorrect"


"""
latent_dim = 100
batch_size = 32
"""

def load_dataset() :
    try : 
        data = pd.read_csv(os.path.dirname(__file__) + '/../../Basic_methods_test/datasets/{}/{}_TRAIN.tsv'.format(DATASET_NAME,DATASET_NAME), sep='\t', header = None)
    except Exception as e :
        data = pd.read_csv('{}_TRAIN.tsv'.format(DATASET_NAME), sep='\t', header = None)

    unique_labels = np.sort( data[0].unique() )
    count_label = np.array( [ len(data[data[0] == label].index) for label in unique_labels ] )
    indice_min = np.argmin(count_label)
    min_label_count, label_min = np.min(count_label), unique_labels[indice_min]

    A = np.array( data[data[0] == label_min] )[:,1:]
    mini, maxi = np.min(A), np.max(A)

    A = 2*(A - mini)/(maxi - mini) - 1

    print(f"Taille du dataset d'entrainement : {len(A)}")
    print(f"Longueur des timesseries : {len(A[0])}\n")

    return A.reshape( (A.shape[0], 1, 1, A.shape[1]) )


def get_small_factor(n, factor_max = 20):
    list_factor = []
    for i in range(2,factor_max) :
        if n%i == 0:
            list_factor.append(i)

    return list_factor



def train(G, D, data, n_epochs = 50, batch_size = 32, latent_dim = 100):
    nb_batch_per_epoch = data.shape[0]//batch_size

    G_opt = torch.optim.Adam(G.parameters(), lr = 0.0002)
    D_opt = torch.optim.Adam(D.parameters(), lr = 0.0002)

    #Train mode
    G.train()
    D.train()

    d_loss_acc = 0
    g_loss_acc = 0

    for epoch in range(n_epochs) :
        for i in range(nb_batch_per_epoch) :

            G_opt.zero_grad()
            D_opt.zero_grad()

            #Train D
            X_real = torch.FloatTensor( data[np.random.randint(0, data.shape[0], size=batch_size), :] )
            y_real = torch.FloatTensor(np.ones((batch_size, 1)))

            noise = torch.FloatTensor(np.random.normal(0, 1, (batch_size,latent_dim)))
            X_fake = G(noise).detach()
            y_fake = torch.FloatTensor(np.zeros((batch_size, 1)))

            D_out_real = D(X_real)
            D_out_fake = D(X_fake)
            
            d_real_loss = nn.MSELoss()(D_out_real, y_real)
            d_fake_loss = nn.MSELoss()(D_out_fake, y_fake)
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()

            D_opt.step()


            #Train G
            noise = torch.FloatTensor(np.random.normal(0, 1, (batch_size,latent_dim)))
            X_temp = G(noise).detach()
            GAN_out_real = D(X_temp)
            y_target = torch.FloatTensor(np.ones((batch_size, 1)))
            g_loss = nn.MSELoss()(GAN_out_real, y_target)
            g_loss.backward()

            G_opt.step() 

            #Verbose
            if epoch == n_epochs - 1: 
                d_loss_acc += d_loss.item()
                g_loss_acc += g_loss.item()

            #Save fig
            if i == nb_batch_per_epoch - 1 and epoch == n_epochs - 1 :

                print(f"[D loss: {d_loss_acc/nb_batch_per_epoch}] [G loss: {g_loss_acc/nb_batch_per_epoch}]")

                return X_temp



if __name__ == "__main__" :
    data = load_dataset()

    """
    G = Generator(seq_len=data.shape[-1], patch_size=8, channels=1, num_classes=9, latent_dim=100, embed_dim=20, depth=3,
                  num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5)
    D = Discriminator(in_channels=1,
                 patch_size=8,
                 emb_size=50, 
                 seq_length = data.shape[-1],
                 depth=3, 
                 n_classes=1)

    train(G, D, data)
    """

    patch_size = get_small_factor(data.shape[-1])
    num_heads = [2,3,5,10]
    emb_size = [10, 20, 50]
    depth = [1,3,5]
    ep = [200, 500, 1000]
    batch = [16, 32, 64, 128]


    for patch in patch_size :
        for head in num_heads :
            for emb in emb_size :
                for d in depth :
                    for epo in ep :
                        for b in batch :
                            print(f"patch : {patch} , head : {head}, emb : {emb} , depth : {d}, nb_epochs : {epo}, batch_size : {b}")
                        
                            G = Generator(seq_len=data.shape[-1], patch_size=patch, channels=1, num_classes=9, latent_dim=100, embed_dim=emb, depth=d,
                                        num_heads=head, forward_drop_rate=0.5, attn_drop_rate=0.5)
                            D = Discriminator(in_channels=1,
                                        patch_size=patch,
                                        emb_size=50, 
                                        seq_length = data.shape[-1],
                                        depth=d, 
                                        n_classes=1)

                            X_temp = train(G, D, data, n_epochs = epo, batch_size = b, latent_dim = 100)

                            fig, axs = plt.subplots(3)
                            for j in range(3) :
                                axs[j].plot(np.array(X_temp[j,0,0,:]))
                            plt.savefig(f"results/example_generation_patch{patch}_head{head}_emb{emb}_depth{d}_epochs{epo}_batchsize{b}.png", dpi = 200)





    

