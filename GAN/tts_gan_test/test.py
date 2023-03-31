import pandas as pd
import numpy as np
from functions import train
from GANModels import *
from tqdm import tqdm
import matplotlib.pyplot as plt

latent_dim = 100
batch_size = 32

def load_dataset() :
    data = pd.read_csv('Wafer_TRAIN.tsv', sep='\t', header = None)
    A = np.array(data[data[0] == -1])[:,1:]
    return A.reshape( (A.shape[0], 1, 1, A.shape[1]) )



def train(G, D, data, n_epochs = 200, batch_size = 32):
    nb_batch_per_epoch = data.shape[0]//batch_size

    G_opt = torch.optim.Adam(G.parameters(), lr = 0.0002)
    D_opt = torch.optim.Adam(D.parameters(), lr = 0.0002)

    #Train mode
    G.train()
    D.train()

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
                tqdm.write(f"[Epoch {epoch}/{n_epochs}] [Batch {i}/{nb_batch_per_epoch}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

            #Save fig
            if i == nb_batch_per_epoch - 1 and epoch == n_epochs - 1 :
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

    patch_size = [2,4,8,19]
    num_heads = [2,3,5,10]
    emb_size = [10, 20, 50]
    depth = [1,3,5]

    for patch in patch_size :
        for head in num_heads :
            for emb in emb_size :
                for d in depth : 

                    print(f"patch : {patch} , head : {head}, emb : {emb} , depth : {d}")
                
                    G = Generator(seq_len=data.shape[-1], patch_size=patch, channels=1, num_classes=9, latent_dim=100, embed_dim=emb, depth=d,
                                num_heads=head, forward_drop_rate=0.5, attn_drop_rate=0.5)
                    D = Discriminator(in_channels=1,
                                patch_size=patch,
                                emb_size=50, 
                                seq_length = data.shape[-1],
                                depth=d, 
                                n_classes=1)

                    X_temp = train(G, D, data)

                    fig, axs = plt.subplots(3)
                    for j in range(3) :
                        axs[j].plot(np.array(X_temp[j,0,0,:]))
                    plt.savefig(f"results/example_generation_patch{patch}_head{head}_emb{emb}_depth{d}.png", dpi = 200)





    

