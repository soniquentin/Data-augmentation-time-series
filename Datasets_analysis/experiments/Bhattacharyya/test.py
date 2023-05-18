import numpy as np
from sklearn.decomposition import PCA
import random as rd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# Generate two example time series classes with some overlap
def bhattacharyya_distance(class1, class2):
    """
        Compute the Bhattacharyya distance between two classes of time series

        Parameters
        ----------
        class1 : array-like, shape=(n_samples, n_features)
            The first class of time series
        class2 : array-like, shape=(n_samples, n_features)
            The second class of time series

        Returns
        -------
        B : float
            The Bhattacharyya distance between the two classes
    """

    # Compute the PCA of each class over time
    pca = PCA(n_components=10)
    pca.fit(np.concatenate((class1, class2)))
    pc1 = pca.transform(class1)
    pc2 = pca.transform(class2)

    # Compute the mean and covariance of each class over the first 5 principal components

    mean1 = np.mean(pc1, axis=0)
    cov1 = np.cov(pc1.T)
    mean2 = np.mean(pc2, axis=0)
    cov2 = np.cov(pc2.T)


    """
    mean1 = np.mean(pc1, axis=0)
    cov1 = np.diag(np.var(pc1, axis=0))
    mean2 = np.mean(pc2, axis=0)
    cov2 = np.diag(np.var(pc2, axis=0))
    """

    # Compute the Bhattacharyya distance
    exp_part = np.exp( - 1/8 * (mean2 - mean1).reshape(-1,1).T @ np.linalg.inv((cov1+cov2)/2) @ (mean2 - mean1).reshape(-1,1) )
    sqrt_part = np.sqrt( np.sqrt(np.linalg.det(cov1)*np.linalg.det(cov2) )  /np.linalg.det(0.5*(cov1+cov2)) )

    B = exp_part*sqrt_part

    # Compute the Bhattacharyya distance
    return B


def generator1(length, size, ir, level) :

    """
        Label 1 : sin(x)
        Label 2 : sin(x) + exp(x)
    """


    #LABEL 1
    def label1(amp, phi, omega):
        X = np.array([i for i in range(length)])
        Y = amp*np.sin(omega*X + phi)
        return Y
    
    #LABEL 2
    def label2(amp, phi, omega):
        X = np.array([i for i in range(length)])
        Y = (1 - level)*amp*np.sin(omega*X + phi) + (level)*( np.exp(X*np.log(2*amp + 1)/length) - 1 - amp)
        return Y

    data = []

    for i in range( int(size/(1+ir)) ):
        data.append(label1(amp = rd.random() + 1, phi = 2*np.pi*rd.random() - np.pi, omega = 0.15*rd.random() + 0.05 ))

    data2 = []
    for i in range(size - int(size/(1+ir))) :
        data2.append(label2(amp = rd.random() + 1, phi = 2*np.pi*rd.random() - np.pi, omega = 0.15*rd.random() + 0.05 ))

    return np.array(data), np.array(data2)



def generator1(length, size, ir, level) :

    """
        Label 1 : sin(x)
        Label 2 : sin(x) + exp(x)
    """


    #LABEL 1
    def label1(amp, phi, omega):
        X = np.array([i for i in range(length)])
        Y = amp*np.sin(omega*X + phi)
        return Y
    
    #LABEL 2
    def label2(amp, phi, omega):
        X = np.array([i for i in range(length)])
        Y = (1 - level)*amp*np.sin(omega*X + phi) + (level)*( np.exp(X*np.log(2*amp + 1)/length) - 1 - amp)
        return Y

    data = []

    for i in range( int(size/(1+ir)) ):
        data.append(label1(amp = rd.random() + 1, phi = 2*np.pi*rd.random() - np.pi, omega = 0.15*rd.random() + 0.05 ))

    data2 = []
    for i in range(size - int(size/(1+ir))) :
        data2.append(label2(amp = rd.random() + 1, phi = 2*np.pi*rd.random() - np.pi, omega = 0.15*rd.random() + 0.05 ))

    return np.array(data), np.array(data2)



def generator2(length, size, ir, level) :

    """
        Label 1 : sin(x)
        Label 2 : amp(level,x)*sin(x)

    """

    #LABEL 1
    def label1(amp, phi, omega):
        X = np.array([i for i in range(length)])
        Y = amp*np.sin(omega*X + phi)
        return Y
    
    #LABEL 2
    def label2(amp, phi, omega):
        amp_percent = np.cumprod((1 - level*(3/length - np.random.normal(scale=rd.random()*0.01, size=length)) ), dtype=float)
        X = np.array([i for i in range(length)])
        Y = amp*amp_percent*np.sin(omega*X + phi)
        return Y

    data = []

    for i in range( int(size/(1+ir)) ):
        data.append(label1(amp = rd.random() + 1, phi = 2*np.pi*rd.random() - np.pi, omega = 0.15*rd.random() + 0.05 ))

    data2 = []
    for i in range(size - int(size/(1+ir))) :
        data2.append(label2(amp = rd.random() + 1, phi = 2*np.pi*rd.random() - np.pi, omega = 0.15*rd.random() + 0.05 ))

    return np.array(data), np.array(data2)



def generator3(length, size, ir, level) :

    """
        Label 1 : sin(x) * indicatrice(x > 100)
    """


    #LABEL 1
    def label1(amp, phi, omega):
        X = np.array([i for i in range(length)])
        Y = amp*np.sin(omega*X + phi)
        indicatrice = X > 100
        return Y*indicatrice
    
    #LABEL 2
    def label2(amp, phi, omega):
        X = np.array([i for i in range(length)])
        Y = amp*np.sin(omega*X + phi)
        indicatrice = ( X < length - int(level*100) ) & ( X > int((1-level)*100))
        return Y*indicatrice

    data = []

    for i in range( int(size/(1+ir)) ):
        data.append(label1(amp = rd.random() + 1, phi = 2*np.pi*rd.random() - np.pi, omega = 0.15*rd.random() + 0.05 ))

    data2 = []
    for i in range(size - int(size/(1+ir))) :
        data2.append(label2(amp = rd.random() + 1, phi = 2*np.pi*rd.random() - np.pi, omega = 0.15*rd.random() + 0.05 ))

    return np.array(data), np.array(data2)



def generator4(length, size, ir, level) :

    """
        Label 1 : sin(x)
        Label 2 : sin(x) + level*3
    """


    #LABEL 1
    def label1(amp, phi, omega):
        X = np.array([i for i in range(length)])
        Y = amp*np.sin(omega*X + phi)
        return Y
    
    #LABEL 2
    def label2(amp, phi, omega):
        X = np.array([i for i in range(length)])
        Y = amp*np.sin(omega*X + phi) + level*3
        return Y

    data = []

    for i in range( int(size/(1+ir)) ):
        data.append(label1(amp = rd.random() + 1, phi = 2*np.pi*rd.random() - np.pi, omega = 0.15*rd.random() + 0.05 ))

    data2 = []
    for i in range(size - int(size/(1+ir))) :
        data2.append(label2(amp = rd.random() + 1, phi = 2*np.pi*rd.random() - np.pi, omega = 0.15*rd.random() + 0.05 ))

    return np.array(data), np.array(data2)




if __name__ == "__main__":

    generator = generator4

    L_max = 100

    L = np.linspace(0,1, L_max)
    Y = []
    count = 0

    for l in tqdm(L) :
        # Generate two example time series classes with some overlap
        class1, class2 = generator(500, 300, 0.5, l)

        pca = PCA(n_components=2)
        pca.fit(np.concatenate((class1, class2)))
        pc1 = pca.transform(class1)
        pc2 = pca.transform(class2)
        #scatterplot with seaborn
        sns.set_theme()
        sns.scatterplot(x=pc1[:,0], y=pc1[:,1], color="blue")
        sns.scatterplot(x=pc2[:,0], y=pc2[:,1], color="red")
        plt.savefig(f"PCA_step{count}.png")
        plt.close()

        sns.set_theme(style="whitegrid", palette="pastel")
        if count%(L_max//10) == 0 or count == L_max-1 :
            nb_to_plot = 3

            fig, axs = plt.subplots(nb_to_plot, sharey=True)
            for j in range(nb_to_plot) :
                axs[j].plot(np.array(class1[j,:]), color = "blue")
            plt.savefig(f"class1_step{count}.png")
            plt.close()

            fig, axs = plt.subplots(nb_to_plot, sharey=True)
            for j in range(nb_to_plot) :
                axs[j].plot(np.array(class2[j,:]) , color = "red")
            plt.savefig(f"class2_step{count}.png")
            plt.close()
        count += 1



        # Compute the Bhattacharyya distance
        B = bhattacharyya_distance(class1, class2)[0][0]

        Y.append(B)

    plt.figure( figsize = (10,5) )
    plt.plot(L,Y)
    plt.savefig("Bhattacharyya_distance.png")
    plt.close()