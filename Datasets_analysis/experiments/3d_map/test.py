import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import pandas as pd
import random as rd

x = np.linspace(0, 4, 60)
y = np.array([rd.random()*5 + 1 for i in range(60)])

z = np.sin(x)


def three_d_plot(x,y,z, nb_points = 100, levels = 20) :

    vmin = np.min(z)
    vmax = np.max(z)

    # Interpolate
    f = interp2d(x, y, z, kind='quintic')

    # New grid
    xnew = np.linspace(np.min(x), np.max(x), nb_points)
    ynew = np.linspace(np.min(y), np.max(y), nb_points)
    znew = f(xnew,ynew)
    xnew, ynew = np.meshgrid(xnew, ynew)

    znew[znew < vmin] = vmin
    znew[znew > vmax] = vmax

    plt.contourf(xnew, ynew, znew, levels=np.linspace(vmin, vmax, levels), extend='max')
    plt.scatter(x, y, c=z)
    plt.colorbar()
    plt.show()


three_d_plot(x,y,z)