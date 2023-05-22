import sys
sys.path.append("../build")
import SBTimeSeries

import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from datetime import date
from scipy.stats import ks_2samp, kstest, kurtosis
import yfinance as yf
yf.pdr_override()
from fbm import FBM
from tensorflow.keras.datasets import mnist
from pathlib import Path
import os
import warnings
warnings.filterwarnings("ignore")


# load dataset
(trainX, _), (_, _) = mnist.load_data()



"""
@params:
- N: integer, number of time steps
- M: integer, number of samples
- Data: list of ndarray (M x (N+1)), true time series samples
- deltai: float, deltai = t_{i+1} - t_i
- H: float (>0), bandwidth of Quartic kernel
- nbStepsPerDeltati: integer, number of time steps
- numberOfSamples, integer, number of samples to generate

Typical utilization :
simulator = SBTimeSeries.SchrodingerBridge(N, M, NbPixels**2,TimeSeriesFlat)       with TimeSeriesFlat.shape = (M, N+1, NbPixels**2)
simulator.SimulateKernel(nbStepsPerDeltati, H, deltati)
"""


M = 10000  
N = 16 
NbPixels = trainX.shape[1]  
deltati = 1./252.

TimeSeries = np.zeros((M, N+1, NbPixels, NbPixels))
TimeSeries[:,1,:,:] = trainX[:M,:,:]/255/10 - 0.05 # Normalization to avoid numerical issues


TimeSeriesFlat = TimeSeries.reshape(M, N+1, -1)
numberOfSamples = 16


simulator = SBTimeSeries.SchrodingerBridge(N, M, NbPixels**2, TimeSeriesFlat)
simuTimeSeriesFlat = np.empty((numberOfSamples, N+1, NbPixels**2))
for i in tqdm(range(numberOfSamples)):
    images = np.array(simulator.SimulateKernelVectorized(50, 1.1, 1./252.))
    simuTimeSeriesFlat[i] = images

simuTimeSeries = simuTimeSeriesFlat.reshape(numberOfSamples, N+1, NbPixels, NbPixels)



fig, ax = plt.subplots(4, 4, figsize=(8,8))

for i in range(4):
    for j in range(4):
        ax[i,j].imshow((TimeSeries[i*4 + j,1,:,:]+0.05)*10*255, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])



plt.savefig('plot/mnist.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(4, 4, figsize=(8,8))

for i in range(4):
    for j in range(4):
        ax[i,j].imshow((simuTimeSeries[i*4+j,1,:,:]+0.05)*10*255, cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])


plt.savefig('plot/mnist_simu.png', dpi=300, bbox_inches='tight')