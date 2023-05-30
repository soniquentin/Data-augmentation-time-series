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
from PIL import Image


def simulateSB(N, M, Data, deltati, H, nbStepsPerDeltati, numberOfSamples):
    
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
    
    assert Data.shape == (M,N+1)
    assert H > 0
    
    # Create C++ object
    simulator = SBTimeSeries.SchrodingerBridge(N, M, Data)

    # Launch SB diffusion
    simu = list()
    for i in tqdm(range(numberOfSamples)):
        simu.append(simulator.SimulateKernel(nbStepsPerDeltati, H, deltati))
    simu = np.array(simu)
    
    return simu


# Import data
trainX = pd.read_csv("../../datasets/OSULeaf/OSULeaf_TRAIN.tsv", sep='\t', header=None)
trainX = np.array( trainX[trainX[0] == 5] )[:,1:]
maxi, mini = np.max(trainX), np.min(trainX) 
trainX = (trainX - mini)/(10*(maxi-mini)) #Normalize min max between 0 and 1

M = trainX.shape[0]
N = trainX.shape[1]
H = 1
nbStepsPerDeltati = 100 
deltati = 1./252.

print("N = ", N)
print("M = ", M)
print("H = ", H)
print("nbStepsPerDeltati = ", nbStepsPerDeltati)
print("deltati = ", deltati)


TimeSeries = np.zeros((M, N+1))
TimeSeries[:,1:] = trainX[:M,:]


numberOfSamples = 16


simulator = SBTimeSeries.SchrodingerBridge(N, M, TimeSeries)
simu = []
for i in tqdm(range(numberOfSamples)):
    simu.append(simulator.SimulateKernel(nbStepsPerDeltati, H, deltati))
simu = np.array(simu)



fig, ax = plt.subplots(4, 4, figsize=(8,8))

for i in range(4):
    for j in range(4):
        #Denormalize
        try : 
            ax[i,j].plot(trainX[i*4+j,1:] * 10*(maxi-mini) + mini)
        except Exception :
            pass



plt.savefig('plot/OSULeaf_TRAIN_TRAIN.png', dpi=300, bbox_inches='tight')

fig, ax = plt.subplots(4, 4, figsize=(8,8))

for i in range(4):
    for j in range(4):
        #Denormalize
        ax[i,j].plot(simu[i*4+j,1:] * 10*(maxi-mini) + mini)


plt.savefig('plot/OSULeaf_TRAIN_gen+.png', dpi=300, bbox_inches='tight')