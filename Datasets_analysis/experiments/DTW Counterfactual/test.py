from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import random
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import pandas as pd


x = np.arange(0, 20, .5)
s1 = np.sin(x)
s2 = np.sin(x - 1)
random.seed(1)

for idx in range(len(s2)):
    if random.random() < 0.05:
        s2[idx] += (random.random() - 0.5) / 2



d, paths = dtw.warping_paths(s1, s2, window=50, psi=2)
best_path = dtw.best_path(paths)

print(dtw.distance(s1, s2))

print(best_path)
print("")
best_path_accumulated_cost = [paths[i+1,j+1] for (i,j) in best_path]
print("")
cost_path = [ (s1[i] - s2[j])**2 for (i,j) in best_path]
try_accumulated_cost = np.sqrt(np.cumsum(cost_path))

for a,b in zip(best_path_accumulated_cost, try_accumulated_cost):
    print(a,b)


dtwvis.plot_warpingpaths(s1, s2, paths, best_path)
plt.show()
