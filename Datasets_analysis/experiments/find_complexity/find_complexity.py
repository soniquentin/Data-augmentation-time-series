import itertools
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

A = [(136,23,30),
(70,20,8),
(152,1000,1951),
(512,322,4361),
(80,600,65),
(96,100,51),
(637,60,2161),
(80,1800,352),
(235,613,1383),
(80,600,52),
(80,600,56),
(512,64,530),
(234,57,16),
(900,181,1328),
(426,300,2054),
(150,136,131),
#(2000,40, 64604),
(82,23,4),
(512,20,1717),
(512,20,630),
(288,20,380),
(84,20,5),
(301,28,210),
(288,20,480),
(288,20,320),
(65,27,9),
(301,150,540),
(500,20,1558),
(431,109,409),
(286,28,17),
#(500,3601,60358),
#(500,3636,55000),
(720,250,8600)
]


"""
alpha = [i for i in range(-1,20)]
beta = [i for i in range(-1,20)]

for a,b in itertools.product(alpha, beta) :
    if a == -1 : 
        if b == -1 :
            X = np.array( [np.log(i)*np.log(j) for i,j,k in A] ).reshape((-1, 1))
        else :
            X = np.array( [np.log(i)*(j**b) for i,j,k in A] ).reshape((-1, 1))
    elif b == -1 :
        X = np.array( [(i**a)*np.log(j) for i,j,k in A] ).reshape((-1, 1))
    else :
        X = np.array( [(i**a)*(j**b) for i,j,k in A] ).reshape((-1, 1))

    y = np.array( [np.log(k) for i,j,k in A] )
    reg = LinearRegression().fit(X, y)
    s = reg.score(X, y) 
    if s > 0.5 :
        print(f"Score : {s} ({a},{b})")
"""



"""
A3 = [(i+j,np.log(k)) for i,j,k in A]
A3.sort(key=lambda x:x[0])
X3 = np.array([i for i,j in A3])
y3 = np.array([j for i,j in A3])


plt.scatter(X3,y3)
plt.show()
"""


data_to_plot = pd.DataFrame(columns= ["Length", "Dataset Size", "Log(Time)"])
index_number = 0
for i,j,k in A :
    data_to_plot.loc[index_number] = [i,j,np.log(k)]
    index_number += 1

plt.figure(figsize = (8,6))
sns.scatterplot(data=data_to_plot, x="Length", y="Dataset Size", hue = "Log(Time)")
plt.savefig("final.png", dpi = 200)
