import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pyperclip


#Open on all files in the directory "raw_dataset"
for filename in os.listdir("raw_dataset"):
    if filename.endswith(".csv") : 

        pyperclip.copy(f"'{filename}' : ( [] , [] ) ,")
        df = pd.read_csv("raw_dataset/"+filename, sep=',')


        plt.figure(figsize=(15,8))
        plt.xticks(np.arange(0,len(np.array(df["High"])),200), np.arange(0,len(np.array(df["High"])),200))
        plt.plot(np.array(df["High"]))
        plt.show()
        plt.close()