import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

window_size = 60

split_dataset = { 'BHARTIARTL.csv' : ([1300] , [1700]),
                   'TATASTEEL.csv' : ([4400, 1800] , [3300]),
                   'TITAN.csv' : ( [1850, 2600] , [2250] ) ,
                   'BRITANNIA.csv' : ( [3800, 4400] , [] ) ,
                   'MM.csv' : ( [1000, 2400, 3600] , [2200, 4800] ) ,
                   'COALINDIA.csv' : ( [] , [1300, 650] ) ,
                   'JSWSTEEL.csv' : ( [650, 1200, 2750] , [1600] ) ,
                   'NESTLEIND.csv' : ( [2200] , [] ) ,
                   'ICICIBANK.csv' : ( [1700, 3500] , [2200] ) ,
                   'MARUTI.csv' : ( [3400] , [] ) ,
                   'ULTRACEMCO.csv' : ( [2600] , [] ) ,
                   'WIPRO.csv' : ( [] , [400] ) ,
                   'NTPC.csv' : ( [700] , [1800, 3600] ) ,
                   'VEDL.csv' : ( [1900] , [] ) ,
                   'ASIANPAINT.csv' : ( [3200] , [] ) ,
                   'ONGC.csv' : ( [1400] , [] ) ,
                   'IOC.csv' : ( [1500] , [3100] ) ,
                   'DRREDDY.csv' : ( [3400] , [4400] ) ,
                   'TECHM.csv' : ( [1900] , [500] ) ,
                   'TCS.csv' : ( [2000] , [1000] ) ,
                   'SUNPHARMA.csv' : ( [2000] , [4300] ) ,
                   'HCLTECH.csv' : ( [1400,3300] , [2200] ) ,
                   'HDFC.csv' : ( [1000,2400] , [2000] ) ,
                   'KOTAKBANK.csv' : ( [1400, 3800] , [1700] ) ,
                   'ZEEL.csv' : ( [2000] , [3300] ) ,
                   'AXISBANK.csv' : ( [2400] , [2100,2900] ) ,
                   'EICHERMOT.csv' : ( [3600, 4200] , [] ) ,
                   'HDFCBANK.csv' : ( [1600,3600] , [2000] ) ,
                   'INFRATEL.csv' : ( [500, 1050] , [700,1200] ) ,
                   'TATAMOTORS.csv' : ( [1400, 2600] , [2100] ) ,
                   'HEROMOTOCO.csv' : ( [1200, 3500] , [4500] ) ,
                   'UPL.csv' : ( [2600] , [3400] ) ,
                   'CIPLA.csv' : ( [] , [600] ) ,
                   'BAJAJ-AUTO.csv' : ( [400] , [] ) ,
                   'ITC.csv' : ( [1200] , [] ) ,
                   'GAIL.csv' : ( [1400, 2100, 3500] , [1800, 2500] ) ,
                   'POWERGRID.csv' : ( [1600, 2100] , [] ) ,
                   'HINDALCO.csv' : ( [] , [600] ) ,
                   'BPCL.csv' : ( [2400] , [] ) ,
                   'LT.csv' : ( [] , [1050] ) ,
                   'HINDUNILVR.csv' : ( [3600] , [] ) ,
                   'GRASIM.csv' : ( [1500, 3400] , [2000] ) ,
                   'RELIANCE.csv' : ( [1800] , [] ) ,
                   'YESBANK.csv' : ( [2600] , [] ) ,
                   'SBIN.csv' : ( [1600, 2300] , [2000, 2700] ) ,
                   'INDUSINDBK.csv' : ( [3600] , [] ) ,
                   'BAJAJFINSV.csv' : ( [1800] , [] ) ,
                   'BAJFINANCE.csv' : ( [3600] , [] ) ,
}


data = []
label = []

#Open each files as key of split_dataset
for filename, couple in tqdm(split_dataset.items()):
    if filename.endswith(".csv") : 
        df = pd.read_csv("raw_dataset/"+filename, sep=',')
        df_list = list(df["High"])

        li_increase, li_decrease = couple
        for i in li_increase:
            data.append(df_list[i-window_size:i])
            label.append(0)
        
        for i in li_decrease:
            data.append( df_list[i-window_size:i])
            label.append(1)
    
data = np.array(data)
#normalize data min max between 0 and 1
data = (data - np.min(data, axis = 1).reshape(-1,1)) / (np.max(data, axis = 1).reshape(-1,1) - np.min(data, axis = 1).reshape(-1,1))

data = np.c_[np.array(label), data]
np.random.shuffle(data)

data_train = data[:int(0.75*len(data))]
data_test = data[int(0.75*len(data)):]


#Save data as pandas dataframe
df_train = pd.DataFrame(data_train)
df_train.to_csv("../../datasets/StockMarket/StockMarket_TRAIN.tsv", index=False, header=False,sep='\t')

df_test = pd.DataFrame(data_test)
df_test.to_csv("../../datasets/StockMarket/StockMarket_TEST.tsv", index=False, header=False, sep='\t')



