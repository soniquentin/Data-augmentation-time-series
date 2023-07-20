import pandas as pd
import sys
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram
import argparse


ALL_DATASETS = [
"ECGFiveDays",
"SonyAIBORobotSurface1",
"Wafer",
"Earthquakes",
"ProximalPhalanxOutlineCorrect",
"ECG200",
"Lightning2",
"PhalangesOutlinesCorrect",
"Strawberry",
"MiddlePhalanxOutlineCorrect",
"HandOutlines",
"DistalPhalanxOutlineCorrect",
"Herring",
"Wine",
"WormsTwoClass",
"Yoga",
"GunPointOldVersusYoung",
"GunPointMaleVersusFemale",
"FordA",
"FordB",
"Computers",
"HouseTwenty",
"TwoLeadECG",
"BeetleFly",
"BirdChicken",
"GunPointAgeSpan",
"ToeSegmentation1",
"GunPoint",
"ToeSegmentation2",
"PowerCons",
"ItalyPowerDemand",
"DodgerLoopWeekend",
"DodgerLoopGame",
"MoteStrain",
"FreezerSmallTrain",
"DodgerLoopWeekend",
"DodgerLoopGame",
"SonyAIBORobotSurface2",
"FreezerRegularTrain",
"ShapeletSim",
"Ham",
"Coffee",
"SemgHandGenderCh2",
"Chinatown"
                    ]


def import_info() :
    return pd.read_csv("infos.csv" , sep=',')


def m_section(dataset_name, nb_plot = 1):

    dataset_folder = f"tests/{dataset_name}"
    if not os.path.exists(dataset_folder) :
        os.makedirs(dataset_folder)

    #Importe les données
    info = import_info()

    #Cherche le chemin du dataset dans le csv info
    file_path = None
    for index, row in info.iterrows():
        if dataset_name in row["Name"] :
            file_path = row["Filepath"]
            break
    if file_path == None :
        raise Exception(f"Dataset '{dataset_name}' not found")


    data = pd.read_csv(file_path, sep='\t', header =None)

    data_final = pd.DataFrame()
    data_final["Label"] = data[0]
    timeseries_data_only = np.array(data)[:,1:]

    ##SMOOTHNESS
    smoothness = np.std( np.diff(timeseries_data_only) , axis = 1)
    data_final["Smoothness"] = smoothness
    plt.title(f"Smoothness in {dataset_name}")
    chart = sns.violinplot(y=data_final["Smoothness"], x = data_final["Label"])
    plt.savefig(f"{dataset_folder}/smoothness.png", dpi = 300)
    plt.close()


    unique_labels = data[0].unique()
    
    #Créer un dictionnaire { label : (count, color) } trié par ordre croissant de label
    count_dict_label = { label : len(data[data[0] == label].index) for label in unique_labels}
    count_dict_label = {k: v for k, v in sorted(count_dict_label.items(), key=lambda item: item[1])}
    i = 0
    for k,v in count_dict_label.items():
        count_dict_label[k] = (v, (i/( len(count_dict_label) - 1) , 0,i/( len(count_dict_label) - 1)) )
        i+= 1

    
    #On trace un example de chaque label
    for label,v in count_dict_label.items():
        count, color = v
        label_samples = np.array( data[data[0] == label] )[:,1:]
        for k in range(nb_plot):
            random_index = rd.randint(0,len(label_samples) - 1)


            #AUTOCORRELATION
            if args.plot_acf == "Y" :
                autocorrelation = acf(label_samples[random_index], nlags = len(label_samples[random_index]), fft = False)
                dat = pd.DataFrame(columns = ["num", "value"])
                for i in range(len(autocorrelation)) :
                    dat.loc[i] = [i, autocorrelation[i]]
                dat['num'] = dat['num'].astype('int')

                fig, ax = plt.subplots(figsize=(10,4))
                chart = sns.barplot(x="num", y="value", data=dat, ax = ax)
                chart.set(xlabel=None)
                chart.set_xticklabels(chart.get_xticklabels(), rotation=30)
                #ax2 = ax.twinx()
                #ax2.plot(label_samples[random_index], color = color)
                #plt.xticks([])
                plt.title(f"ACF of {dataset_name} n°{random_index} ({label})")
                plt.xticks(np.arange(0,len(label_samples[random_index]),20), np.arange(0,len(label_samples[random_index]),20))

                plt.savefig(f"{dataset_folder}/label_{label}_{k}_acf.png", dpi = 300)
                plt.close()

            #Plot example
            if args.plot_example == "Y" :
                fig, ax = plt.subplots(figsize=(10,4))
                plt.title(f"Example of {dataset_name} n°{random_index} ({label})")
                plt.plot(label_samples[random_index], color = color)
                plt.xticks(np.arange(0,len(label_samples[random_index]),20), np.arange(0,len(label_samples[random_index]),20))
                plt.savefig(f"{dataset_folder}/label_{label}_{k}.png", dpi = 300)
                plt.close()

            #PERIODOGRAM
            if args.plot_periodogram == "Y" :
                plt.figure(figsize=(10,4))
                x,y = periodogram(label_samples[random_index])
                plt.plot(x,y, color = color)
                plt.title(f"Periodogram of {dataset_name} n°{random_index} ({label})")
                plt.xticks([ round(0.05*i,2) for i in range(11)], [round(0.05*i,2) for i in range(11)])
                plt.savefig(f"{dataset_folder}/label_{label}_{k}_periodogram.png", dpi = 300)
                plt.close()



def d_section(dataset_name):
    #Importe les données
    info = import_info()

    #Cherche le chemin du dataset dans le csv info
    file_path = None
    for index, row in info.iterrows():
        if dataset_name in row["Name"] :
            file_path = row["Filepath"]
            break
    if file_path == None :
        raise Exception(f"Dataset '{dataset_name}' not found")


    data = np.array( pd.read_csv(file_path, sep='\t', header =None) )
    length = data.shape[-1] - 1
    df = pd.DataFrame()
    for i in range(1, length + 1) :
        step_df = pd.DataFrame()
        step_df["label"] = data[:,0]
        step_df["step"] = i
        step_df["value"] = data[:,i]

        df = pd.concat( [df, step_df], axis = 0)
    
    sns.lineplot(data = df, x="step", y="value", hue = "label")

    dataset_folder = f"tests/{dataset_name}"
    if not os.path.exists(dataset_folder) :
        os.makedirs(dataset_folder)
    plt.savefig(f"{dataset_folder}/plot_domains.png", dpi = 500)
    plt.close()




    

if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_example", help="Y, N", default="Y")
    parser.add_argument("--plot_smoothness", help="Y, N", default="Y")
    parser.add_argument("--plot_acf", help="Y, N", default="Y")
    parser.add_argument("--plot_mean", help="Y, N", default="Y")
    parser.add_argument("--plot_periodogram", help="Y, N", default="Y")
    parser.add_argument("--dataset", help="[DATASET NAME], all", default="all")
    parser.add_argument("--nb_plot", help="<int>", default="1")
    args = parser.parse_args()

    dataset_name = args.dataset
    nb_plot = int(args.nb_plot)

    if args.mean == "Y" :
        if dataset_name == 'all' :
            for dataset in tqdm(ALL_DATASETS) : 
                d_section(dataset)
        else :
            d_section(dataset_name)
        
    
    if dataset_name == 'all' :
        for dataset in tqdm(ALL_DATASETS) : 
            m_section(dataset, nb_plot, args)
    else :
        m_section(dataset_name, nb_plot, args)

