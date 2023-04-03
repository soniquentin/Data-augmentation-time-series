 
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from res.imbalance_degree import imbalance_degree
import csv
from numpy import genfromtxt


def get_datasets_infos():

    dict_infos = {}
    
    dataset_folder = "../datasets"
    all_files = []

    for dirpath, dirnames, filenames in os.walk(dataset_folder):
        for filename in [f for f in filenames if f.endswith("_TRAIN.tsv")]:
            all_files.append( ( os.path.join(dirpath, filename) ,filename )  )
    

    for file_path, file_name in tqdm(all_files) :
        dict_dataset = {"filepath" : file_path}


        data = pd.read_csv(file_path,sep='\t', header =None)
        

        #Nombre de series :
        dict_dataset["n_samples"] = len(data.index)

        dict_dataset["length"] = len(data.columns) - 1

        #Labels et leur compte :
        unique_labels = np.sort( data[0].unique() )
        label_dict = {}
        for label in unique_labels :
            label_dict[label] = len( data[data[0] == label].index)
        dict_dataset["labels"] = label_dict

        #Imbalance degree
        classes_count = np.array(list( label_dict.values() ))
        im_deg = imbalance_degree(classes_count, "EU")
        if im_deg == 0 : #On divise par 10, peut-être que c'est parce que les nombre sont trop grands
            classes_count //= 10
            im_deg = imbalance_degree(classes_count, "EU")
        dict_dataset["ID"] = im_deg

        #Imbalance ratio (si possible)
        dict_dataset["IR"] = None
        if len(label_dict) == 2 :
            dict_dataset["IR"] = min(label_dict.values())/max(label_dict.values())
            dict_dataset["ID"] = None


        dict_infos[file_name] = dict_dataset


    return dict_infos


def make_csv(ds_info) :

    with open('infos.csv', 'w') as f:
        
        data = np.array( pd.read_csv("res/DataSummary.csv" ,sep=',') )
        type_dict = { f'{info[2]}_TRAIN.tsv' :  info[1] for info in data}

        writer = csv.writer(f)
        writer.writerow(["Name", "Type", "N samples", "Length timeseries", "ID (multi)", "IR (binary)","Label Count", "Path" ])

        for k,v in ds_info.items() :
            label_count_sorted = {k: v for k, v in sorted(v["labels"].items(), key=lambda item: item[1])}
            writer.writerow([k, type_dict[k] ,v["n_samples"], v["length"], v["ID"], v["IR"], label_count_sorted, v["filepath"]])



if __name__ == "__main__" :
    dict_dataset = get_datasets_infos()
    make_csv(dict_dataset)