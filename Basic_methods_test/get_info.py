
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from imbalance_degree import imbalance_degree

def print_title(s) :
    print(f'\n\n{s:=^50}')


def get_datasets_infos():

    dict_infos = {}
    
    dataset_folder = "./datasets"
    all_files = []

    for dirpath, dirnames, filenames in os.walk(dataset_folder):
        for filename in [f for f in filenames if f.endswith("_TRAIN.tsv")]:
            all_files.append( ( os.path.join(dirpath, filename) ,filename )  )
    

    for file_path, file_name in tqdm(all_files) :
        dict_dataset = {}


        data = pd.read_csv(file_path,sep='\t', header =None)
        

        #Nombre de series :
        dict_dataset["n_samples"] = len(data.index)

        #Labels et leur compte :
        unique_labels = np.sort( data[0].unique() )
        label_dict = {}
        for label in unique_labels :
            label_dict[label] = len( data[data[0] == label].index)
        dict_dataset["labels"] = label_dict

        #Imbalance degree
        classes_count = np.array(list( label_dict.values() ))
        im_deg = imbalance_degree(classes_count, "EU")
        dict_dataset["ID"] = im_deg

        #Imbalance ratio (si possible)
        dict_dataset["IR"] = None
        if len(label_dict) == 2 :
            dict_dataset["IR"] = min(label_dict.values())/max(label_dict.values())


        dict_infos[file_name] = dict_dataset


    return {k: v for k, v in sorted(dict_infos.items(), key=lambda item: item[1]["ID"])}


def print_dataset_info(ds_info) :

    for k,v in ds_info.items() :

        print_title(" " + k + " ") 
        print("--> Nombre de sample :", v["n_samples"])

        print("--> Labels :")
        for label, cnt_label in v["labels"].items() : 
            print("    --> {} : {} samples".format( label, cnt_label ))

        if v["IR"] == None :
            print("--> Imbalance degree :", v["ID"])
        else :
            print("--> Imbalance ratio :", v["IR"])


def get_info(): 
    dataset_infos = get_datasets_infos()
    print_dataset_info(dataset_infos)

    