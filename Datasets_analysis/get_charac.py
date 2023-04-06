 
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from res.imbalance_degree import imbalance_degree
import csv
from numpy import genfromtxt


def calc_variances_normalized(data, file_name) -> (float, float) :
    """
    Au lieu d'utiliser une variance, on va utiliser un coefficient de variation i.e on divise les écart-types par les moyennes
    """

    def aux(data_aux) :     
        new_data = np.array(data)[:,1:]
        upper_quantile = np.nanquantile(new_data, 0.9)
        lower_quantile = np.nanquantile(new_data, 0.1)

        new_data = (data - lower_quantile)/(upper_quantile - lower_quantile)

        return new_data.var().mean()

    #Labels et leur compte :
    stdev_acc = 0

    nb_samples = len(data.index)

    unique_labels = np.sort( data[0].unique() )
    for label in unique_labels :
        count = len( data[data[0] == label].index)
        label_var = aux( data[data[0] == label] )

        stdev_acc += (count/nb_samples)*label_var

    return aux(data), stdev_acc/len(unique_labels)



def calc_length(data,file_name) -> int :
    return (len(data.columns) - 1)


def calc_nb(data,file_name) -> (int,int,dict) :
    #Labels et leur compte :
    unique_labels = np.sort( data[0].unique() )
    label_dict = {}
    for label in unique_labels :
        label_dict[label] = len( data[data[0] == label].index)

    return len(data.index), sum(label_dict.values()) / len(label_dict) , {k: v for k, v in sorted(label_dict.items(), key=lambda item: item[1])}

def calc_IR_ID(data,file_name) -> (float,float) :

    unique_labels = np.sort( data[0].unique() )
    classes_count = np.array( [len( data[data[0] == label].index) for label in unique_labels] )
    im_deg = imbalance_degree(classes_count, "EU")
    if im_deg == 0 : #On divise par 10, peut-être que c'est parce que les nombre sont trop grands
        im_deg = imbalance_degree(classes_count//10, "EU")

    #Imbalance ratio (si possible)
    im_rat = None
    if len(unique_labels) == 2 :
        im_rat = np.min(classes_count)/np.max(classes_count)
        im_deg = None
    
    return im_deg, im_rat


def calc_smoothness_each_label(data,file_name) -> dict :
    unique_labels = np.sort( data[0].unique() )
    label_dict = {}
    for label in unique_labels :
        data_label = np.array( data[data[0] == label] )[:,1:]
        individual_smoothness = np.sum( np.diff( np.diff(data_label) )**2, axis = 1)
        label_dict[label] = individual_smoothness.mean()

    return label_dict


def get_datasets_infos(caract_list):

    infos = []
    
    dataset_folder = "../datasets"
    all_files = []

    for dirpath, dirnames, filenames in os.walk(dataset_folder):
        for filename in [f for f in filenames if f.endswith("_TRAIN.tsv")]:
            all_files.append( ( os.path.join(dirpath, filename) ,filename )  )
    

    for file_path, file_name in tqdm(all_files) :
        #NAME AND TYPE
        dict_dataset = {"Name" : file_name}

        summary_data = np.array( pd.read_csv("res/DataSummary.csv" ,sep=',') )
        type_dict = { f'{info[2]}_TRAIN.tsv' :  info[1] for info in summary_data}
        dict_dataset["Type"] = type_dict[file_name]


        data = pd.read_csv(file_path,sep='\t', header =None)
        #OTHER CHARACTERISTICS
        for function, v in caract_list.items() :
            carac_values = function(data, file_name)

            try :
                if "dict" not in str(type(carac_values)) :
                    carac_values = list(carac_values)
                    v = list(v)
                else :
                    raise TypeError
            except TypeError :
                carac_values = [carac_values]
                v = [v]

            for carac_name, carac_value  in zip(v, carac_values) :
                dict_dataset[carac_name] = carac_value

        #FILEPATH
        dict_dataset["Filepath"] = file_path
        infos.append(dict_dataset)


    return infos


def make_csv(info) :

    with open('infos.csv', 'w') as f:
        
        data = np.array( pd.read_csv("res/DataSummary.csv" ,sep=',') )
        type_dict = { f'{info[2]}_TRAIN.tsv' :  info[1] for info in data}

        writer = csv.writer(f)

        writer.writerow(list(info[0].keys()))

        for info_dataset in info :

            writer.writerow(list(info_dataset.values()))


if __name__ == "__main__" :

    CARAC_DICT = {calc_length : "Length",
                  calc_nb : ("Dataset size", "Avg label size", "Details size label"),
                  calc_variances_normalized : ("Dataset variance", "Intra-class variance"),
                  calc_IR_ID : ("ID", "IR"),
                  calc_smoothness_each_label : "Smoothness"
                  }


    info = get_datasets_infos(CARAC_DICT)

    make_csv(info)