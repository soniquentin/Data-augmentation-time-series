"""
Trace la spearman's rank correlation à partir des fichier pickle dans tmp généré pendant make_tests.py
"""
import sys
import os
from pathlib import Path
#Rajoute le dossier Basic_methods_test/ dans sys.path pour pouvoir importer les fonctions
sys.path.append(str(Path(os.getcwd()).parent.absolute()) + "/Basic_methods_test")
import pandas as pd
from train import *
import seaborn as sns
sns.set_theme(style="whitegrid")
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import itertools
import matplotlib.pyplot as plt
import pickle
from params import *
from math import ceil



if __name__ == "__main__" :
    range_picke_file = int(ceil(len(DATASETS_TO_TEST)/group_size))


    f = open(f"tmp/charac_lists1.pickle",'rb')
    charac_lists = pickle.load(f)
    f.close()

    f = open(f"tmp/delta_metric1.pickle",'rb')
    delta_metric = pickle.load(f)
    f.close()

    for i in range(2,range_picke_file+1) :
        f = open(f"tmp/charac_lists{i}.pickle",'rb')
        new_charac = pickle.load(f)
        for c in charac_lists :
            for k,v in new_charac[c].items():
                charac_lists[c][k] = v
        f.close()

        f = open(f"tmp/delta_metric{i}.pickle",'rb')
        new_delta = pickle.load(f)
        for k,v in new_delta.items():
            delta_metric[k] = v
        f.close()

    for k,v in delta_metric.items():
        for k_p, v_p in v.items():
            if "Ada" not in v_p : 
                v_p["Ada"] = v_p["Basic"]




    #Construit le graphes de Spearman’s Rank Correlation Coefficients à l'aide de delta_metric
    nb_caracs = len(DATASET_CHARACTERISTICS)
    for j in range(nb_caracs):

        data_to_plot = pd.DataFrame(columns= ["Model", "Transformation", "Delta_acc"])

        c = DATASET_CHARACTERISTICS[j]
        list_carac_through_dataset = list(charac_lists[c].values()) #La liste des valeurs de tous les datasets pour cette caractéristiques

        #Récupère la liste de tous les models surlesquels on a fait les tests ainsi que toutes les transformations
        all_models = list(delta_metric[DATASETS_TO_TEST[0]].keys())
        all_transfo = list(delta_metric[DATASETS_TO_TEST[0]][all_models[0]].keys())

        index_number = 0
        for model, transfo in itertools.product(all_models, all_transfo) :
            list_deltametric_through_dataset = [dataset_delta_metric[model][transfo] for dataset_delta_metric in delta_metric.values()]
            data_to_plot.loc[index_number] = [model, transfo, stats.spearmanr(list_deltametric_through_dataset,list_carac_through_dataset).statistic]
            index_number += 1
            
        g = sns.catplot(
            data=data_to_plot, kind="bar",
            x="Model", y="Delta_acc", hue="Transformation",
            errorbar="sd", palette="dark", alpha=.6, height=6,
            legend_out = True
        )

        g.despine(left=True)
        g.set_axis_labels("Classifier model", f"Spearman's Rank Correlation")
        g.legend.set_title(f"{c} ({summary_metric})")

    
        plt.savefig(f"tests/impact_{c}.png", dpi = 200) 
    
    plt.close("all")
