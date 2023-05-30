"""
Trace la spearman's rank correlation à partir des fichier pickle dans tmp généré pendant make_tests.py
"""
import sys
import os
from pathlib import Path
#Rajoute le dossier Basic_methods_test/ dans sys.path pour pouvoir importer les fonctions
sys.path.append(str(Path(os.getcwd()).parent.absolute()) + "/Basic_methods_test")
import pandas as pd
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
import numpy as np



if __name__ == "__main__" :

    range_picke_file = int(ceil(len(DATASETS_TO_TEST)/group_size))

    #Récupère les listes des caractéristiques des datasets

    infos = pd.read_csv("infos.csv",sep=',')


    for metric_name in summary_metric :

        delta_metric = {}
        charac_lists = {charac : {} for charac in DATASET_CHARACTERISTICS}

        for i in range(1,range_picke_file+1) :
            f = open(f"tmp/delta_metric{i}_{metric_name}.pickle",'rb')
            new_delta = pickle.load(f)
            for dataset_name,v in new_delta.items():
                if dataset_name in DATASETS_TO_TEST : 
                    delta_metric[dataset_name] = v

                    #Retrouve les characteristiques des datasets
                    for index, row in infos.iterrows():
                        if f"{dataset_name}_TRAIN.tsv" == row["Name"] :
                            for c in DATASET_CHARACTERISTICS : 
                                charac_lists[c][dataset_name] = row[c]
                            break

            f.close()

        for k,v in delta_metric.items():
            for k_p, v_p in v.items():
                if "Ada" not in v_p : #Si Ada n'a pas été testé, on le remplace par Basic
                    v_p["Ada"] = v_p["Basic"]


        #charac_lists de la forme : {carac1 : {dataset1 : valeur, dataset2 : valeur, ...}, carac2 : {dataset1 : valeur, dataset2 : valeur, ...}, ...}
        #delta_metric de la forme : {dataset1 : {model1 : {method_DA1 : Delta_Acc, method_DA2 : Delta_Acc, ...}, ...} ...}


        #Liste des datasets qui contiennent au moins un modèle qui a été testé avec succès
        effective_test_dataset = [k for k,v in delta_metric.items() if bool(v)] #bool(v) == True si v n'est pas vide
        if effective_test_dataset == [] : 
            raise Exception("Erreur : delta_metric est vide pour tous les datasets")
        #Récupère la liste de tous les models et transformation surlesquels on a fait les tests
        #On suppose que tous les datasets ont été testés avec les mêmes models et les mêmes transformations et que le premier de la liste est parfaitement représentatif
        all_models = list(delta_metric[effective_test_dataset[0]].keys()) 
        all_transfo = list(delta_metric[effective_test_dataset[0]][all_models[0]].keys())
        print(f"all_models : {all_models}")
        print(f"all_transfo : {all_transfo}\n")


        #Construit le graphes de Spearman’s Rank Correlation Coefficients à l'aide de delta_metric
        nb_caracs = len(DATASET_CHARACTERISTICS)
        for j in range(nb_caracs):

            data_to_plot = pd.DataFrame(columns= ["Model", "Transformation", "Delta_acc"])

            c = DATASET_CHARACTERISTICS[j] #La caractéristique en cours
            list_carac_through_dataset = {dataset_name : charac_lists[c][dataset_name] for dataset_name in effective_test_dataset} #La liste des valeurs de la caractéristique pour tous les datasets


            index_number = 0 #sert à remplir data_to_plot
            for model, transfo in itertools.product(all_models, all_transfo) :
                
                #Récupère la liste des delta_metric pour le model et la transformation en cours
                list_deltametric_for_spearmanr_temp = [delta_metric[dataset_name][model][transfo] for dataset_name in effective_test_dataset if model in delta_metric[dataset_name] and transfo in delta_metric[dataset_name][model]]
                list_carac_for_spearmanr_temp = [list_carac_through_dataset[dataset_name] for dataset_name in effective_test_dataset if model in delta_metric[dataset_name] and transfo in delta_metric[dataset_name][model]]

                list_deltametric_for_spearmanr, list_carac_for_spearmanr = [], []
                #Gère les cas des Nan : on les retire des listes
                assert len(list_deltametric_for_spearmanr_temp) == len(list_carac_for_spearmanr_temp)
                for i in range(len(list_deltametric_for_spearmanr_temp)) :
                    if not (np.isnan(list_deltametric_for_spearmanr_temp[i]) or np.isnan(list_carac_for_spearmanr_temp[i])) :
                        list_deltametric_for_spearmanr.append(list_deltametric_for_spearmanr_temp[i])
                        list_carac_for_spearmanr.append(list_carac_for_spearmanr_temp[i])

                data_to_plot.loc[index_number] = [model, transfo, stats.spearmanr(list_deltametric_for_spearmanr,list_carac_for_spearmanr).statistic]
                index_number += 1
                
            g = sns.catplot(
                data=data_to_plot, kind="bar",
                x="Model", y="Delta_acc", hue="Transformation",
                errorbar="sd", palette="dark", alpha=.6, height=6,
                legend_out = True
            )

            g.despine(left=True)
            g.set_axis_labels("Classifier model", f"Spearman's Rank Correlation")
            g.legend.set_title(f"{c} ({metric_name})")

        
            plt.savefig(f"tests/impact_{c}_{metric_name}.png", dpi = 200) 
        
        plt.close("all")