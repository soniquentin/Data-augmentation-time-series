import os
import pandas as pd
from params import *
import numpy as np


def get_charac_and_metric(delta = False) :

    """
    Récupère les caractéristiques des datasets testés et les résultats de metric à partir des dossiers/fichiers dans tests

    delta : si True, on récupère les delta_metric, sinon on récupère les metric
    """

    #Récupère les listes des caractéristiques des datasets testés
    #charac_lists de la forme : {carac1 : {dataset1 : valeur, dataset2 : valeur, ...}, carac2 : {dataset1 : valeur, dataset2 : valeur, ...}, ...}
    infos = pd.read_csv("infos.csv",sep=',')
    tested_dataset = [ folder for folder in os.listdir("tests") if os.path.isdir(f"tests/{folder}") ] #Récupère la liste de tous les datasets testés = tous les dossiers dans tests
    charac_lists = {charac : {} for charac in DATASET_CHARACTERISTICS}
    for dataset_name in tested_dataset :
        for index, row in infos.iterrows(): #Iter dans le fichier infos.csv
            if f"{dataset_name}_TRAIN.tsv" == row["Name"] :
                for c in DATASET_CHARACTERISTICS :
                    assert c in row, f"Erreur : {c} n'est pas dans infos.csv"
                    charac_lists[c][dataset_name] = row[c]
                break


    #On va récupérer les delta_metric de tous les groupes de tests
    #global_delta_metric de la forme : { Metric1 : {dataset1 : {model1 : {method_DA1 : Delta_Acc, method_DA2 : Delta_Acc, ...}, ...} ...} ...}
    model_dict = {} #les clés sont les modèles rencontrées et les valeurs sont le nombre de datasets qui les ont utilisées
    transfo_dict = {} #les clés sont les transformations rencontrées et les valeurs sont le nombre de datasets qui les ont utilisées
    global_delta_metric = {}
    for dataset_name in tested_dataset :

        first_time = True #Pour savoir si c'est le premier modèle du dataset qu'on rencontre

        for model_name in os.listdir(f"tests/{dataset_name}") : #On itère sur les modèles (qui sont les dossier dans chaque dataset)

            if os.path.isdir(f"tests/{dataset_name}/{model_name}") and model_name != "plot_TSNE" and model_name != "synthesized_data" : #Si c'est un dossier et que ce n'est pas le dossier plot_TSNE et synthesized_data

                #Incrémente le nombre de datasets qui ont utilisé ce modèle
                model_dict[model_name] = model_dict.get(model_name, 0) + 1

                #Récupère les moyennes
                mean_score = pd.read_csv(f"tests/{dataset_name}/{model_name}/Mean.csv", sep='\t', index_col=0)

                if delta :
                    #Récupère la ligne dont l'index est "Default"
                    default_line = mean_score[mean_score.index == "Default"]

                    #Soustrait la ligne "Default" à toutes les autres lignes et supprime la ligne "Default"
                    delta_metric_tmp = mean_score.sub(default_line.iloc[0], axis=1).drop("Default")
                else :
                    delta_metric_tmp = mean_score

                #Rempli global_delta_metric : on itère sur les colonnes de delta_metric_tmp (qui sont les métriques)
                first_metric = True #Pour savoir si c'est la première métrique qu'on rencontre
                for metric_name in delta_metric_tmp.columns :
                    if metric_name not in global_delta_metric :
                        global_delta_metric[metric_name] = {}
                    if dataset_name not in global_delta_metric[metric_name] :
                        global_delta_metric[metric_name][dataset_name] = {}
                    global_delta_metric[metric_name][dataset_name][model_name] = {}
                    for transfo in delta_metric_tmp.index :
                        if first_time and first_metric :
                            transfo_dict[transfo] = transfo_dict.get(transfo, 0) + 1
                        global_delta_metric[metric_name][dataset_name][model_name][transfo] = delta_metric_tmp.loc[transfo, metric_name] 

                    first_metric = False
                
                first_time = False

    #List de tous les modèles et transformations qui sont sensés avoir été testés (changer dans params.py si probleme)
    all_models = list(model_dict.keys())
    all_transfo = list(transfo_dict.keys())



    #Preprocess global_delta_metric : 
    #   - supprimer les datasets dont il manque des modèles de all_models
    #   - si pas de Ada, alors on remplace par Basic
    #   - si pas de GAN, alors supprime le dataset

    #Supprime les datasets dont il manque des modèles de all_models
    datasets_to_delete = []
    for metric_name in global_delta_metric :
        for dataset_name in global_delta_metric[metric_name].keys() :
            if len(global_delta_metric[metric_name][dataset_name]) != len(all_models) :
                if dataset_name not in datasets_to_delete :
                    datasets_to_delete.append(dataset_name)
    for dataset_name in datasets_to_delete :
        for metric_name in global_delta_metric :
            del global_delta_metric[metric_name][dataset_name]

    #Si pas de Ada, alors on remplace par Basic
    for metric_name in global_delta_metric :
        for dataset_name in global_delta_metric[metric_name].keys() :
            for model_name in global_delta_metric[metric_name][dataset_name] :
                if "Ada" not in global_delta_metric[metric_name][dataset_name][model_name] :
                    try :
                        global_delta_metric[metric_name][dataset_name][model_name]["Ada"] = global_delta_metric[metric_name][dataset_name][model_name]["Basic"]
                    except Exception as e:
                        print(f"Erreur : {model_name} n'a pas de Basic ({e})")
                        exit()
    
    #Si pas de GAN, alors supprime le dataset
    datasets_to_delete = []
    for metric_name in global_delta_metric :
        for dataset_name in global_delta_metric[metric_name].keys() :
            for model_name in global_delta_metric[metric_name][dataset_name] :
                if "GAN" not in global_delta_metric[metric_name][dataset_name][model_name] and dataset_name not in datasets_to_delete :
                    datasets_to_delete.append(dataset_name)
    for dataset_name in datasets_to_delete :
        for metric_name in global_delta_metric :
            del global_delta_metric[metric_name][dataset_name]


    #Liste des datasets finale
    effective_test_dataset = list(global_delta_metric[list(global_delta_metric.keys())[0]].keys())


    #Print les récapitulatifs
    print(f"\n================== RECAPITULATIF DES TESTS EFFECTUÉS ==================")
    print(f"Nombre de datasets : {len(tested_dataset)}")
    print(f"Répartition des modèles : {model_dict}")
    print(f"Répartition des transformations : {transfo_dict}")
    print(f"Nombre de datasets après suppression : {len(effective_test_dataset)}")
    print("==========================================================================\n")


    return global_delta_metric, charac_lists, effective_test_dataset, all_models, all_transfo
