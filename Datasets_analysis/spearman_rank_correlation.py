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
from params import *
from math import ceil
import numpy as np
import matplotlib as mpl
from tqdm import tqdm



if __name__ == "__main__" :

    #Créer le dossier pour stocker les impacts
    if not os.path.exists("impact") :
        os.makedirs("impact")

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

                #Récupère la ligne dont l'index est "Default"
                default_line = mean_score[mean_score.index == "Default"]

                #Soustrait la ligne "Default" à toutes les autres lignes et supprime la ligne "Default"
                delta_metric_tmp = mean_score.sub(default_line.iloc[0], axis=1).drop("Default")

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


    #Print les récapitulatifs
    print(f"\n================== RECAPITULATIF DES TESTS EFFECTUÉS ==================")
    print(f"Nombre de datasets : {len(tested_dataset)}")
    print(f"Répartition des modèles : {model_dict}")
    print(f"Répartition des transformations : {transfo_dict}")
    print(f"Nombre de datasets après suppression : {len(global_delta_metric[list(global_delta_metric.keys())[0]])}")
    print("==========================================================================")


    #Liste des datasets finale
    effective_test_dataset = list(global_delta_metric[list(global_delta_metric.keys())[0]].keys())


    for metric_name in summary_metric :

        print(f"\n\n================== IMPACT SUR {metric_name} ==================")

        delta_metric = global_delta_metric[metric_name]

        #Crée le dossier pour la métrique
        if not os.path.exists(f"impact/{metric_name}") :
            os.makedirs(f"impact/{metric_name}")

        #### ===================================== ####
        #### ===================================== ####
        #### ==== SPEARMAN'S RANK CORRELATION ==== ####
        #### ========== + CLOUD POINTS =========== ####
        #### ===================================== ####
        #### ===================================== ####

        #Construit le graphes de Spearman’s Rank Correlation Coefficients à l'aide de delta_metric
        nb_caracs = len(DATASET_CHARACTERISTICS)
        for j in tqdm( range(nb_caracs) , desc="Spearman's rank correlation" ):

            data_to_plot = pd.DataFrame(columns= ["Model", "Transformation", "Delta_acc"])

            c = DATASET_CHARACTERISTICS[j] #La caractéristique en cours
            list_carac_through_dataset = {dataset_name : charac_lists[c][dataset_name] for dataset_name in effective_test_dataset} #La liste des valeurs de la caractéristique pour tous les datasets

            #Créer le dossier pour la caractéristique
            if not os.path.exists(f"impact/{metric_name}/{c}") :
                os.makedirs(f"impact/{metric_name}/{c}")


            #Fig for the cloud plot
            fig, axs = plt.subplots(len(all_models), len(all_transfo), figsize=(20, 12))
            fig.suptitle(f"Cloud plot : log({c}+1) VS Delta {metric_name}", fontsize=20)

            index_number = 0 #sert à remplir data_to_plot
            for model_index , transfo_index in itertools.product(range(len(all_models)), range(len(all_transfo))) :

                model, transfo = all_models[model_index], all_transfo[transfo_index]

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


                list_carac_for_spearmanr = np.array(list_carac_for_spearmanr) 
                list_deltametric_for_spearmanr = np.array(list_deltametric_for_spearmanr) 

                # On retire les valeurs nulles : FACULTATIF
                list_carac_for_spearmanr = list_carac_for_spearmanr[list_deltametric_for_spearmanr != 0.0]
                list_deltametric_for_spearmanr = list_deltametric_for_spearmanr[list_deltametric_for_spearmanr != 0.0]

                #Si on veut faire le graphe en log : FACULTATIF
                list_carac_for_spearmanr = np.log(list_carac_for_spearmanr + 1)
                #list_deltametric_for_spearmanr = np.log(list_deltametric_for_spearmanr + 1)


                # Add data to panda dataframe for Spearman’s Rank Correlation Coefficients
                data_to_plot.loc[index_number] = [model, transfo, stats.spearmanr(list_deltametric_for_spearmanr,list_carac_for_spearmanr).statistic]
                index_number += 1

                #Plot cloud points
                axs[model_index, transfo_index].scatter(list_carac_for_spearmanr, list_deltametric_for_spearmanr, s = 3)
                if model_index == 0 :
                    axs[model_index, transfo_index].set_title(f"{transfo}" , fontdict = {'fontweight' : "bold"})
                if transfo_index == 0 :
                    #Set y label in bold
                    axs[model_index, transfo_index].set_ylabel(f"{model}", fontdict=dict(weight='bold'))

            plt.savefig(f"impact/{metric_name}/{c}/cloud.png", dpi = 200)
            plt.close()


                
            g = sns.catplot(
                data=data_to_plot, kind="bar",
                x="Model", y="Delta_acc", hue="Transformation",
                errorbar="sd", palette="dark", alpha=.6, height=6,
                legend_out = True
            )

            g.despine(left=True)
            g.set_axis_labels("Classifier model", f"Spearman's Rank Correlation")
            g.legend.set_title(f"{c} ({metric_name})")

        
            plt.savefig(f"impact/{metric_name}/{c}/SRC.png", dpi = 200) 
        
        plt.close("all")



        
        #### ===================================== ####
        #### ===================================== ####
        #### ========= CLOUD BI-VARIABLES ======== ####
        #### ===================================== ####
        #### ===================================== ####

        if not os.path.exists(f"impact/{metric_name}/multi_carac") :
            os.makedirs(f"impact/{metric_name}/multi_carac")

        #Construit un panda Dataframe à l'aide de delta_metric et de charac_lists où chaque ligne correspond à un dataset et les colonnes sont : model ,transfo, carac1, carac2, Delta_acc , Delta_metric
        all_info = pd.DataFrame()

        for dataset, model_dict in delta_metric.items():
            for model, method_dict in model_dict.items():
                for transfo, delta_acc in method_dict.items():
                    row = { 'dataset': dataset, 'model': model, 'transfo': transfo, f'delta_{metric_name}': delta_acc}

                    for charac, dataset_dict in charac_lists.items():
                        charac_value = dataset_dict.get(dataset)
                        row[charac] = np.log(charac_value + 1)

                    all_info = all_info._append(row, ignore_index=True)

        median = np.quantile(np.array(all_info[f'delta_{metric_name}'] ), 0.5)
        hue_min, hue_max = np.quantile(np.array(all_info[f'delta_{metric_name}'] ), 0.1), np.quantile(np.array(all_info[f'delta_{metric_name}'] ), 0.9)
        if median > 0 :
            hue_min = -hue_max
        else : 
            hue_max = -hue_min
        norm = plt.Normalize(hue_min, hue_max)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        
        
        #On test tous les couples de caractéristiques        
        for i in tqdm(range(len(DATASET_CHARACTERISTICS)) , desc = "Multi-carac cloud plot") :
            for j in range(i) :
                c1, c2 = DATASET_CHARACTERISTICS[i], DATASET_CHARACTERISTICS[j]

                fig, axs = plt.subplots(len(all_models), len(all_transfo), figsize=(20, 12))
                fig.suptitle(f"Cloud plot : log({c1}+1), log({c2}+1) VS Delta {metric_name}", fontsize=20)
                fig.subplots_adjust(right=0.8)
                cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                fig.colorbar(sm, ax = cbar_ax)

                for model_index , transfo_index in itertools.product(range(len(all_models)), range(len(all_transfo))) :

                    model, transfo = all_models[model_index], all_transfo[transfo_index]

                    sub_df = all_info[(all_info['model'] == model) & (all_info['transfo'] == transfo)] 
                    
                    a = sns.scatterplot( data = sub_df , x = c1 , y = c2, hue = f'delta_{metric_name}', ax = axs[model_index, transfo_index], legend = False, palette = "coolwarm", hue_norm=(hue_min, hue_max), s = 10) 
                    axs[model_index, transfo_index].set_ylabel("")


                    if model_index == 0 :
                        axs[model_index, transfo_index].set_title(f"{transfo}" , fontdict = {'fontweight' : "bold"})
                    if transfo_index == 0 :
                        #Set y label in bold
                        axs[model_index, transfo_index].set_ylabel(f"{model}", fontdict=dict(weight='bold'))

                plt.savefig(f"impact/{metric_name}/multi_carac/{c1}_???_{c2}.png", dpi = 200) 
                plt.close("all")

