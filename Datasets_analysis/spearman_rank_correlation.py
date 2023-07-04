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
from utils import get_charac_and_metric



if __name__ == "__main__" :

    #Créer le dossier pour stocker les impacts
    if not os.path.exists("impact") :
        os.makedirs("impact")

    #Récupère les caractéristiques et les delta_metric
    #charac_lists de la forme : {carac1 : {dataset1 : valeur, dataset2 : valeur, ...}, carac2 : {dataset1 : valeur, dataset2 : valeur, ...}, ...}
    #global_delta_metric de la forme : { Metric1 : {dataset1 : {model1 : {method_DA1 : Delta_Acc, method_DA2 : Delta_Acc, ...}, ...} ...} ...}
    global_delta_metric, charac_lists, effective_test_dataset, all_models, all_transfo = get_charac_and_metric(delta = True)

    for metric_name in summary_metric :

        print(f"\n================== IMPACT SUR {metric_name} ==================")

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

            data_to_plot = pd.DataFrame(columns= ["Model", "Transformation", "Correlation"])

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
                x="Model", y="Correlation", hue="Transformation",
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

