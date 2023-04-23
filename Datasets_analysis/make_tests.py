"""
Effectue les tests en utilisant les fonctions de dans Basic_methods_test/
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



## ========== INFO TO MODIFY =========== ##
DATASETS_TO_TEST = [
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

#(Model, nb_iterations)
MODELS_TO_TEST = [("NN",20),
                  ("RF",20),
                  ("TS-RF",20),
                  #("DTW_NEIGBOURS",3),
                  ("KERNEL",20),
                  ("SHAPELET",20)
                  ]

summary_metric = "F1"

#Les caractéristiques de dataset dont il faut analyser l'influence (correspond au nom des colonnes dans info.csv)
DATASET_CHARACTERISTICS = [
                           "Length",
                           "Dataset size",
                           "Avg label size",
                           "Dataset variance",
                           "Intra-class variance",
                          ]
## ===================================== ##




#Pour cacher les prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout




if __name__ == "__main__" :

    delta_metric = {} #Dictionnaire {dataset_name : {model : {DA : Delta_Acc}} } qui va contenir Delta_Acc ou Delta_F1 (i.e Delta_summary_metric)
    charac_lists = { c : {} for c in DATASET_CHARACTERISTICS} #Dictionnaire {characteristiques : {dataset : valeur}}. Va permettre de récupérer dans la foulée les charactéristiques des datasets dans le fichier info.csv

    #Ouvre le fichier d'info
    infos = pd.read_csv("infos.csv",sep=',')

    for dataset_name in DATASETS_TO_TEST :

        #Sauvegarde delta_metric temp et charac_lists temp
        with open('tmp/delta_metric.pickle', 'wb') as handle:
            pickle.dump(delta_metric, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('tmp/charac_lists.pickle', 'wb') as handle:
            pickle.dump(charac_lists, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
        #Initialise le dictionnaire (qui est la valeur associée à la clé dataset_name dans delta_metric)
        delta_metric[dataset_name] = {}

        #Retrouve le chemin du dataset et en profite pour pour importer les characteristiques des datasets
        for index, row in infos.iterrows():
            if f"{dataset_name}_TRAIN.tsv" == row["Name"] :
                file_parent = str(Path(row["Filepath"]).parent.absolute())
                for c in DATASET_CHARACTERISTICS : 
                    charac_lists[c][dataset_name] = row[c]
                break

        #Datafinal bilan (pour le bar chart final)
        data_final = pd.DataFrame()

        #Check que le dossier du dataset exist pour pouvoir y mettre les résultats
        dataset_folder = f"tests/{dataset_name}"
        if not os.path.exists(dataset_folder) :
            os.makedirs(dataset_folder)

        for model, nb_iteration in MODELS_TO_TEST :

            with HiddenPrints() :   
                #Entraine et fais les tests
                data = pd.read_csv(file_parent + f"/{dataset_name}_TRAIN.tsv" ,sep='\t', header =None)
                data_test = pd.read_csv(file_parent + f"/{dataset_name}_TEST.tsv",sep='\t', header =None)
                score_matrix = make_score_test(data, data_test, dataset_name, model_name = model, nb_iteration = nb_iteration)
                
                #Transforme les résultats des tests en tableaux (mean et p-value) et sauvegarde les plots + les tableaux
                model_foler = f"tests/{dataset_name}/{model}"
                if not os.path.exists(model_foler) :
                    os.makedirs(model_foler)
                final_tab_mean, final_tab_p_value = make_final_tab(score_matrix, save_plot_path = model_foler)
                final_tab_mean.to_csv(model_foler + f"/Mean.csv", sep='\t')
                final_tab_p_value.to_csv(model_foler + f"/Pvalues.csv", sep='\t')

                #Concatène le dataframe de score pour le rendu du bar chart final
                data_final = pd.concat([data_final,score_matrix])

                #Ajoute les moyennes dans delta_metric
                delta_metric[dataset_name][model] = {}
                default_score = final_tab_mean.loc["Default"][summary_metric]
                for index, row in final_tab_mean.iterrows():
                    if index != "Default" :
                        delta_metric[dataset_name][model][index] = row[summary_metric] - default_score


        #Construit la group bar chart bilan
        plt.figure(figsize=(30, 10)) 
        g = sns.catplot(
            data=data_final, kind="bar",
            x="Model", y=summary_metric, hue="Transformation",
            errorbar="sd", palette="dark", alpha=.6, height=6, legend_out = True
        )

        g.despine(left=True)
        g.set_axis_labels("Classifier model", f"{summary_metric}")
        g.legend.set_title(f"{dataset_name}")

        plt.savefig(dataset_folder + f"/summary_barchart.png", dpi=200)
        plt.close("all")
        
    
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