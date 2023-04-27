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
from params import *



#Pour cacher les prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def usage():
    msg = """
python make_tests.py <ID_GROUP>

Example : python make_tests.py 1

<ID_GROUP> :
Dans 'params.py', on retrouve la liste des datasets de test (DATASETS_TO_TEST). 
Pour éviter de faire tous les tests d'un coup, on split cette liste en groupes de taille group_size (réglable dans 'params.py').
<ID_GROUP> correspond simplement à l'idée du groupe. Ainsi, <ID_GROUP> = 1 correspond au group_size premier datasets, <ID_GROUP> = 2 au group_size suivants, etc.

Si <ID_GROUP> = 0, alors on ne fait aucun groupe et on prend tout le dataset
        """
    print(msg)
    exit()


if __name__ == "__main__" :

    #Récupère l'id du groupe donné en argument par l'utilisateur
    try :
        id_group = int(sys.argv[1])
    except Exception as e:
        raise_usage()
    if id_group > 0 :
        DATASETS_TO_TEST = DATASETS_TO_TEST[(id_group - 1)*group_size: id_group*group_size]


    delta_metric = {} #Dictionnaire {dataset_name : {model : {DA : Delta_Acc}} } qui va contenir Delta_Acc ou Delta_F1 (i.e Delta_summary_metric)
    charac_lists = { c : {} for c in DATASET_CHARACTERISTICS} #Dictionnaire {characteristiques : {dataset : valeur}}. Va permettre de récupérer dans la foulée les charactéristiques des datasets dans le fichier info.csv

    #Ouvre le fichier d'info
    infos = pd.read_csv("infos.csv",sep=',')

    print(DATASETS_TO_TEST)
    for dataset_name in DATASETS_TO_TEST :

        #Sauvegarde delta_metric temp et charac_lists temp
        with open(f"tmp/delta_metric{id_group}.pickle", 'wb') as handle:
            pickle.dump(delta_metric, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'tmp/charac_lists{id_group}.pickle', 'wb') as handle:
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
        
