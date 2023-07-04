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
import traceback



#Pour cacher les prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout



def log_error(dataset_name, info, id_group):
    """
        Ecrit l'erreur dans le fichier log.txt
    """

    #Si une erreur est levée, on l'écrit dans le fichier log.txt
    if os.path.exists("tests/log.txt") :
        with open(f"tests/log.txt", "a") as f :
            f.write(f"\n\n\n===={dataset_name} (group {id_group}): {info}====\n")
            f.write(traceback.format_exc())
    else :
        with open(f"tests/log.txt", "w") as f :
            f.write(f"===={dataset_name} : {info}====\n")
            f.write(traceback.format_exc())


def raise_usage():
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


    #Ouvre le fichier d'info
    infos = pd.read_csv("infos.csv",sep=',')

    print(f"\n\n\nDatasets testés : {DATASETS_TO_TEST}\n\n\n")
    for dataset_name in DATASETS_TO_TEST :

        #Retrouve le chemin du dataset
        for index, row in infos.iterrows():
            if f"{dataset_name}_TRAIN.tsv" == row["Name"] :
                file_parent = str(Path(row["Filepath"]).parent.absolute())

        #Datafinal bilan (pour le bar chart final)
        data_final = pd.DataFrame()

        #Check que le dossier du dataset exist pour pouvoir y mettre les résultats
        dataset_folder = f"tests/{dataset_name}"
        if not os.path.exists(dataset_folder) :
            os.makedirs(dataset_folder)

        for model_dict in MODELS_TO_TEST :

            model = model_dict["Name"]
            nb_iteration = model_dict["Iterations"]
            takes = model_dict["Make Test"]

            if takes : 
                try : 
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

                except Exception :
                    log_error(dataset_name, model, id_group)

        try : 
            for metric_name in summary_metric :
            #Construit la group bar chart bilan
                plt.figure(figsize=(30, 10)) 
                g = sns.catplot(
                    data=data_final, kind="bar",
                    x="Model", y=metric_name, hue="Transformation",
                    errorbar="sd", palette="dark", alpha=.6, height=6, legend_out = True
                )

                g.despine(left=True)
                g.set_axis_labels("Classifier model", f"{metric_name}")
                g.legend.set_title(f"{dataset_name}")

                plt.savefig(dataset_folder + f"/summary_barchart_{metric_name}.png", dpi=200)
                plt.close("all")
        except Exception :
            log_error(dataset_name, f"summary_barchart {metric_name}", id_group)
            
