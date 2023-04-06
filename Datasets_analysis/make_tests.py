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
                  ("TS-RF",20)
                  #("DTW_NEIGBOURS",5)
                  ]

summary_metric = "F1"
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

    for dataset_name in DATASETS_TO_TEST :

        #Retrouve le chemin du dataset
        infos = pd.read_csv("infos.csv".format(dataset_name, dataset_name) ,sep=',')
        for index, row in infos.iterrows():
            if f"{dataset_name}_TRAIN.tsv" == row["Name"] :
                file_parent = str(Path(row["Filepath"]).parent.absolute())
                break

        #Datafinal bilan (pour le bar chart final)
        data_final = pd.DataFrame()

        #Check que le dossier du dataset exist pour pouvoir y mettre les résultats
        dataset_folder = f"tests/{dataset_name}"
        if not os.path.exists(dataset_folder) :
            os.makedirs(dataset_folder)

        for model, nb_iteration in MODELS_TO_TEST :

            #Verbose
            print(f"****** [{dataset_name}] testing on {model} *******")

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

        #Construit la group bar chart bilan
        plt.figure(figsize=(30, 10), dpi=200) 
        g = sns.catplot(
            data=data_final, kind="bar",
            x="Model", y=summary_metric, hue="Transformation",
            errorbar="sd", palette="dark", alpha=.6, height=6
        )

        g.despine(left=True)
        g.set_axis_labels("Classifier model", f"{summary_metric}")
        g.legend.set_title(f"{dataset_name}")

        plt.savefig(dataset_folder + f"/summary_barchart.png")
