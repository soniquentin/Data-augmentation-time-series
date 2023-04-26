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

    
    f = open("tmp/charac_lists1.pickle",'rb')
    charac_lists = pickle.load(f)
    f.close()

    f = open("tmp/delta_metric1.pickle",'rb')
    delta_metric = pickle.load(f)
    f.close()

    for i in range(2,6) :
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
