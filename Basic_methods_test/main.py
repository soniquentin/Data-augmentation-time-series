import sys
from get_info import get_info, get_datasets_infos
from tools import *
from train import *
import pandas as pd
import os, sys
import time
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


### Listes pour le multitest
MULTI_TEST_DATASETS = ["ECGFiveDays", "Wafer", "Earthquakes", "Lightning2", "NonInvasiveFetalECGThorax2"]
MULTI_TEST_MODELS = ["NN","RF","TS-RF", "DTW_NEIGBOURS"] #["RF","NN","DTW_NEIGBOURS","TS-RF"]
ITERATIONS = [20,20,20,5]


#Pour cacher les prints
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def raise_usage():
    usage = """
Run command : python main.py <OPTION>

<OPTION> :
    --get_info_dataset : affiche infos des fichiers d'entrainement (nb de samples, compte pour chaque label, imbalence degree).
    --make_tab <DATASET_NAME> <MODEL>("RF","NN","DTW_NEIGBOURS","TS-RF") <NB_ITERATION> : Construit les tableaux pour un dataset et un model donné
    --multi_test : make_tab itéré sur différents datasets et différents model (modifier les listes dans le fichier main.py)
    --draw_example_minority <DATASET_NAME> : plot min(3,cnt_minority) exemples de la time series de la classe minoritaire du dataset
            """
    print(usage)
    exit()


def make_tab_section(dataset_name, model_name, nb_iteration):
    dataset_folder = "./datasets"

    data = pd.read_csv(dataset_folder + "/{}/{}_TRAIN.tsv".format(dataset_name, dataset_name) ,sep='\t', header =None)
    data_test = pd.read_csv(dataset_folder + "/{}/{}_TEST.tsv".format(dataset_name, dataset_name) ,sep='\t', header =None)
    score_matrix = make_score_test(data, data_test, dataset_name, model_name = model_name, nb_iteration = nb_iteration)

    final_tab_mean, final_tab_p_value = make_final_tab(score_matrix)

    #Save the file
    final_tab_mean.to_csv("results/Mean_{}_{}.csv".format(dataset_name, model_name), sep='\t')
    final_tab_p_value.to_csv("results/Pvalues_{}_{}.csv".format(dataset_name, model_name), sep='\t')


def multi_test_section():
    for i in range(len(MULTI_TEST_DATASETS)) :
        for j in range(len(MULTI_TEST_MODELS)) :

            model_name =  MULTI_TEST_MODELS[j]
            dataset_name = MULTI_TEST_DATASETS[i]
            nb_iteration = ITERATIONS[j]
            
            print("Dataset {} and model {}...".format(dataset_name, model_name))
            print("    --> Training and testing...")

            t_i = time.time()
            with HiddenPrints() :
                make_tab_section(dataset_name, model_name, nb_iteration)
            
            print("  --> Done ! (duration : {} s)".format(time.time() - t_i))


def draw_example_minority_section(dataset_name):
    dict_info = get_datasets_infos()
    for k in dict_info :
        if dataset_name in k : 
            path_file = dict_info[k]["filepath"]
            data = pd.read_csv(path_file,sep='\t', header =None) 
            unique_labels = np.sort( data[0].unique() )
            count_label = np.array( [ len(data[data[0] == label].index) for label in unique_labels ] )
            indice_min = np.argmin(count_label)
            min_label_count, label_min = np.min(count_label), unique_labels[indice_min]

            data_min = np.array( data[data[0] == label_min] )[:,1:]

            to_display = min(3,min_label_count)

            fig, axs = plt.subplots(to_display)
            for i in range(to_display) :
                axs[i].plot(data_min[i])
            plt.show()




if __name__ == "__main__" :

    try :
        option = sys.argv[1]
    except Exception as e:
        raise_usage()

    if "--get_info_dataset" in option :
        get_info()
    elif "--make_tab" in option :
        try :
            dataset_name = sys.argv[2]
            model_name = sys.argv[3]
            nb_iteration = int(sys.argv[4])
        except Exception as e:
            raise_usage()  
        make_tab_section(dataset_name, model_name, nb_iteration)
    elif "--multi_test" in option :
        multi_test_section()
    elif "--draw_example_minority" :
        try :
            dataset_name = sys.argv[2]
        except Exception as e:
            raise_usage()  
        draw_example_minority_section(dataset_name)
    else :
        raise_usage()

        